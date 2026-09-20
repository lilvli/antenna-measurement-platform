from __future__ import annotations

import asyncio
import math
import re
import uuid
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np

from antenna_service.devices.base import DeviceAdapter, RtcDeviceAdapter, azimuth_scan_geometry
from antenna_service.errors import ServiceError, invalid
from antenna_service.events import EventBus
from antenna_service.models import DeviceSource, DeviceState
from antenna_service.protocol.crc import verify_crc_be
from antenna_service.protocol.flash import (
    PAGE_READ_OPCODE,
    PAGE_SIZE,
    PAGE_WRITE_OPCODE,
    decode_page_response,
    encode_page_read,
    encode_page_write,
    response_has_identity,
)
from antenna_service.protocol.rtc import RtcClient, RtcEndpoint


class RealBeamController(DeviceAdapter):
    """Binary-safe serial transport with independent transmit and receive paths.

    The reader stays active for the whole connection.  Manual debugging therefore never
    consumes or flushes a reply, while an automatic run can still wait for the next complete
    antenna frame through ``send_frame``.  Every received byte sequence is published before
    optional profile parsing so unmatched vendor/debug data remains visible to the operator.
    """

    def __init__(
        self,
        port: str,
        baud_rate: int = 115200,
        *,
        events: EventBus | None = None,
        frame_decoder: Callable[[bytes], dict[str, Any] | None] | None = None,
    ) -> None:
        super().__init__("beam_controller", DeviceSource.REAL)
        self.port = port
        self.baud_rate = baud_rate
        self._events = events
        self._frame_decoder = frame_decoder
        self._serial: Any = None
        self._write_lock = asyncio.Lock()
        self._reader_task: asyncio.Task[None] | None = None
        self._closing = False
        self._receive_buffer = bytearray()
        self._receive_last_byte_at = time.monotonic()
        self._response_waiters: deque[tuple[asyncio.Future[bytes], Callable[[bytes], bool]]] = deque()
        self._exchange_lock = asyncio.Lock()

    async def connect(self) -> dict[str, Any]:
        self.snapshot.update(state=DeviceState.CONNECTING)
        try:
            import serial

            # 真实串口按 8N1 打开，禁用软/硬流控。timeout=0.1 只限制一次后台读取，
            # 不代表协议应答超时；协议超时由 send_frame 的 timeout_ms 单独控制。
            self._serial = await asyncio.to_thread(
                serial.Serial,
                self.port,
                self.baud_rate,
                timeout=0.1,
                write_timeout=1.0,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
            )
            await asyncio.to_thread(self._serial.reset_input_buffer)
            await asyncio.to_thread(self._serial.reset_output_buffer)
        except Exception as exc:
            self.snapshot.update(state=DeviceState.FAULT, error=str(exc))
            raise ServiceError("DEVICE_FAULT", "波控机串口打开失败", "beam_connect", self.port, {"error": str(exc)}) from exc
        self.snapshot.identity = f"SERIAL:{self.port}@{self.baud_rate}"
        # Protocol initialization is performed with the loaded profile before a run starts.
        self.snapshot.update(state=DeviceState.CONNECTED)
        self._closing = False
        self._reader_task = asyncio.create_task(
            self._receive_loop(), name=f"beam-rx-{self.port}"
        )
        return self.snapshot.as_dict()

    async def disconnect(self) -> None:
        # 先终止后台接收任务，再关闭串口句柄；顺序不可反转，否则读取线程可能
        # 在句柄关闭后继续访问驱动。尚未完成的自动流程等待者会收到明确断开错误。
        self._closing = True
        if self._reader_task is not None:
            try:
                await asyncio.wait_for(self._reader_task, timeout=0.3)
            except (TimeoutError, asyncio.CancelledError):
                self._reader_task.cancel()
                try:
                    await self._reader_task
                except asyncio.CancelledError:
                    pass
            self._reader_task = None
        if self._serial is not None:
            await asyncio.to_thread(self._serial.close)
            self._serial = None
        while self._response_waiters:
            waiter, _ = self._response_waiters.popleft()
            if not waiter.done():
                waiter.set_exception(
                    ServiceError("NOT_RUNNABLE", "波控机串口已断开", "beam_receive", self.port)
                )
        self._receive_buffer.clear()
        self.snapshot.update(state=DeviceState.DISCONNECTED)

    def _validate_outbound_frame(self, frame: bytes) -> None:
        if self._serial is None:
            raise ServiceError("NOT_RUNNABLE", "波控机串口未连接", "beam_send", self.port)
        if len(frame) not in {22, 286} or frame[:2] != b"\xAA\x55" or int.from_bytes(frame[2:4], "big") != len(frame):
            raise invalid("波控发送边界只接受已编译的 22 或 286 字节 AA55 帧", stage="beam_send", target=self.port)

    async def send_only(self, frame: bytes) -> None:
        """Write one compiled frame without clearing input or waiting for a response."""
        self._validate_outbound_frame(frame)
        async with self._write_lock:
            try:
                # serial.write 是对真实波控机产生副作用的边界。写锁保证一帧不会
                # 与另一帧交叉；写后 flush 只等待操作系统发完 TX，不清空 RX。
                # Never reset the input buffer here. The background reader owns RX and every
                # byte received from the real device must remain observable in the raw log.
                written = await asyncio.to_thread(self._serial.write, frame)
                await asyncio.to_thread(self._serial.flush)
                if written != len(frame):
                    raise IOError(f"只写入 {written}/{len(frame)} 字节")
            except ServiceError:
                raise
            except Exception as exc:
                raise ServiceError(
                    "DEVICE_FAULT",
                    "波控机串口收发失败",
                    "beam_send",
                    self.port,
                    {"error": str(exc), "request": frame.hex(" ").upper()},
                    side_effect_possible=True,
                    next_action="请读取设备状态；不要自动重发可能已执行的命令",
                ) from exc

    async def send_frame(
        self,
        frame: bytes,
        *,
        timeout_ms: int = 1000,
        response_opcode: int | None = None,
        response_rule: Any = None,
    ) -> bytes | list[bytes]:
        """Send and await the CRC-valid response correlated to opcode and array ID."""
        self._validate_outbound_frame(frame)
        expected_opcode = frame[4] if response_opcode is None else int(response_opcode)
        expected_array_id = frame[5]
        assembly = None
        multi = response_rule is not None and response_rule.mode == "MULTI"
        if multi:
            timeout_ms = min(timeout_ms, response_rule.timeout_ms)

        def matches(candidate: bytes) -> bool:
            nonlocal assembly
            if not (
                len(candidate) == 22
                and candidate[4] == expected_opcode
                and candidate[5] == expected_array_id
                and verify_crc_be(candidate)
            ):
                return False
            if multi:
                from antenna_service.protocol.profile import ResponseAssembly

                if assembly is None:
                    assembly = ResponseAssembly(response_rule)
                return assembly.add(candidate)
            return True

        response = await self._exchange(
            frame,
            matcher=matches,
            timeout_ms=timeout_ms,
            stage="beam_wait_response",
        )
        return assembly.frames if assembly is not None else response

    async def _exchange(
        self,
        frame: bytes,
        *,
        matcher: Callable[[bytes], bool],
        timeout_ms: int,
        stage: str,
        log_context: str | None = None,
    ) -> bytes:
        """Register a correlated waiter before writing one side-effecting request.

        The exchange lock prevents an automatic antenna command and a FLASH request
        from competing for the same response.  A timeout never retransmits the frame:
        the caller must inspect device state or use the protocol's explicit readback
        recovery path because the original operation may already have taken effect.
        """

        self._validate_outbound_frame(frame)
        waiter: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
        entry = (waiter, matcher)
        async with self._exchange_lock:
            self._response_waiters.append(entry)
            try:
                await self.send_only(frame)
                if log_context is not None:
                    await self._publish_transmitted(frame, context=log_context)
                return await asyncio.wait_for(waiter, timeout=timeout_ms / 1000)
            except TimeoutError as exc:
                raise ServiceError(
                    "TIMEOUT",
                    "波控机应答超时",
                    stage,
                    self.port,
                    side_effect_possible=True,
                    next_action="不要盲目重发；FLASH 页写请先执行同地址读回",
                ) from exc
            finally:
                try:
                    self._response_waiters.remove(entry)
                except ValueError:
                    pass

    async def _receive_loop(self) -> None:
        """Continuously collect physical RX bytes until the explicit disconnect action."""
        while not self._closing and self._serial is not None:
            try:
                available = int(getattr(self._serial, "in_waiting", 0) or 0)
                chunk = await asyncio.to_thread(self._serial.read, max(1, available))
                if chunk:
                    # Log physical bytes immediately and exactly once, including partial
                    # or corrupt frames. Decoding below publishes fields without repeating HEX.
                    await self._publish_received(chunk, parsed=None)
                    self._receive_last_byte_at = time.monotonic()
                    self._receive_buffer.extend(chunk)
                await self._drain_receive_buffer()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._closing:
                    break
                self.snapshot.update(state=DeviceState.FAULT, error=str(exc))
                for waiter, _ in tuple(self._response_waiters):
                    if not waiter.done():
                        waiter.set_exception(ServiceError(
                            "DEVICE_FAULT", "波控机后台接收已中断", "beam_receive", self.port,
                            {"error": str(exc)}, side_effect_possible=True,
                        ))
                if self._events is not None:
                    await self._events.publish(
                        "device.raw",
                        device_id="beam_controller",
                        direction="ERROR",
                        command_id=None,
                        raw_hex="",
                        transport="SERIAL",
                        context="BACKGROUND_RECEIVE",
                        error=f"串口接收失败：{exc}",
                    )
                break

    async def _drain_receive_buffer(self) -> None:
        """Recover framing without dropping a CRC-valid frame behind corrupt bytes.

        Only 22-byte antenna replies and 286-byte FLASH replies are supported. An
        incomplete plausible frame is allowed an inter-byte gap; after that gap we
        seek a complete CRC-valid successor rather than keeping stale bytes forever.
        Physical RX logging belongs exclusively to the reader, not to this parser.
        """
        expired = time.monotonic() - self._receive_last_byte_at > max(0.3, 2860 / self.baud_rate)
        while self._receive_buffer:
            header = self._receive_buffer.find(b"\xAA\x55")
            if header < 0:
                keep = 1 if not expired and self._receive_buffer[-1:] == b"\xAA" else 0
                self._receive_buffer[:] = self._receive_buffer[-keep:] if keep else b""
                return
            if header > 0:
                del self._receive_buffer[:header]
            if len(self._receive_buffer) < 4:
                if expired:
                    self._receive_buffer.clear()
                return
            frame_length = int.from_bytes(self._receive_buffer[2:4], "big")
            if frame_length not in {22, 286}:
                del self._receive_buffer[:1]
                continue
            if len(self._receive_buffer) < frame_length:
                if not expired:
                    return
                del self._receive_buffer[:1]
                continue
            frame = bytes(self._receive_buffer[:frame_length])
            if not verify_crc_be(frame):
                # Slide one byte, retaining a subsequent frame header even when the
                # corrupt length would otherwise consume part of the next reply.
                del self._receive_buffer[:1]
                continue
            del self._receive_buffer[:frame_length]
            try:
                parsed = self._frame_decoder(frame) if self._frame_decoder is not None else None
            except Exception:
                # Parsing must never stop physical reception. The raw frame is still truthful
                # and remains visible even when a profile implementation rejects it.
                parsed = None
            if parsed is not None and self._events is not None:
                await self._events.publish(
                    "device.raw", device_id="beam_controller", direction="RX",
                    command_id=parsed.get("command_id"), raw_hex="", transport="SERIAL",
                    context="PROFILE_DECODE", parsed=parsed,
                )
            # Only the waiter whose request identity matches may consume this frame.
            # Unmatched frames stay visible in the raw log but cannot complete another
            # command's transaction.
            for entry in tuple(self._response_waiters):
                waiter, matcher = entry
                if waiter.done():
                    continue
                try:
                    matched = matcher(frame)
                except Exception as exc:
                    self._response_waiters.remove(entry)
                    waiter.set_exception(exc)
                    break
                if not matched:
                    continue
                self._response_waiters.remove(entry)
                waiter.set_result(frame)
                break

    async def _publish_received(self, payload: bytes, *, parsed: dict[str, Any] | None) -> None:
        if not payload or self._events is None:
            return
        event: dict[str, Any] = {
            "device_id": "beam_controller",
            "direction": "RX",
            "command_id": parsed.get("command_id") if parsed else None,
            "raw_hex": payload.hex(" ").upper(),
            "transport": "SERIAL",
            "context": "BACKGROUND_RECEIVE",
        }
        if parsed is not None:
            event["parsed"] = parsed
        await self._events.publish("device.raw", **event)

    async def _publish_transmitted(self, payload: bytes, *, context: str) -> None:
        if self._events is None:
            return
        await self._events.publish(
            "device.raw",
            device_id="beam_controller",
            direction="TX",
            command_id=None,
            raw_hex=payload.hex(" ").upper(),
            transport="SERIAL",
            context=context,
        )

    async def flash_write(self, tile_id: int, address: int, payload: bytes) -> None:
        """Write one 256-byte page and require the correlated 22-byte acknowledgement.

        This is the physical device side-effect boundary.  Address is a 24-bit byte
        address aligned to 0x100; ``tile_id`` selects the target array/FLASH space.
        The request is sent once only.  Timeout or invalid acknowledgement is returned
        to the workflow with ``side_effect_possible`` so it can read the same address
        instead of blindly repeating a write that may already have succeeded.
        """

        try:
            request = encode_page_write(tile_id, address, payload)
        except ValueError as exc:
            raise invalid(str(exc), stage="flash_write", target=self.port) from exc
        response = await self._exchange(
            request,
            matcher=lambda frame: response_has_identity(frame, PAGE_WRITE_OPCODE, tile_id, address),
            timeout_ms=2000,
            stage="flash_write_ack",
            log_context="FLASH_PAGE_WRITE",
        )
        try:
            decode_page_response(response, PAGE_WRITE_OPCODE, tile_id, address)
        except ValueError as exc:
            raise ServiceError(
                "DATA_INTEGRITY",
                f"FLASH 页写应答无效：{exc}",
                "flash_write_ack",
                self.port,
                {"tile_id": tile_id, "address": address},
                side_effect_possible=True,
                next_action="不要重发页写；请先读取同一 tile_id 和地址",
            ) from exc

    async def flash_read(self, tile_id: int, address: int, length: int = PAGE_SIZE) -> bytes:
        """Read one complete page; retry policy belongs to the calling workflow."""

        if length != PAGE_SIZE:
            raise invalid("真实 FLASH 读回长度必须为 256 字节", stage="flash_read", target=self.port)
        try:
            request = encode_page_read(tile_id, address)
        except ValueError as exc:
            raise invalid(str(exc), stage="flash_read", target=self.port) from exc
        response = await self._exchange(
            request,
            matcher=lambda frame: response_has_identity(frame, PAGE_READ_OPCODE, tile_id, address),
            timeout_ms=2000,
            stage="flash_read_response",
            log_context="FLASH_PAGE_READ",
        )
        try:
            return decode_page_response(response, PAGE_READ_OPCODE, tile_id, address)
        except ValueError as exc:
            raise ServiceError(
                "DATA_INTEGRITY",
                f"FLASH 页读应答无效：{exc}",
                "flash_read_response",
                self.port,
                {"tile_id": tile_id, "address": address},
            ) from exc


class RealVna(DeviceAdapter):
    """Keysight-compatible VISA adapter exposing only structured measurement actions."""

    def __init__(self, resource: str, backend: str | None = None, timeout_ms: int = 10000) -> None:
        super().__init__("vna", DeviceSource.REAL)
        self.resource = resource
        self.backend = backend
        self.timeout_ms = timeout_ms
        self._manager: Any = None
        self._instrument: Any = None
        self._lock = asyncio.Lock()
        self._configured_frequencies: np.ndarray | None = None
        self._trigger_mode = "INTERNAL_SINGLE"
        self._averaging_enabled = False
        self._averaging_count = 1
        self._sweep_time_seconds = 0.0
        self._measurement_number: int | None = None
        self._configured_s_parameter: str | None = None
        self._display_window_number: int | None = None
        self._display_trace_number: int | None = None
        self._display_needs_autoscale = False
        self._buffer_plan: dict[str, Any] | None = None
        self._buffer_memory_name: str | None = None

    @staticmethod
    def _is_no_error(value: str) -> bool:
        try:
            return int(float(value.split(",", 1)[0].strip())) == 0
        except (TypeError, ValueError):
            return False

    def _drain_error_queue_blocking(self, limit: int = 20) -> list[str]:
        errors: list[str] = []
        for _ in range(limit):
            value = str(self._instrument.query("SYST:ERR?")).strip()
            errors.append(value)
            if self._is_no_error(value):
                break
        return errors

    @classmethod
    def _fault_errors(cls, errors: list[str]) -> list[str]:
        return [value for value in errors if not cls._is_no_error(value)]

    @staticmethod
    def _measurement_catalog(value: str) -> dict[str, str]:
        fields = [field.strip().strip("\"") for field in value.strip().strip("\"").split(",")]
        return {fields[index]: fields[index + 1] for index in range(0, len(fields) - 1, 2)}

    @staticmethod
    def _number_catalog(value: str) -> list[int]:
        text = value.strip().strip("\"")
        if not text or text.upper() == "EMPTY":
            return []
        return [int(float(field.strip())) for field in text.split(",")]

    async def _ensure_measurement_visible(self) -> dict[str, Any]:
        """Bind ANTENNA_MEAS to one visible trace without replacing operator traces."""

        window_number: int | None = None
        trace_number: int | None = None
        windows_text = str(await asyncio.to_thread(self._instrument.query, "DISP:CAT?")).strip()
        windows = self._number_catalog(windows_text)
        for candidate_window in windows:
            traces_text = str(
                await asyncio.to_thread(self._instrument.query, f"DISP:WIND{candidate_window}:CAT?")
            ).strip()
            for candidate_trace in self._number_catalog(traces_text):
                await asyncio.to_thread(
                    self._instrument.write,
                    f"DISP:WIND{candidate_window}:TRAC{candidate_trace}:SEL",
                )
                active_measurement = str(
                    await asyncio.to_thread(self._instrument.query, "SYST:ACT:MEAS?")
                ).strip().strip("\"")
                if active_measurement == "ANTENNA_MEAS":
                    window_number = candidate_window
                    trace_number = candidate_trace
                    break
            if window_number is not None:
                break

        if window_number is None:
            window_number = windows[0] if windows else 1
            if not windows:
                await asyncio.to_thread(self._instrument.write, f"DISP:WIND{window_number}:STAT ON")
            trace_number = int(
                float(
                    await asyncio.to_thread(
                        self._instrument.query,
                        f"DISP:WIND{window_number}:TRAC:NEXT?",
                    )
                )
            )
            await asyncio.to_thread(
                self._instrument.write,
                f"DISP:WIND{window_number}:TRAC{trace_number}:FEED 'ANTENNA_MEAS'",
            )

        await asyncio.to_thread(self._instrument.write, f"DISP:WIND{window_number}:ENAB ON")
        await asyncio.to_thread(
            self._instrument.write,
            f"DISP:WIND{window_number}:TRAC{trace_number}:STAT ON",
        )
        await asyncio.to_thread(
            self._instrument.write,
            f"DISP:WIND{window_number}:TRAC{trace_number}:TITL OFF",
        )
        await asyncio.to_thread(
            self._instrument.write,
            f"DISP:WIND{window_number}:TRAC{trace_number}:SEL",
        )
        await asyncio.to_thread(self._instrument.write, "CALC1:PAR:SEL 'ANTENNA_MEAS'")
        await asyncio.to_thread(self._instrument.query, "*OPC?")
        active_measurement = str(
            await asyncio.to_thread(self._instrument.query, "SYST:ACT:MEAS?")
        ).strip().strip("\"")
        if active_measurement != "ANTENNA_MEAS":
            raise RuntimeError(f"VNA 前面板活动测量 {active_measurement!r} != ANTENNA_MEAS")

        self._display_window_number = window_number
        self._display_trace_number = trace_number
        self._display_needs_autoscale = True
        return {
            "display_window": window_number,
            "display_trace": trace_number,
            "display_measurement": active_measurement,
        }

    async def _autoscale_after_sweep(self) -> None:
        if (self._display_needs_autoscale and self._display_window_number is not None
                and self._display_trace_number is not None):
            # Called after a completed internal sweep or validated RTC row, while
            # HOLD makes the visible trace stable. Autoscale once per configuration.
            await asyncio.to_thread(self._instrument.write,
                f"DISP:WIND{self._display_window_number}:TRAC{self._display_trace_number}:Y:AUTO")
            await asyncio.to_thread(self._instrument.query, "*OPC?")
            self._display_needs_autoscale = False

    async def connect(self) -> dict[str, Any]:
        self.snapshot.update(state=DeviceState.CONNECTING)
        try:
            import pyvisa

            # 使用 Windows 已安装的 VISA 库打开用户填写的真实资源。clear 仅清理
            # VISA 会话状态；随后用 *IDN? 和错误队列确认目标确实是可通信仪表。
            self._manager = await asyncio.to_thread(pyvisa.ResourceManager, self.backend) if self.backend else await asyncio.to_thread(pyvisa.ResourceManager)
            self._instrument = await asyncio.to_thread(self._manager.open_resource, self.resource)
            self._instrument.timeout = self.timeout_ms
            await asyncio.to_thread(self._instrument.clear)
            initial_errors = await asyncio.to_thread(self._drain_error_queue_blocking)
            identity = str(await asyncio.to_thread(self._instrument.query, "*IDN?")).strip()
            if not identity:
                raise RuntimeError("*IDN? 返回为空")
            identity_errors = await asyncio.to_thread(self._drain_error_queue_blocking)
            if self._fault_errors(identity_errors):
                raise RuntimeError(f"*IDN? 后错误队列非零：{identity_errors}")
        except Exception as exc:
            self.snapshot.update(state=DeviceState.FAULT, error=str(exc))
            raise ServiceError(
                "DEVICE_FAULT",
                "VNA VISA 身份查询失败",
                "vna_identity",
                self.resource,
                {"backend": self.backend, "error": str(exc)},
                next_action="请在 Keysight Connection Expert 中用同一 VISA 资源验证 *IDN?",
            ) from exc
        self.snapshot.identity = identity
        self.snapshot.update(
            state=DeviceState.READY,
            initial_error_queue=initial_errors,
            identity_error_queue=identity_errors,
            visa_resource=self.resource,
        )
        return self.snapshot.as_dict()

    async def disconnect(self) -> None:
        if self._instrument is not None:
            if self._buffer_memory_name is not None:
                await self.abort_buffered_acquisition()
            await asyncio.to_thread(self._instrument.close)
            self._instrument = None
        if self._manager is not None:
            await asyncio.to_thread(self._manager.close)
            self._manager = None
        self.snapshot.update(state=DeviceState.DISCONNECTED)

    async def configure(
        self,
        *,
        s_parameter: str,
        frequencies_hz: np.ndarray,
        if_bandwidth_hz: float,
        source_power_dbm: float,
        averaging_enabled: bool,
        averaging_count: int,
        trigger_mode: str = "INTERNAL_SINGLE",
    ) -> dict[str, Any]:
        if self._instrument is None:
            raise ServiceError("NOT_RUNNABLE", "VNA 未连接", "vna_configure", self.resource)
        if s_parameter not in {"S11", "S21", "S12", "S22"}:
            raise invalid("不支持的 S 参数", stage="vna_configure", target=s_parameter)
        points = int(frequencies_hz.size)
        trigger_map = {
            "INTERNAL_SINGLE": "IMM",
            "EXTERNAL_SWEEP": "EXT",
            "EXTERNAL_POINT": "EXT",
        }
        trigger_behavior = {
            "INTERNAL_SINGLE": "CHAN",
            "EXTERNAL_SWEEP": "CHAN",
            "EXTERNAL_POINT": "POIN",
        }
        if trigger_mode not in trigger_map:
            raise invalid("不支持的 VNA 触发方式", stage="vna_configure", target=trigger_mode)
        if averaging_enabled and trigger_mode == "EXTERNAL_SWEEP":
            raise invalid("外部整扫暂不支持平均，请使用内部单扫或 RTC 逐点平均", stage="vna_configure", target=trigger_mode)
        if averaging_enabled and not 2 <= averaging_count <= 65536:
            raise invalid("启用矢网内部平均时，平均次数必须在 2..65536", stage="vna_configure", target=str(averaging_count))
        if not averaging_enabled and averaging_count != 1:
            raise invalid("未启用矢网内部平均时，平均次数必须为 1", stage="vna_configure", target=str(averaging_count))
        source_port = int(s_parameter[2])
        averaging_mode = "POINT" if trigger_mode == "EXTERNAL_POINT" else "SWEEP"
        self._buffer_plan = None
        async with self._lock:
            try:
                # E5080B 的 HOLD/SINGLE 状态决定通道接收多少次触发。先进入 HOLD 并
                # 等待完成，避免在半写入配置上扫描；采集时再切到 SINGLE。平台只
                # 创建或修改自己的 ANTENNA_MEAS，不删除操作员的其他测量/窗口。
                await asyncio.to_thread(self._instrument.write, "SENS1:SWE:MODE HOLD")
                await asyncio.to_thread(self._instrument.query, "*OPC?")
                hold_errors = await asyncio.to_thread(self._drain_error_queue_blocking)
                if self._fault_errors(hold_errors):
                    raise RuntimeError(f"VNA 进入 HOLD 失败：{hold_errors}")

                catalog_before_text = str(
                    await asyncio.to_thread(self._instrument.query, "CALC1:PAR:CAT:EXT? DEF")
                ).strip()
                catalog_before = self._measurement_catalog(catalog_before_text)
                if "ANTENNA_MEAS" in catalog_before:
                    await asyncio.to_thread(self._instrument.write, "CALC1:PAR:SEL 'ANTENNA_MEAS'")
                    if catalog_before["ANTENNA_MEAS"] != s_parameter:
                        await asyncio.to_thread(self._instrument.write, f"CALC1:PAR:MOD:EXT '{s_parameter}'")
                else:
                    await asyncio.to_thread(
                        self._instrument.write,
                        f"CALC1:PAR:DEF:EXT 'ANTENNA_MEAS','{s_parameter}'",
                    )
                    await asyncio.to_thread(self._instrument.write, "CALC1:PAR:SEL 'ANTENNA_MEAS'")

                # Changing an S parameter can asynchronously rebuild the E5080B channel.
                # Wait for that rebuild before writing the stimulus; otherwise the channel
                # can restore its previous single-point frequency after our later writes.
                await asyncio.to_thread(self._instrument.query, "*OPC?")
                measurement_errors = await asyncio.to_thread(self._drain_error_queue_blocking)
                if self._fault_errors(measurement_errors):
                    raise RuntimeError(f"VNA 创建或修改测量失败：{measurement_errors}")

                # CALC:PAR:DEF only creates a background measurement. Explicitly feed
                # the platform-owned measurement to an unused visible trace, select it
                # for front-panel operation, and retain every operator-created trace.
                display_readback = await self._ensure_measurement_visible()
                measurement_number = int(float(await asyncio.to_thread(
                    self._instrument.query, "CALC1:PAR:MNUM?"
                )))

                center_hz = float((frequencies_hz[0] + frequencies_hz[-1]) / 2)
                span_hz = float(frequencies_hz[-1] - frequencies_hz[0])
                commands = [
                    "SENS1:SWE:TYPE LIN",
                    f"SENS1:FREQ:CENT {center_hz:.12g}",
                    f"SENS1:FREQ:SPAN {span_hz:.12g}",
                    f"SENS1:SWE:POIN {points}",
                    f"SENS1:BWID {if_bandwidth_hz:.12g}",
                    # Sij uses port j as its stimulus source. Set and later read back
                    # that port's leveled RF power in dBm; source AUTO behavior remains
                    # owned by the analyzer and is not replaced with a persistent ON.
                    f"SOUR1:POW{source_port} {source_power_dbm:.12g}",
                    f"SENS1:AVER:MODE {averaging_mode}",
                    f"SENS1:AVER:COUN {averaging_count}",
                    f"SENS1:AVER {'ON' if averaging_enabled else 'OFF'}",
                    f"SENS1:SWE:TRIG:MODE {trigger_behavior[trigger_mode]}",
                    f"TRIG:SOUR {trigger_map[trigger_mode]}",
                    "FORM:DATA ASC,0",
                ]
                for command in commands:
                    await asyncio.to_thread(self._instrument.write, command)
                await asyncio.to_thread(self._instrument.query, "*OPC?")
                errors = await asyncio.to_thread(self._drain_error_queue_blocking)
                if self._fault_errors(errors):
                    raise RuntimeError(f"VNA 设置错误队列非零：{errors}")
                catalog_after_text = str(
                    await asyncio.to_thread(self._instrument.query, "CALC1:PAR:CAT:EXT? DEF")
                ).strip()
                catalog_after = self._measurement_catalog(catalog_after_text)
                readback = {
                    "start_hz": float(await asyncio.to_thread(self._instrument.query, "SENS1:FREQ:STAR?")),
                    "stop_hz": float(await asyncio.to_thread(self._instrument.query, "SENS1:FREQ:STOP?")),
                    "points": int(float(await asyncio.to_thread(self._instrument.query, "SENS1:SWE:POIN?"))),
                    "if_bandwidth_hz": float(await asyncio.to_thread(self._instrument.query, "SENS1:BWID?")),
                    "source_power_dbm": float(
                        await asyncio.to_thread(self._instrument.query, f"SOUR1:POW{source_port}?")
                    ),
                    "source_port": source_port,
                    "averaging_enabled": bool(
                        int(float(await asyncio.to_thread(self._instrument.query, "SENS1:AVER?")))
                    ),
                    "averaging_count": int(
                        float(await asyncio.to_thread(self._instrument.query, "SENS1:AVER:COUN?"))
                    ),
                    "averaging_mode": str(
                        await asyncio.to_thread(self._instrument.query, "SENS1:AVER:MODE?")
                    ).strip(),
                    "sweep_time_seconds": float(
                        await asyncio.to_thread(self._instrument.query, "SENS1:SWE:TIME?")
                    ),
                    "sweep_type": str(await asyncio.to_thread(self._instrument.query, "SENS1:SWE:TYPE?")).strip(),
                    "s_parameter": catalog_after.get("ANTENNA_MEAS"),
                    "measurement_number": measurement_number,
                    "trigger_source": str(await asyncio.to_thread(self._instrument.query, "TRIG:SOUR?")).strip(),
                    "trigger_behavior": str(
                        await asyncio.to_thread(self._instrument.query, "SENS1:SWE:TRIG:MODE?")
                    ).strip(),
                    "data_format": str(await asyncio.to_thread(self._instrument.query, "FORM:DATA?")).strip(),
                    "trigger_mode": trigger_mode,
                    "error_queue": errors,
                    **display_readback,
                }
                expected_source = trigger_map[trigger_mode]
                expected_behavior = trigger_behavior[trigger_mode]
                mismatches: list[str] = []
                if not math.isclose(readback["start_hz"], float(frequencies_hz[0]), rel_tol=1e-10, abs_tol=1.0):
                    mismatches.append(f"起始频率 {readback['start_hz']} != {frequencies_hz[0]}")
                if not math.isclose(readback["stop_hz"], float(frequencies_hz[-1]), rel_tol=1e-10, abs_tol=1.0):
                    mismatches.append(f"终止频率 {readback['stop_hz']} != {frequencies_hz[-1]}")
                if readback["points"] != points:
                    mismatches.append(f"点数 {readback['points']} != {points}")
                if not math.isclose(readback["if_bandwidth_hz"], if_bandwidth_hz, rel_tol=1e-9, abs_tol=1e-6):
                    mismatches.append(f"IFBW {readback['if_bandwidth_hz']} != {if_bandwidth_hz}")
                if not math.isclose(readback["source_power_dbm"], source_power_dbm, rel_tol=0, abs_tol=0.01):
                    mismatches.append(f"源功率 {readback['source_power_dbm']} != {source_power_dbm} dBm")
                if readback["averaging_enabled"] != averaging_enabled:
                    mismatches.append(f"平均状态 {readback['averaging_enabled']} != {averaging_enabled}")
                if readback["averaging_count"] != averaging_count:
                    mismatches.append(f"平均次数 {readback['averaging_count']} != {averaging_count}")
                if not readback["averaging_mode"].upper().startswith(averaging_mode[:3]):
                    mismatches.append(f"平均模式 {readback['averaging_mode']} != {averaging_mode}")
                if readback["sweep_type"] != "LIN":
                    mismatches.append(f"扫描类型 {readback['sweep_type']} != LIN")
                if readback["s_parameter"] != s_parameter:
                    mismatches.append(f"S参数 {readback['s_parameter']} != {s_parameter}")
                if readback["trigger_source"] != expected_source:
                    mismatches.append(f"触发源 {readback['trigger_source']} != {expected_source}")
                if readback["trigger_behavior"] != expected_behavior:
                    mismatches.append(f"触发方式 {readback['trigger_behavior']} != {expected_behavior}")
                if not readback["data_format"].startswith("ASC"):
                    mismatches.append(f"数据格式 {readback['data_format']} 不是 ASCII")
                if mismatches:
                    raise RuntimeError("；".join(mismatches))
            except Exception as exc:
                raise ServiceError("DEVICE_FAULT", "VNA 设置或读回验证失败", "vna_configure", self.resource, {"error": str(exc)}) from exc
        self._configured_frequencies = np.asarray(frequencies_hz, dtype=float).copy()
        self._trigger_mode = trigger_mode
        self._averaging_enabled = averaging_enabled
        self._averaging_count = averaging_count
        self._sweep_time_seconds = readback["sweep_time_seconds"]
        self._measurement_number = measurement_number
        self._configured_s_parameter = s_parameter
        self.snapshot.update(settings=readback)
        return readback

    async def _close_buffer_unlocked(self) -> None:
        # Release only this adapter's named allocation, never MEM:RESET (all users).
        if self._buffer_memory_name is not None:
            await asyncio.to_thread(self._instrument.write, f'SYST:DATA:MEM:CLOS "{self._buffer_memory_name}"')
            self._buffer_memory_name = None

    def _validate_buffer_request(self, frequencies_hz: np.ndarray, sweep_count: int) -> np.ndarray:
        frequencies = np.asarray(frequencies_hz, dtype=float)
        if self._instrument is None or self._measurement_number is None or self._trigger_mode != "EXTERNAL_POINT":
            raise ServiceError("NOT_RUNNABLE", "请先配置 RTC 外部逐点测量", "vna_buffer", self.resource)
        if isinstance(sweep_count, bool) or not isinstance(sweep_count, int) or sweep_count < 1:
            raise invalid("缓冲扫频次数必须为正整数", stage="vna_buffer")
        if (self._configured_frequencies is None or frequencies.shape != self._configured_frequencies.shape
                or not np.allclose(frequencies, self._configured_frequencies, rtol=1e-10, atol=1.0)):
            raise ServiceError("NOT_RUNNABLE", "缓冲频率与最近配置不一致", "vna_buffer", self.resource)
        return frequencies

    async def _buffer_identity_and_sources(self) -> dict[str, Any]:
        """Read actual measurement/calibration configuration; never disable calibration.

        Point triggers visit source directions in sequence. A full multiport correction
        requires extra directions and cannot be treated as F triggers per spectrum.
        This first implementation accepts only proven single-source S-parameter paths.
        """
        query = self._instrument.query
        name = str(await asyncio.to_thread(query, f"SYST:MEAS{self._measurement_number}:NAME?")).strip().strip('"')
        catalog = self._measurement_catalog(str(await asyncio.to_thread(query, "CALC1:PAR:CAT:EXT? DEF")))
        if name != "ANTENNA_MEAS" or catalog.get(name) != self._configured_s_parameter:
            raise RuntimeError("VNA 平台测量身份或 S 参数已改变，请重新准备")
        source = self._configured_s_parameter[2]
        if any(not re.fullmatch(r"S[12][12]", parameter) or parameter[2] != source for parameter in catalog.values()):
            raise RuntimeError("RTC 缓冲要求通道 1 的测量使用同一源端口；当前存在其他源方向或非标准 S 参数")
        # CURR distributes triggers between triggerable channels. Do not silently
        # alter the operator's other channels; require them already held.
        channels = self._number_catalog(str(await asyncio.to_thread(query, "SYST:CHAN:CAT?")))
        for channel in channels:
            if channel != 1:
                mode = str(await asyncio.to_thread(query, f"SENS{channel}:SWE:MODE?")).strip().upper()
                if mode != "HOLD":
                    raise RuntimeError(f"通道 {channel} 尚未 HOLD，会分走 RTC 外部触发")
        correction: dict[str, str] = {}
        numbers = self._number_catalog(str(await asyncio.to_thread(query, "SYST:MEAS:CAT? 1")))
        if len(numbers) != len(catalog) or self._measurement_number not in numbers:
            raise RuntimeError("VNA 通道测量目录不一致")
        for number in numbers:
            measurement_name = str(await asyncio.to_thread(query, f"SYST:MEAS{number}:NAME?")).strip().strip('"')
            if measurement_name not in catalog:
                raise RuntimeError("VNA 测量编号目录与名称目录不一致")
            enabled = bool(int(float(await asyncio.to_thread(query, f"CALC1:MEAS{number}:CORR:STAT?"))))
            if not enabled:
                correction[measurement_name] = "OFF"
                continue
            kind = str(await asyncio.to_thread(query, f"CALC1:MEAS{number}:CORR:TYPE?")).strip().strip('"')
            parameter = catalog[measurement_name]
            safe_kinds = {f"Response({parameter})", f"ResponseAndIsolation({parameter})"}
            if parameter[1] == source:
                safe_kinds.update({f"Full 1 Port({source})", f"Full 1 Port with power({source})"})
            if kind not in safe_kinds:
                raise RuntimeError(f"测量 {measurement_name} 的校准 {kind!r} 未确认单源触发映射；保留校准并停止准备")
            indicator = str(await asyncio.to_thread(query, f"CALC1:MEAS{number}:CORR:IND?")).strip().upper()
            if indicator not in {"MAST", "INT", "DELT"}:
                raise RuntimeError(f"测量 {measurement_name} 的校准无效：{indicator}")
            correction[measurement_name] = kind
        return {"measurement_catalog": catalog, "correction_types": correction, "source_port": int(source)}

    async def prepare_buffered_acquisition(self, frequencies_hz: np.ndarray, sweep_count: int) -> dict[str, Any]:
        """Prepare one RTC group/row, then continuously accept one trigger per frequency.

        Buffer depth is computed by the workflow, with no arbitrary software capacity
        ceiling. MEM:SIZE is allocated bytes, never proof of completed acquisition.
        """
        frequencies = self._validate_buffer_request(frequencies_hz, sweep_count)
        self._buffer_plan = None
        async with self._lock:
            try:
                # HOLD before touching buffer ownership or settings. Commands alter
                # the real VNA once only; a timeout is reported, never retransmitted.
                await asyncio.to_thread(self._instrument.write, "SENS1:SWE:MODE HOLD")
                await asyncio.to_thread(self._instrument.query, "*OPC?")
                identity = await self._buffer_identity_and_sources()
                await self._close_buffer_unlocked()
                memory_name = "ANTENNA_" + uuid.uuid4().hex
                self._buffer_memory_name = memory_name
                mnum = self._measurement_number
                commands = [
                    "TRIG:SOUR EXT", "TRIG:SCOP CURR", "SENS1:SWE:TRIG:MODE POIN",
                    "TRIG:ROUT:INP MAIN", "TRIG:TYPE EDGE", "TRIG:SLOP POS", "TRIG:READ:POL HIGH",
                    # Ignore trigger edges before Ready; do not queue an old edge
                    # and consume it as the first RTC point after arming.
                    "CONT:SIGN:TRIG:ATBA 0",
                    "SENS1:AVER:MODE POIN", "SYST:DATA:MEM:INIT",
                    f"SYST:DATA:MEM:MEAS{mnum}:FORM SDATA",
                    f"SYST:DATA:MEM:MEAS{mnum}:REP {sweep_count}",
                    f"SYST:DATA:MEM:MEAS{mnum}:ADD", f'SYST:DATA:MEM:COMM "{memory_name}"',
                ]
                for command in commands:
                    await asyncio.to_thread(self._instrument.write, command)
                await asyncio.to_thread(self._instrument.query, "*OPC?")
                allocated_bytes = int(float(await asyncio.to_thread(self._instrument.query, "SYST:DATA:MEM:SIZE?")))
                depth = int(float(await asyncio.to_thread(self._instrument.query, f"SYST:DATA:MEM:MEAS{mnum}:REP?")))
                if depth != sweep_count or allocated_bytes < sweep_count * frequencies.size * 8:
                    raise RuntimeError(f"VNA 缓冲分配不完整：重复数 {depth}/{sweep_count}，分配 {allocated_bytes} 字节")
                expected = {
                    "TRIG:SOUR?": "EXT", "TRIG:SCOP?": "CURR", "SENS1:SWE:TRIG:MODE?": "POIN",
                    "TRIG:ROUT:INP?": "MAIN", "TRIG:TYPE?": "EDGE", "TRIG:SLOP?": "POS",
                    "TRIG:READ:POL?": "HIGH", "SENS1:AVER:MODE?": "POIN",
                }
                for command, value in expected.items():
                    actual = str(await asyncio.to_thread(self._instrument.query, command)).strip().upper()
                    if not actual.startswith(value):
                        raise RuntimeError(f"VNA 读回 {command}={actual}，期望 {value}")
                accept_before_armed = float(await asyncio.to_thread(self._instrument.query, "CONT:SIGN:TRIG:ATBA?"))
                if accept_before_armed != 0:
                    raise RuntimeError(f"VNA 尚未关闭预就绪触发记忆：ATBA={accept_before_armed}")
                average_enabled = bool(int(float(await asyncio.to_thread(self._instrument.query, "SENS1:AVER?"))))
                average_count = int(float(await asyncio.to_thread(self._instrument.query, "SENS1:AVER:COUN?")))
                if average_enabled != self._averaging_enabled or average_count != self._averaging_count:
                    raise RuntimeError("VNA 平均设置已改变，请重新准备")
                errors = await asyncio.to_thread(self._drain_error_queue_blocking)
                if self._fault_errors(errors):
                    raise RuntimeError(f"VNA 缓冲设置错误：{errors}")
                # CONT accepts external pulses indefinitely, it never emits triggers.
                # Avoid *OPC? while waiting for external points; use static ready query
                # before RTC ARM and require the configured CONT state to read back.
                await asyncio.to_thread(self._instrument.write, "SENS1:SWE:MODE CONT")
                mode = str(await asyncio.to_thread(self._instrument.query, "SENS1:SWE:MODE?")).strip().upper()
                if not mode.startswith("CONT"):
                    raise RuntimeError(f"VNA 未进入 Continuous：{mode}")
                deadline = time.monotonic() + self.timeout_ms / 1000
                while not bool(int(float(await asyncio.to_thread(self._instrument.query, "TRIG:STAT:READ? MEAS")))):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("VNA 尚未就绪接受首个外部触发")
                    await asyncio.sleep(0.02)
                errors = await asyncio.to_thread(self._drain_error_queue_blocking)
                if self._fault_errors(errors):
                    raise RuntimeError(f"VNA Continuous 启动错误：{errors}")
                plan = {
                    "sweep_count": sweep_count, "points_per_sweep": int(frequencies.size),
                    "triggers_per_sweep": int(frequencies.size), "memory_bytes": allocated_bytes,
                    "measurement_number": mnum, "buffer_kind": "REPEATED_SWEEP", "averaging_mode": "POINT",
                    "averaging_count": average_count, "trigger_mode": "EXTERNAL_POINT_CONTINUOUS",
                    "ready_polarity": "HIGH", "accept_trigger_before_armed": False, **identity,
                }
                self._buffer_plan = plan
                self.snapshot.update(buffer=plan)
                return dict(plan)
            except Exception as exc:
                raise ServiceError("DEVICE_FAULT", "VNA RTC 缓冲准备失败", "vna_buffer_prepare", self.resource,
                                   {"error": str(exc)}, side_effect_possible=True) from exc

    async def read_buffered_acquisition(
        self, frequencies_hz: np.ndarray, sweep_count: int, *, completed_trigger_count: int,
        sample_contexts: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """HOLD only after RTC drain/count confirmation, read oldest to newest.

        completed_trigger_count must come from RTC readback, never buffer depth.
        Each returned row is one complete complex spectrum. Real adapters ignore
        sample_contexts; those label physical samples in the workflow/simulator.
        """
        frequencies = self._validate_buffer_request(frequencies_hz, sweep_count)
        plan = self._buffer_plan
        if plan is None or plan["sweep_count"] != sweep_count:
            raise ServiceError("NOT_RUNNABLE", "VNA 缓冲未准备或行规模改变", "vna_buffer_read", self.resource)
        expected_triggers = sweep_count * plan["triggers_per_sweep"]
        if completed_trigger_count != expected_triggers:
            raise ServiceError("DATA_INTEGRITY", "RTC 完成点数与 VNA 缓冲计划不一致", "vna_buffer_read", self.resource,
                               {"expected": expected_triggers, "completed": completed_trigger_count})
        async with self._lock:
            try:
                await asyncio.to_thread(self._instrument.write, "SENS1:SWE:MODE HOLD")
                await asyncio.to_thread(self._instrument.query, "*OPC?")
                if str(await asyncio.to_thread(self._instrument.query, "SENS1:SWE:MODE?")).strip().upper() != "HOLD":
                    raise RuntimeError("VNA 未确认 HOLD，不能读取稳定行数据")
                identity = await self._buffer_identity_and_sources()
                if any(identity[key] != plan[key] for key in identity):
                    raise RuntimeError("VNA 测量或校准在采集期间发生变化")
                average_mode = str(await asyncio.to_thread(self._instrument.query, "SENS1:AVER:MODE?")).strip().upper()
                average_count = int(float(await asyncio.to_thread(self._instrument.query, "SENS1:AVER:COUN?")))
                average_enabled = bool(int(float(await asyncio.to_thread(self._instrument.query, "SENS1:AVER?"))))
                if (not average_mode.startswith("POIN") or average_count != plan["averaging_count"]
                        or average_enabled != self._averaging_enabled):
                    raise RuntimeError("VNA 平均设置在采集期间发生变化")
                actual = np.asarray(await asyncio.to_thread(self._instrument.query_ascii_values,
                    f"CALC1:MEAS{self._measurement_number}:X?", container=np.array), dtype=float)
                if (actual.shape != frequencies.shape or not np.all(np.isfinite(actual))
                        or not np.allclose(actual, frequencies, rtol=1e-10, atol=1.0)
                        or (actual.size > 1 and not np.all(np.diff(actual) > 0))):
                    raise RuntimeError("VNA 缓冲频率轴与冻结计划不一致")
                spectra = np.empty((sweep_count, frequencies.size), dtype=np.complex128)
                for index in range(sweep_count):
                    # REP1 is oldest, REP{depth} newest. RI supplies two numbers per
                    # frequency in chronological order; do not read the live SDATA.
                    values = np.asarray(await asyncio.to_thread(self._instrument.query_ascii_values,
                        f"SYST:DATA:MEM:READ:MEAS{self._measurement_number}:REP{index + 1}? RI", container=np.array), dtype=float)
                    if values.ndim != 1 or values.size != frequencies.size * 2 or not np.all(np.isfinite(values)):
                        raise RuntimeError(f"VNA 缓冲记录 {index + 1} 数量错误或含非有限值")
                    spectra[index] = values[0::2] + 1j * values[1::2]
                await self._autoscale_after_sweep()
                errors = await asyncio.to_thread(self._drain_error_queue_blocking)
                if self._fault_errors(errors):
                    raise RuntimeError(f"VNA 缓冲读取错误：{errors}")
                await self._close_buffer_unlocked()
                close_errors = await asyncio.to_thread(self._drain_error_queue_blocking)
                if self._fault_errors(close_errors):
                    raise RuntimeError(f"VNA 缓冲释放错误：{close_errors}")
                self.snapshot.update(last_buffer_sweeps=sweep_count, last_acquired_points=int(frequencies.size * sweep_count),
                                     last_error_queue=errors, last_frequency_axis_hz=actual.tolist())
                return spectra
            except Exception as exc:
                raise ServiceError("DATA_INTEGRITY", "VNA RTC 缓冲读取或校验失败", "vna_buffer_read", self.resource,
                                   {"error": str(exc)}, side_effect_possible=True) from exc
            finally:
                # A group/row can be read once only; retries must not relabel stale data.
                self._buffer_plan = None

    async def abort_buffered_acquisition(self) -> None:
        """Stop accepting triggers without saving partial data or restarting a sweep."""
        self._buffer_plan = None
        if self._instrument is None:
            return
        async with self._lock:
            try:
                await asyncio.to_thread(self._instrument.write, "SENS1:SWE:MODE HOLD")
                await asyncio.to_thread(self._instrument.query, "*OPC?")
                mode = str(await asyncio.to_thread(self._instrument.query, "SENS1:SWE:MODE?")).strip().upper()
                errors = await asyncio.to_thread(self._drain_error_queue_blocking)
                if mode != "HOLD" or self._fault_errors(errors):
                    raise RuntimeError(f"HOLD={mode}, errors={errors}")
                await self._close_buffer_unlocked()
                close_errors = await asyncio.to_thread(self._drain_error_queue_blocking)
                if self._fault_errors(close_errors):
                    raise RuntimeError(f"VNA 缓冲释放错误：{close_errors}")
            except Exception as exc:
                raise ServiceError("DEVICE_FAULT", "VNA RTC 停止未确认", "vna_buffer_abort", self.resource,
                                   {"error": str(exc)}, side_effect_possible=True) from exc

    async def acquire(self, frequencies_hz: np.ndarray, **_: Any) -> np.ndarray:
        if self._instrument is None:
            raise ServiceError("NOT_RUNNABLE", "VNA 未连接", "vna_acquire", self.resource)
        if self._trigger_mode != "INTERNAL_SINGLE":
            raise invalid("外部 RTC 测量必须使用缓冲采集接口", stage="vna_acquire")
        expected_frequencies = np.asarray(frequencies_hz, dtype=float)
        if self._configured_frequencies is None or expected_frequencies.shape != self._configured_frequencies.shape or not np.allclose(
            expected_frequencies,
            self._configured_frequencies,
            rtol=1e-10,
            atol=1.0,
        ):
            raise ServiceError("NOT_RUNNABLE", "VNA 采集频率与最近配置不一致", "vna_acquire", self.resource)
        async with self._lock:
            try:
                # E5080B 扫频平均在每个原子采集前清零，再用 GROUPS 完成规定
                # 次数；*OPC? 返回后才读取一条最终复数 SDATA。超时只报告，
                # 不盲目重发可能重复采集的命令。
                original_timeout = self._instrument.timeout
                # Address X/SDATA by the platform measurement's unique Tr#; selecting
                # another front-panel trace must never redirect our measurement data.
                name = str(await asyncio.to_thread(
                    self._instrument.query, f"SYST:MEAS{self._measurement_number}:NAME?"
                )).strip().strip('"')
                catalog = self._measurement_catalog(str(await asyncio.to_thread(
                    self._instrument.query, "CALC1:PAR:CAT:EXT? DEF"
                )))
                if name != "ANTENNA_MEAS" or catalog.get(name) != self._configured_s_parameter:
                    raise RuntimeError("VNA 平台测量身份或 S 参数已改变，请重新准备测试")
                estimated_ms = int(self._sweep_time_seconds * self._averaging_count * 1000 + 5000)
                self._instrument.timeout = max(self.timeout_ms, estimated_ms)
                if self._averaging_enabled:
                    await asyncio.to_thread(self._instrument.write, "SENS1:AVER:CLE")
                    await asyncio.to_thread(
                        self._instrument.write,
                        f"SENS1:SWE:GRO:COUN {self._averaging_count}",
                    )
                    await asyncio.to_thread(self._instrument.write, "SENS1:SWE:MODE GRO")
                if not self._averaging_enabled:
                    await asyncio.to_thread(self._instrument.write, "SENS1:SWE:MODE SING")
                await asyncio.to_thread(self._instrument.query, "*OPC?")
                actual_frequencies = await asyncio.to_thread(
                    self._instrument.query_ascii_values,
                    f"CALC1:MEAS{self._measurement_number}:X?",
                    container=np.array,
                )
                values = await asyncio.to_thread(
                    self._instrument.query_ascii_values,
                    f"CALC1:MEAS{self._measurement_number}:DATA:SDATA?",
                    container=np.array,
                )
                await self._autoscale_after_sweep()
                errors = await asyncio.to_thread(self._drain_error_queue_blocking)
                if self._fault_errors(errors):
                    raise RuntimeError(f"VNA 采集错误队列非零：{errors}")
            except Exception as exc:
                raise ServiceError("DEVICE_FAULT", "VNA SDATA 采集失败", "vna_acquire", self.resource, {"error": str(exc)}) from exc
            finally:
                if 'original_timeout' in locals():
                    self._instrument.timeout = original_timeout
        if actual_frequencies.size != expected_frequencies.size or not np.allclose(
            actual_frequencies,
            expected_frequencies,
            rtol=1e-10,
            atol=1.0,
        ):
            raise ServiceError(
                "DATA_INTEGRITY",
                "VNA 实际频率轴与测试计划不一致",
                "vna_acquire",
                self.resource,
                {
                    "expected_hz": expected_frequencies.tolist(),
                    "actual_hz": np.asarray(actual_frequencies).tolist(),
                },
            )
        if actual_frequencies.size > 1 and not np.all(np.diff(actual_frequencies) > 0):
            raise ServiceError("DATA_INTEGRITY", "VNA 实际频率轴不是严格递增", "vna_acquire", self.resource)
        if values.size != frequencies_hz.size * 2:
            raise ServiceError(
                "DATA_INTEGRITY",
                "VNA SDATA 数量与频率点数不一致",
                "vna_acquire",
                self.resource,
                {"expected_pairs": int(frequencies_hz.size), "values": int(values.size)},
            )
        complex_values = values[0::2] + 1j * values[1::2]
        if not np.all(np.isfinite(complex_values.real)) or not np.all(np.isfinite(complex_values.imag)):
            raise ServiceError("DATA_INTEGRITY", "VNA 返回非有限复数", "vna_acquire", self.resource)
        self.snapshot.update(
            last_frequency_axis_hz=np.asarray(actual_frequencies).tolist(),
            last_acquired_points=int(actual_frequencies.size),
            last_error_queue=errors,
        )
        return complex_values


class RealTurntable(DeviceAdapter):
    """Windows PMAC/PComm bridge using the supplied ImacFxDll.

    All engineering-unit values cross the DLL boundary with a scale of 10000, per
    the user's confirmed controller configuration. Completion is based on fresh
    position *and* velocity readback; a successful method return is not enough.
    """

    AXES = {1: "azimuth", 2: "elevation", 3: "polarization", 4: "feed", 7: "translation"}
    # ImacFxDll remaps logical axis 7 to PMAC motor 5 before issuing #5J=...
    # or reading #5P/M574.  Keep the mapping explicit for safety diagnostics;
    # never infer that PMAC motor 7 is the translation stage.
    PHYSICAL_MOTORS = {1: 1, 2: 2, 3: 3, 4: 4, 7: 5}
    SPEED_LIMITS = {1: 20.0, 2: 3.0, 3: 20.0, 4: 20.0, 7: 20.0}
    SCALE = 10000.0
    STATIONARY_VELOCITY_TOLERANCE = 0.000001
    STABLE_READBACK_SAMPLES = 3

    def __init__(
        self,
        dll_path: str,
        *,
        device_number: int = 0,
        controller_ip: str = "192.168.1.101",
        position_limits: dict[int, tuple[float, float]] | None = None,
    ) -> None:
        super().__init__("turntable", DeviceSource.REAL)
        self.dll_path = str(Path(dll_path).resolve())
        self.device_number = int(device_number)
        self.controller_ip = controller_ip
        self.position_limits = position_limits or {
            1: (-360.0, 360.0),
            2: (-90.0, 90.0),
            3: (-360.0, 360.0),
            4: (-360.0, 360.0),
            7: (-1000.0, 1000.0),
        }
        self._imac: Any = None
        # A motion owns the motion lock until it reaches a terminal state, preventing
        # a second move/home from being queued as an accidental duplicate. Individual
        # vendor-DLL calls only hold the I/O lock briefly so readback and Stop can run
        # while an axis is moving.
        self._motion_lock = asyncio.Lock()
        self._io_lock = asyncio.Lock()
        self._motion_cancelled = asyncio.Event()
        self._active_motion_axis: int | None = None

    @asynccontextmanager
    async def _motion(self, axis: int):
        async with self._motion_lock:
            self._active_motion_axis = axis
            self._motion_cancelled.clear()
            try:
                yield
            finally:
                self._active_motion_axis = None

    async def connect(self) -> dict[str, Any]:
        self.snapshot.update(state=DeviceState.CONNECTING)
        try:
            # DLL Connect 成功后立即读取全部轴。只有能获得有限的位置和速度值，
            # 才把真实转台标记为 READY，避免仅凭厂商连接返回值判断设备可用。
            await asyncio.to_thread(self._load_and_connect)
            telemetry = await self.read_all_axes()
        except Exception as exc:
            self.snapshot.update(state=DeviceState.FAULT, error=str(exc))
            if isinstance(exc, ServiceError):
                raise
            raise ServiceError(
                "DEVICE_FAULT",
                "转台 PCommServer/DLL 连接失败",
                "turntable_connect",
                self.controller_ip,
                {"dll": self.dll_path, "device_number": self.device_number, "error": str(exc)},
                next_action="请确认 PeWin32Pro2 已配置 PMAC 00、IP 192.168.1.101，且 PCommServer 已安装",
            ) from exc
        self.snapshot.identity = f"PMAC:{self.device_number}@{self.controller_ip}"
        self.snapshot.update(state=DeviceState.READY, scale=self.SCALE, **telemetry)
        return self.snapshot.as_dict()

    def _load_and_connect(self) -> None:
        if not Path(self.dll_path).is_file():
            raise FileNotFoundError(self.dll_path)
        # pythonnet loads the managed wrapper; the wrapper itself locates the native
        # PCommServer installation registered by PeWin32Pro2. The controller IP is
        # intentionally not sent through this DLL because its public API accepts only
        # a PMAC device number; 192.168.1.101 must be bound to device 0 in PeWin32Pro2.
        import clr

        interop = Path(self.dll_path).with_name("Interop.PCOMMSERVERLib.dll")
        if interop.is_file():
            clr.AddReference(str(interop))
        clr.AddReference(self.dll_path)
        from ImacFxDll import ImacFxDllT

        self._imac = ImacFxDllT()
        if not bool(self._imac.Connect(self.device_number)):
            self._imac = None
            raise RuntimeError("ImacFxDll.Connect 返回 false")

    async def disconnect(self) -> None:
        async with self._motion_lock:
            async with self._io_lock:
                if self._imac is not None:
                    try:
                        await asyncio.to_thread(self._imac.Close, self.device_number)
                    finally:
                        self._imac = None
        self.snapshot.update(state=DeviceState.DISCONNECTED)

    def _require(self) -> Any:
        if self._imac is None:
            raise ServiceError("NOT_RUNNABLE", "转台尚未连接", "turntable", self.controller_ip)
        return self._imac

    @property
    def has_connection(self) -> bool:
        # FAULT may mean either no connection was established or a connected motor
        # failed. Only the former can skip hardware Stop during application shutdown.
        return self._imac is not None

    def _read_axis_blocking(self, axis: int) -> tuple[float, float]:
        imac = self._require()
        # 厂商 GetPosStr/GetVelStr 返回控制器原始量；现场实测比例系数为 10000，
        # 因此读回统一除以 10000 后才作为角度（轴 1/2/3/4）或毫米（轴 7）。
        position = float(str(imac.GetPosStr(self.device_number, axis))) / self.SCALE
        velocity = float(str(imac.GetVelStr(self.device_number, axis))) / self.SCALE
        if not math.isfinite(position) or not math.isfinite(velocity):
            raise ValueError(f"轴 {axis} 回读不是有限数")
        return position, velocity

    def _read_motion_diagnostics_blocking(self, axis: int) -> dict[str, Any]:
        """Read PMAC interlocks without changing or acknowledging controller state."""
        imac = self._require()
        motor = self.PHYSICAL_MOTORS[axis]
        base = motor * 100
        return {
            "logical_axis": axis,
            "physical_motor": motor,
            # Ixx00 is the PMAC motor activation control.  Mx31/Mx32 and
            # Mx40..Mx43 are the installed suggested motor-status definitions.
            "activation_i": int(imac.GetI(self.device_number, base)),
            "positive_limit": bool(imac.GetM(self.device_number, base + 31)),
            "negative_limit": bool(imac.GetM(self.device_number, base + 32)),
            "amplifier_enabled": bool(imac.GetM(self.device_number, base + 39)),
            "background_in_position": bool(imac.GetM(self.device_number, base + 40)),
            "warning_following_error": bool(imac.GetM(self.device_number, base + 41)),
            "fatal_following_error": bool(imac.GetM(self.device_number, base + 42)),
            "amplifier_fault": bool(imac.GetM(self.device_number, base + 43)),
        }

    @staticmethod
    def _motion_diagnostics_ready(diagnostics: dict[str, Any]) -> bool:
        return bool(
            diagnostics["activation_i"] == 1
            and not diagnostics["positive_limit"]
            and not diagnostics["negative_limit"]
            and not diagnostics["fatal_following_error"]
            and not diagnostics["amplifier_fault"]
        )

    async def _preflight_motion_unlocked(self, axis: int) -> None:
        if axis != 7:
            return
        # Both absolute motion and homing act on physical PMAC motor 5. Read the
        # interlocks before either side effect; never activate/clear/save implicitly.
        diagnostics = await asyncio.to_thread(self._read_motion_diagnostics_blocking, axis)
        self.snapshot.update(axis7_motion_diagnostics=diagnostics)
        if not self._motion_diagnostics_ready(diagnostics):
            raise ServiceError(
                "NOT_RUNNABLE",
                "轴 7 对应的 PMAC 电机 5 未激活或存在驱动故障，已拒绝发送运动命令",
                "turntable_move_preflight", str(axis), diagnostics,
                next_action=(
                    "请现场检查平移轴伺服电源、驱动器告警、急停和限位；在 PeWin32Pro2 中确认 "
                    "I500=1、M542=0、M543=0 后重新连接。软件不会自动修改或保存 PMAC 参数"
                ),
            )

    async def read_all_axes(self) -> dict[str, Any]:
        # Only serialize the five short DLL reads. The long-running move wait no longer
        # owns this lock, so the operator can obtain fresh position/speed while moving.
        async with self._io_lock:
            telemetry = await self._read_all_axes_unlocked()
        self.snapshot.update(**telemetry)
        return telemetry

    def _validate_motion(self, axis: int, target: float, speed: float) -> None:
        if axis not in self.AXES:
            raise invalid("轴号必须为 1/2/3/4/7", stage="turntable_move", target=str(axis))
        low, high = self.position_limits[axis]
        if not low <= target <= high:
            raise invalid("目标位置超出部署范围", stage="turntable_move", target=str(axis), target_value=target, limits=[low, high])
        if not math.isfinite(speed) or speed <= 0:
            raise invalid("转台移动速度必须大于 0，不能为负数", stage="turntable_move", target=str(axis), speed=speed)
        if speed > self.SPEED_LIMITS[axis]:
            raise invalid("速度超出轴上限", stage="turntable_move", target=str(axis), speed=speed, limit=self.SPEED_LIMITS[axis])

    async def move_to(
        self,
        axis: int,
        target: float,
        speed: float,
        *,
        timeout_seconds: float = 120,
        position_tolerance: float = 0.01,
        velocity_tolerance: float = STATIONARY_VELOCITY_TOLERANCE,
    ) -> dict[str, Any]:
        # The operator-facing precision is four decimal places. Normalize before applying
        # the fixed 10000 engineering-unit scale used by the vendor DLL.
        speed = round(float(speed), 4)
        self._validate_motion(axis, target, speed)
        async with self._motion(axis):
            async with self._io_lock:
                imac = self._require()
                await self._preflight_motion_unlocked(axis)
                # The five-argument overload names its final parameter bMoveAbsType.
                # Vendor-DLL reflection and real-device readback both confirm True means
                # absolute positioning; False performs a relative displacement.
                # MoveDeviceToPos 是真实机械运动的副作用边界：目标和速度都乘以 10000，
                # bMoveAbsType=True 表示绝对位置。发送完成后不长期占用 DLL I/O 锁；
                # 等待期间每次只短暂读取，允许界面并发读取五轴位置/速度或执行 Stop。
                vendor_response = await asyncio.to_thread(
                    imac.MoveDeviceToPos,
                    self.device_number,
                    int(axis),
                    float(self.SCALE * target),
                    float(self.SCALE * speed),
                    True,
                )
                vendor_response = str(vendor_response or "").strip()
                if vendor_response:
                    raise ServiceError(
                        "DEVICE_FAULT",
                        "厂商 DLL 拒绝转台运动命令",
                        "turntable_move_send",
                        str(axis),
                        {"vendor_response": vendor_response, "target": target, "speed": speed},
                        side_effect_possible=True,
                    )
            deadline = time.monotonic() + timeout_seconds
            stable_samples = 0
            # The minimum legal motion speed is 0.0001 engineering units/s. A
            # stationary threshold must be smaller; 0.01 formerly accepted motion
            # at full commanded low speed. Require three fresh stationary samples.
            stationary_limit = min(velocity_tolerance, self.STATIONARY_VELOCITY_TOLERANCE)
            while time.monotonic() < deadline:
                async with self._io_lock:
                    position, velocity = await asyncio.to_thread(self._read_axis_blocking, axis)
                if self._motion_cancelled.is_set():
                    raise ServiceError(
                        "NOT_RUNNABLE",
                        "转台运动已被软件停止",
                        "turntable_move_wait",
                        str(axis),
                        {"target": target, "position": position, "velocity": velocity},
                        side_effect_possible=True,
                    )
                stable_samples = stable_samples + 1 if (
                    abs(position - target) <= position_tolerance and abs(velocity) <= stationary_limit
                ) else 0
                if stable_samples >= self.STABLE_READBACK_SAMPLES:
                    telemetry = await self.read_all_axes()
                    return {"axis": axis, "position": position, "velocity": velocity, **telemetry}
                await asyncio.sleep(0.1)
            raise ServiceError(
                "TIMEOUT",
                "转台未在时限内到位并静止",
                "turntable_move_wait",
                str(axis),
                {"target": target, "speed": speed},
                side_effect_possible=True,
                next_action="请使用软件停止并现场核对；软件停止不等同硬件急停",
            )

    async def _disable_azimuth_pulses(self) -> dict[str, Any]:
        """Disable comparison output once and confirm both vendor enable flags.

        SetTableEquEnable(device, axis=1, false) sets P5001=0, P5002=2001,
        M7104=0 and M7105=0 in the supplied DLL. These are temporary output
        settings, not a motion stop or SAVE. Unknown outcomes are never retried.
        """
        try:
            async with self._io_lock:
                imac = self._require()
                await asyncio.to_thread(imac.SetTableEquEnable, self.device_number, 1, False)
                flags = {str(register): int(await asyncio.to_thread(imac.GetM, self.device_number, register))
                         for register in (7104, 7105)}
            if any(flags.values()):
                raise RuntimeError(f"方位脉冲使能位未清零：{flags}")
            result = {"pulse_output_disabled": True, "pulse_enable_readback": flags}
            self.snapshot.update(**result)
            return result
        except Exception as exc:
            self.snapshot.update(state=DeviceState.UNKNOWN, pulse_output_disabled=False, error=str(exc))
            raise ServiceError("UNKNOWN", "方位脉冲输出关闭未确认", "turntable_pulse_disable", "1",
                               {"error": str(exc)}, side_effect_possible=True,
                               next_action="保持 RTC 禁止接受新组并现场核对脉冲输出；不要盲目重发扫描或关闭命令") from exc

    async def scan_azimuth(self, start_deg: float, end_deg: float, step_deg: float, speed: float, *,
                           timeout_seconds: float = 120, position_tolerance: float = 0.01) -> dict[str, Any]:
        """One vendor continuous scan, followed by verified pulse-output disable.

        Caller positions at start before arming RTC. PMAC program 11's endpoint
        pulse inclusion is undocumented; the workflow must check RTC counts and
        never derive per-point physical readbacks from this row-end evidence.
        """
        intervals, signed_step = azimuth_scan_geometry(start_deg, end_deg, step_deg, speed)
        self._validate_motion(1, start_deg, speed)
        self._validate_motion(1, end_deg, speed)
        # DLL sets P105=start-2 and P106=end+2 in raw controller units. Permit only
        # rows whose known 0.0002 degree pre-roll/overrun stays inside deployment limits.
        self._validate_motion(1, start_deg - 2 / self.SCALE, speed)
        self._validate_motion(1, end_deg + 2 / self.SCALE, speed)
        async with self._motion(1):
            attempted = False
            result: dict[str, Any] = {}
            try:
                async with self._io_lock:
                    imac = self._require()
                    position, velocity = await asyncio.to_thread(self._read_axis_blocking, 1)
                    if abs(position - start_deg) > position_tolerance or abs(velocity) > self.STATIONARY_VELOCITY_TOLERANCE:
                        raise ServiceError("NOT_RUNNABLE", "连续扫描前方位轴须已在起点并静止", "turntable_scan", "1",
                                           {"start_deg": start_deg, "position": position, "velocity": velocity})
                    if int(await asyncio.to_thread(imac.GetM, self.device_number, 7107)) == 1:
                        raise ServiceError("NOT_RUNNABLE", "转台扫描程序正在运行", "turntable_scan", "1")
                    # Real motion side effect: angles (deg) and speeds (deg/s) x10000;
                    # returnSpeed=scanSpeed; fDelta=0; iTime=0 ms selects continuous.
                    # This enables axis 1, M7104/7105 and temporary P variables,
                    # executes program 11, and can make the micro pre-roll/overrun.
                    # No SAVE, homing, or software retry is issued here.
                    attempted = True
                    self.snapshot.update(pulse_output_disabled=False)
                    sending = asyncio.create_task(asyncio.to_thread(
                        imac.MoveToPosByType, self.device_number, 1,
                        start_deg * self.SCALE, end_deg * self.SCALE, signed_step * self.SCALE,
                        speed * self.SCALE, speed * self.SCALE, 0.0, 0,
                    ))
                    try:
                        response = await asyncio.shield(sending)
                    except asyncio.CancelledError:
                        # Python cancellation does not stop the DLL thread. Wait for
                        # this one send to return before disabling its output, so it
                        # cannot subsequently turn pulses back on behind cleanup.
                        await sending
                        raise
                    if str(response or "").strip():
                        raise ServiceError("DEVICE_FAULT", "厂商 DLL 拒绝连续扫描", "turntable_scan_send", "1",
                                           {"vendor_response": str(response)}, side_effect_possible=True)
                deadline = time.monotonic() + timeout_seconds
                stable = 0
                motion_observed = False
                departure_threshold = min(position_tolerance, abs(end_deg - start_deg) / 2)
                while time.monotonic() < deadline:
                    async with self._io_lock:
                        position, velocity = await asyncio.to_thread(self._read_axis_blocking, 1)
                    if self._motion_cancelled.is_set():
                        raise ServiceError("NOT_RUNNABLE", "连续扫描已被软件停止", "turntable_scan_wait", "1", side_effect_possible=True)
                    # A short row may fit inside tolerance; require departure so
                    # stale stationary start samples cannot complete the scan.
                    motion_observed = motion_observed or abs(velocity) > self.STATIONARY_VELOCITY_TOLERANCE or abs(position - start_deg) >= departure_threshold
                    stable = stable + 1 if motion_observed and abs(position - end_deg) <= position_tolerance and abs(velocity) <= self.STATIONARY_VELOCITY_TOLERANCE else 0
                    if stable >= self.STABLE_READBACK_SAMPLES:
                        telemetry = await self.read_all_axes()
                        result.update(axis=1, position=position, velocity=velocity, interval_count=intervals,
                                      planned_point_count=intervals + 1, endpoint_pulse_count_verified=False,
                                      position_source="ROW_END_READBACK", **telemetry)
                        return result
                    await asyncio.sleep(0.1)
                raise ServiceError("TIMEOUT", "连续扫描未确认到达行末并静止", "turntable_scan_wait", "1",
                                   {"end_deg": end_deg, "speed_deg_s": speed}, side_effect_possible=True,
                                   next_action="执行软件停止并现场确认；不要重发扫描命令")
            finally:
                # Stationary does not prove comparison pulses are disabled. Normal,
                # timeout, software-stop and task-cancellation exits share this one
                # disable/readback. Failed preflight never touches existing outputs.
                if attempted:
                    result.update(await self._disable_azimuth_pulses())

    async def _read_all_axes_unlocked(self) -> dict[str, Any]:
        readings = {axis: await asyncio.to_thread(self._read_axis_blocking, axis) for axis in self.AXES}
        return {
            "positions": {str(axis): values[0] for axis, values in readings.items()},
            "velocities": {str(axis): values[1] for axis, values in readings.items()},
        }

    async def home(self, axis: int, *, timeout_seconds: float = 180) -> dict[str, Any]:
        if axis not in self.AXES:
            raise invalid("轴号必须为 1/2/3/4/7", stage="turntable_home", target=str(axis))
        async with self._motion(axis):
            async with self._io_lock:
                imac = self._require()
                await self._preflight_motion_unlocked(axis)
                # SHome 会触发真实轴寻零。到位判定以连续的位置零位误差和静止
                # 速度回读为准；部分控制器/轴（现场馈源轴）到位后仍不可靠地置位
                # GetHomeComplete，因此该厂商位只作为证据记录而不再阻塞完成。
                await asyncio.to_thread(imac.SHome, self.device_number, int(axis))
            deadline = time.monotonic() + timeout_seconds
            earliest_stationary_confirmation = time.monotonic() + 1.0
            stable_samples = 0
            motion_observed = False
            vendor_home_complete: bool | None = None
            while time.monotonic() < deadline:
                async with self._io_lock:
                    position, velocity = await asyncio.to_thread(self._read_axis_blocking, axis)
                    try:
                        vendor_home_complete = bool(
                            await asyncio.to_thread(imac.GetHomeComplete, self.device_number, int(axis))
                        )
                    except Exception:
                        vendor_home_complete = None
                if self._motion_cancelled.is_set():
                    raise ServiceError("NOT_RUNNABLE", "转台寻零已被软件停止", "turntable_home_wait", str(axis), side_effect_possible=True)
                motion_observed = motion_observed or abs(position) > 0.01 or abs(velocity) > self.STATIONARY_VELOCITY_TOLERANCE
                stable_samples = stable_samples + 1 if abs(position) <= 0.01 and abs(velocity) <= self.STATIONARY_VELOCITY_TOLERANCE else 0
                if stable_samples >= self.STABLE_READBACK_SAMPLES and (motion_observed or time.monotonic() >= earliest_stationary_confirmation):
                    telemetry = await self.read_all_axes()
                    return {
                        "axis": axis,
                        "position": position,
                        "velocity": velocity,
                        "home_complete": vendor_home_complete,
                        "completion_basis": "POSITION_AND_VELOCITY_STABLE",
                        **telemetry,
                    }
                await asyncio.sleep(0.1)
            raise ServiceError("TIMEOUT", "转台寻零超时", "turntable_home_wait", str(axis), side_effect_possible=True)

    async def stop(self, axis: int | str = "all") -> dict[str, Any]:
        axes = list(self.AXES) if axis == "all" else [int(axis)]
        for item in axes:
            if item not in self.AXES:
                raise invalid("轴号必须为 1/2/3/4/7", stage="turntable_stop", target=str(item))
        async with self._io_lock:
            imac = self._require()
            for item in axes:
                # Stop is a vendor software stop. It is intentionally not described as E-stop.
                # 此处调用厂商软件 Stop，不等同现场硬件急停；发送后还要读回速度，
                # 只有目标轴均静止才返回成功，无法确认时明确要求人工使用硬件急停。
                await asyncio.to_thread(imac.Stop, self.device_number, item)
            if self._active_motion_axis in axes:
                self._motion_cancelled.set()
        deadline = time.monotonic() + 10
        stable_samples = 0
        while time.monotonic() < deadline:
            telemetry = await self.read_all_axes()
            stable_samples = stable_samples + 1 if all(
                abs(float(telemetry["velocities"][str(item)])) <= self.STATIONARY_VELOCITY_TOLERANCE
                for item in axes
            ) else 0
            if stable_samples >= self.STABLE_READBACK_SAMPLES:
                return {"stopped": axes, **telemetry}
            await asyncio.sleep(0.1)
        raise ServiceError(
            "UNKNOWN",
            "已发送厂商 Stop，但未确认所有目标轴静止",
            "turntable_stop_verify",
            str(axis),
            side_effect_possible=True,
            next_action="请使用现场硬件急停或按现场规程确认",
        )


class RealRtc(RtcDeviceAdapter):
    def __init__(self, port: str, baud_rate: int = 115200, *, events: EventBus | None = None,
                 frame_decoder: Callable[[bytes], dict[str, Any] | None] | None = None) -> None:
        super().__init__(DeviceSource.REAL, events=events, frame_decoder=frame_decoder)
        self.client = RtcClient(RtcEndpoint(port=port, baudrate=baud_rate), on_event=self._on_client_event)
