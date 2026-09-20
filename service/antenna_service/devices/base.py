from __future__ import annotations

import asyncio
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from antenna_service.models import DeviceSource, DeviceState
from antenna_service.events import EventBus
from antenna_service.errors import invalid, ServiceError


@dataclass(slots=True)
class DeviceSnapshot:
    device_id: str
    source: DeviceSource
    state: DeviceState = DeviceState.DISCONNECTED
    identity: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def update(self, *, state: DeviceState | None = None, **details: Any) -> None:
        if state is not None:
            self.state = state
        self.details.update(details)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def as_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "source": self.source.value,
            "state": self.state.value,
            "identity": self.identity,
            "details": self.details,
            "updated_at": self.updated_at,
        }


class DeviceAdapter(ABC):
    def __init__(self, device_id: str, source: DeviceSource) -> None:
        self.snapshot = DeviceSnapshot(device_id=device_id, source=source)

    @abstractmethod
    async def connect(self) -> dict[str, Any]: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    async def status(self) -> dict[str, Any]:
        return self.snapshot.as_dict()



class RtcDeviceAdapter(DeviceAdapter):
    """Shared device lifecycle/events around the fixed-frame RTC protocol client."""

    def __init__(self, source: DeviceSource, *, events: EventBus | None = None,
                 frame_decoder: Callable[[bytes], dict[str, Any] | None] | None = None) -> None:
        super().__init__("rtc", source)
        self.client: Any = None
        self._events = events
        self._frame_decoder = frame_decoder
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._event_tasks: set[asyncio.Task] = set()
        self._antenna_exchange_lock = asyncio.Lock()

    def _on_client_event(self, event: dict[str, Any]) -> None:
        # Serial callbacks may originate in the reader thread. EventBus queues and
        # snapshot mutation belong to the service event loop only.
        if self._event_loop is not None and not self._event_loop.is_closed():
            self._event_loop.call_soon_threadsafe(self._schedule_event, dict(event))

    def _schedule_event(self, event: dict[str, Any]) -> None:
        task = asyncio.create_task(self._publish_client_event(event))
        self._event_tasks.add(task)
        task.add_done_callback(self._event_tasks.discard)

    async def _publish_client_event(self, event: dict[str, Any]) -> None:
        kind = str(event.get("event", ""))
        if kind.lower() in {"fault", "transport_error"}:
            self.snapshot.update(state=DeviceState.FAULT, rtc_event=event)
        else:
            self.snapshot.update(rtc_event=event)
        if self._events is None:
            return
        raw = str(event.get("raw_hex", ""))
        if raw and kind.lower() == "raw":
            await self._events.publish("device.raw", device_id="rtc", direction=event.get("direction", "RX"),
                                       raw_hex=raw, transport="SERIAL", context="RTC_FRAME", command_id=None)
        payload_hex = event.get("payload_hex")
        if payload_hex:
            payload = bytes.fromhex(payload_hex)
            parsed = None
            if self._frame_decoder is not None:
                try:
                    parsed = self._frame_decoder(payload)
                except Exception:
                    pass
            # E2 payload is independently received antenna data, not the B0 TX ack.
            await self._events.publish("device.raw", device_id="rtc", direction="RX", raw_hex=payload.hex(" ").upper(),
                                       transport="RTC_ANTENNA", context="ANTENNA_RECEIVE", parsed=parsed, command_id=None)
        if kind.lower() != "raw":
            await self._events.publish("device.rtc", device_id="rtc", rtc_event=event)

    async def connect(self) -> dict[str, Any]:
        self._event_loop = asyncio.get_running_loop()
        self.snapshot.update(state=DeviceState.CONNECTING)
        try:
            details = await self.client.connect()
            self.snapshot.identity = f"RTC-V1.0:{self.client.endpoint.port}@{self.client.endpoint.baudrate}"
            self.snapshot.update(state=DeviceState.READY, **{**details, "protocol": "1.0", "transport": "SERIAL"})
            return self.snapshot.as_dict()
        except Exception as exc:
            self.snapshot.update(state=DeviceState.FAULT, error=str(exc))
            raise

    async def disconnect(self) -> None:
        if self.snapshot.state != DeviceState.DISCONNECTED:
            # STOP acknowledgements mean accepted, not drained. Keep serial RX and
            # activity monitoring alive until queried flags confirm outputs have stopped.
            status = await self.get_status()
            if status.get("debug_context") or status.get("state") == "DEBUG_TR":
                await self.client.stop_debug_tr()
            elif status.get("state") in {"ARMED", "RUNNING", "STOPPING"}:
                if status.get("state") != "STOPPING":
                    await self.client.stop_graceful()
            deadline = time.monotonic() + 15
            while True:
                status = await self.get_status()
                if not any(status.get(key) for key in ("group_in_flight", "tx_busy", "io_busy", "tr_running", "tr_stopping")):
                    break
                if time.monotonic() >= deadline:
                    raise ServiceError("TIMEOUT", "RTC 输出尚未确认停止，保持串口以便查询", "rtc_disconnect", "rtc",
                                       status, side_effect_possible=True, next_action="等待排空或按现场情况执行立即停止；不要重发原命令")
                await asyncio.sleep(0.05)
        await self.client.close()
        await asyncio.sleep(0)
        if self._event_tasks:
            await asyncio.gather(*tuple(self._event_tasks), return_exceptions=True)
        self._event_loop = None
        self.snapshot.update(state=DeviceState.DISCONNECTED)

    async def get_status(self) -> dict[str, Any]:
        result = await self.client.get_status()
        if result.get("fault_code") or result.get("state") == "FAULT" or result.get("activity_monitor_error"):
            state = DeviceState.FAULT
        elif result.get("unknown_result") or result.get("result_uncertain"):
            state = DeviceState.UNKNOWN
        else:
            state = DeviceState.READY
        self.snapshot.update(state=state, rtc_status=result)
        return result

    async def write_wave_entry(self, address: int, antenna_frame: bytes) -> None:
        await self.client.write_wave_entry(address, antenna_frame)

    async def read_wave_entry(self, address: int) -> bytes:
        return await self.client.read_wave_entry(address)

    async def transfer_antenna(self, antenna_frame: bytes) -> dict[str, Any]:
        async with self._antenna_exchange_lock:
            return await self.client.transfer_antenna(antenna_frame)

    async def send_frame(
        self,
        frame: bytes,
        *,
        timeout_ms: int = 1000,
        response_opcode: int | None = None,
        response_rule: Any = None,
        response_matcher: Callable[[bytes], bool] | None = None,
    ) -> bytes | list[bytes]:
        """Transfer one antenna request and separately confirm its E2 response.

        B0 confirms physical TX completion, never antenna success. The local RX
        cursor excludes earlier E2 frames, and opcode/array/CRC plus the optional
        request-specific matcher exclude unrelated replies. The caller still
        validates decoded business fields using the selected protocol profile.
        """
        from antenna_service.protocol.crc import verify_crc_be
        from antenna_service.protocol.profile import ResponseAssembly

        if len(frame) != 22 or frame[:4] != b"\xAA\x55\x00\x16" or not verify_crc_be(frame):
            raise invalid("RTC 波控应答交换只接受已编译且CRC有效的22字节AA55帧", stage="rtc_antenna_response")
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or not 1 <= timeout_ms <= 65535:
            raise invalid("波控应答超时必须为1..65535毫秒", stage="rtc_antenna_response")
        expected_opcode = frame[4] if response_opcode is None else response_opcode
        if isinstance(expected_opcode, bool) or not isinstance(expected_opcode, int) or not 0 <= expected_opcode <= 255:
            raise invalid("波控应答操作码无效", stage="rtc_antenna_response")
        multi = response_rule is not None and response_rule.mode == "MULTI"
        if multi:
            timeout_ms = min(timeout_ms, response_rule.timeout_ms)
        assembly = None

        def matches(candidate: bytes) -> bool:
            nonlocal assembly
            if (len(candidate) != 22 or candidate[:4] != b"\xAA\x55\x00\x16"
                    or candidate[4] != expected_opcode or candidate[5] != frame[5]
                    or not verify_crc_be(candidate)):
                return False
            if response_matcher is not None and not response_matcher(candidate):
                return False
            if multi:
                if assembly is None:
                    assembly = ResponseAssembly(response_rule)
                return assembly.add(candidate)
            return True

        async with self._antenna_exchange_lock:
            cursor = self.client.antenna_rx_cursor()
            tx_confirmed = False
            try:
                # 0x30 is sent once. Register the local cursor first so E2 that
                # arrives before B0 remains available, without clearing RX/logs.
                await self.client.transfer_antenna(frame)
                tx_confirmed = True
                response = await self.client.wait_antenna_response(cursor, matches, timeout_ms=timeout_ms)
                return assembly.frames if assembly is not None else response
            except ServiceError as exc:
                if tx_confirmed:
                    exc.side_effect_possible = True
                    exc.next_action = "核对波控状态；不要盲目重发原指令"
                    self.snapshot.update(state=DeviceState.UNKNOWN, antenna_response_error=exc.as_dict())
                raise


    async def get_capability(self) -> dict[str, Any]:
        return await self.client.get_capability()

    async def get_tr_config(self) -> dict[str, Any]:
        return await self.client.get_tr_config()

    async def get_timing(self) -> dict[str, Any]:
        return await self.client.get_timing()

    async def get_config(self) -> dict[str, Any]:
        return await self.client.get_config()

    async def get_progress(self) -> dict[str, Any]:
        return await self.client.get_progress()

    async def get_antenna_io_status(self) -> dict[str, Any]:
        return await self.client.get_antenna_io_status()

    async def set_timing(self, **parameters: Any) -> dict[str, Any]:
        return await self.client.set_timing(**parameters)

    async def set_tr_config(self, mode: int | str, period_us: float, high_us: float, delay_us: float = 1) -> dict[str, Any]:
        return await self.client.set_tr_config(mode, period_us, high_us, delay_us)

    async def set_counts(self, wave_count: int, points_per_wave: int, start_address: int = 1) -> dict[str, Any]:
        return await self.client.set_counts(wave_count, points_per_wave, start_address)

    async def set_trigger_mode(self, mode: int | str) -> dict[str, Any]:
        return await self.client.set_trigger_mode(mode)

    async def arm(self) -> dict[str, Any]:
        return await self.client.arm()

    async def software_trigger(self) -> dict[str, Any]:
        return await self.client.software_trigger()

    async def stop_graceful(self) -> dict[str, Any]:
        return await self.client.stop_graceful()

    async def stop_immediate(self) -> dict[str, Any]:
        return await self.client.stop_immediate()

    async def clear_fault(self) -> dict[str, Any]:
        result = await self.client.clear_fault()
        await self.get_status()
        return result

    async def start_debug_tr(self) -> dict[str, Any]:
        return await self.client.start_debug_tr()

    async def stop_debug_tr(self) -> dict[str, Any]:
        return await self.client.stop_debug_tr()


def azimuth_scan_geometry(start_deg: float, end_deg: float, step_deg: float, speed: float) -> tuple[int, float]:
    """Validate a continuous row from the existing angle/step/speed fields."""
    if not all(math.isfinite(value) for value in (start_deg, end_deg, step_deg, speed)):
        raise invalid("连续扫描参数必须为有限值", stage="turntable_scan")
    if start_deg == end_deg:
        raise invalid("连续扫描至少需要两个方位点；单点请使用走停模式", stage="turntable_scan")
    if step_deg < 0.0001 or speed <= 0 or speed > 20 or round(speed, 4) != speed:
        raise invalid("连续扫描步进至少 0.0001°，速度须大于 0、至多四位小数且不超过 20°/s", stage="turntable_scan")
    intervals = abs(end_deg - start_deg) / step_deg
    count = round(intervals)
    if count < 1 or not math.isclose(intervals, count, rel_tol=0, abs_tol=1e-7):
        raise invalid("连续扫描的角度范围必须可被方位步进整除", stage="turntable_scan")
    return count, math.copysign(step_deg, end_deg - start_deg)
