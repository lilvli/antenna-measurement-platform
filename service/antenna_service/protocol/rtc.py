from __future__ import annotations

import asyncio
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from antenna_service.errors import ServiceError, invalid
from antenna_service.protocol.crc import append_crc_be, verify_crc_be

SOF = b"\xA5\x5A"
FIXED_LENGTH = 32
OPCODES = {name: value for value, name in enumerate((
    "GET_INFO", "GET_CAPABILITY", "GET_STATUS", "SET_COUNTS", "SET_TIMING",
    "SET_TRIGGER_MODE", "ARM", "SOFTWARE_TRIGGER", "STOP_GRACEFUL",
    "STOP_IMMEDIATE", "CLEAR_FAULT", "GET_LAST_TIME", "GET_PROGRESS", "PING",
    "GET_CONFIG", "GET_TIMING", "SET_TR_CONFIG", "GET_TR_CONFIG",
    "GET_ANTENNA_INTERFACE", "SET_ANTENNA_INTERFACE", "START_DEBUG_TR", "STOP_DEBUG_TR",
), 1)}
OPCODES.update(WRITE_WAVE_ENTRY=0x21, READ_WAVE_ENTRY=0x24,
               ANTENNA_TRANSFER_22=0x30, GET_ANTENNA_IO_STATUS=0x32)
QUERIES = {0x01, 0x02, 0x03, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x12, 0x13, 0x24, 0x32}
RTC_STATES = {0: "RESET", 1: "IDLE", 3: "CONFIGURED", 4: "ARMED", 5: "RUNNING",
              6: "STOPPING", 7: "COMPLETE", 8: "FAULT", 9: "ANTENNA_DEBUG"}
ERRORS = {1: "BAD_LENGTH", 2: "BAD_CRC", 3: "BAD_OPCODE", 4: "BAD_PARAMETER",
          5: "INVALID_STATE", 6: "BUSY", 7: "WAVE_ADDRESS_ERROR", 8: "WAVE_ENTRY_NOT_VALID",
          0x0A: "ANTENNA_TX_TIMEOUT", 0x0B: "VNA_READY_TIMEOUT", 0x0C: "TR_WINDOW_ERROR",
          0x0D: "ANTENNA_INTERFACE_ERROR", 0x0E: "ANTENNA_TX_FIFO_OVERFLOW",
          0x0F: "ANTENNA_RX_FIFO_OVERFLOW", 0x13: "TRIGGER_OVERRUN", 0x14: "HOST_TIMEOUT",
          0x15: "SYNC_IO_TIMEOUT", 0x16: "COUNTER_OVERFLOW", 0x17: "IO_NOT_READY", 0x18: "ABORTED"}
TIMING_FIELDS = ("pulse_qualification_us", "host_timeout_ms", "antenna_tx_timeout_ms",
                 "vna_ready_timeout_ms", "ack_timeout_ms", "stable_wait_us", "sync_io_timeout_ms")


def _u(data: bytes, start: int, size: int = 2) -> int:
    return int.from_bytes(data[start:start + size], "big")


def _integer(value: int, low: int, high: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise invalid(f"{name} 必须是 {low}..{high} 的整数", stage="rtc_encode", target=name)
    return value


def build_fixed(opcode: int, *, control: int = 0, aux: int = 0, data: bytes = b"") -> bytes:
    _integer(opcode, 0, 255, "opcode")
    _integer(control, 0, 255, "control")
    _integer(aux, 0, 65535, "aux")
    if len(data) > 22:
        raise invalid("RTC 固定帧数据区最多 22 字节", stage="rtc_encode")
    return append_crc_be(SOF + b"\x00\x20" + bytes([opcode, control]) + aux.to_bytes(2, "big") + bytes(data).ljust(22, b"\x00"))


def _status_flags(flags: int) -> dict[str, bool]:
    return dict(zip(("tx_busy", "group_in_flight", "tr_running", "debug_context",
                     "result_uncertain", "tr_stopping", "io_busy"), (bool(flags & (1 << bit)) for bit in range(7))))



def _validate_response(frame: bytes) -> None:
    """Validate V1.0 reserved bytes after transaction correlation, before success."""
    op, d = frame[4], frame[8:30]
    bad = False
    if op == 0x7F:
        bad = any(d[6:])
    elif op != 0x8E:  # PING echoes every D5..D29 byte verbatim.
        bad = frame[5] != 0 or (op not in {0xA1, 0xA4} and any(frame[6:8]))
        tail = {0x81: 8, 0x82: 16, 0x84: 8, 0x85: 14, 0x86: 1,
                0x87: 2, 0x88: 2, 0x89: 2, 0x8A: 2, 0x8B: 2, 0x8C: 5,
                0x8F: 11, 0x90: 14, 0x91: 10, 0x92: 12, 0x93: 1,
                0x94: 1, 0x95: 2, 0x96: 2, 0xB0: 6}.get(op)
        bad = bad or (tail is not None and any(d[tail:]))
        if op in {0x84, 0x8F}:
            bad = bad or any(d[:2])  # External source S is reserved in V1.0.
        if op == 0x83:
            bad = bad or any(d[6:8]) or bool(d[18] & 0x80) or bool(d[21] & 0xF8)
            bad = bad or d[0] not in RTC_STATES or d[1] > 0x0B or d[4] != 0 or d[5] > 2 or d[19] > 2
        if op == 0x82:
            bad = bad or bool(_u(d, 12, 4) & ~0x1F)
        if op == 0x86:
            bad = bad or d[0] > 1
        if op == 0x8D:
            bad = bad or d[20] > 1 or bool(d[21] & 0xF0)
        if op == 0x8F:
            bad = bad or d[8] > 1 or d[9] != 0 or d[10] > 2
        if op in {0x91, 0x92}:
            bad = bad or d[1] != 0 or d[0] > 2
            if op == 0x92:
                bad = bad or d[10] > 3 or d[11] > 2
        if op in {0x93, 0x94}:
            bad = bad or d[0] != 0
        if op in {0x87, 0x88, 0x89, 0x8A, 0x8B}:
            bad = bad or d[0] not in RTC_STATES or d[1] > 0x0B
        if op in {0x95, 0x96}:
            bad = bad or d[0] > 3 or d[1] not in RTC_STATES
        if op == 0xB2:
            bad = bad or d[20] != 0 or bool(d[21] & 0xF8)
    if bad:
        raise ServiceError("DATA_INTEGRITY", "RTC 应答保留位或枚举字段不符合 V1.0", "rtc_decode",
                           details={"opcode": op, "raw_hex": frame.hex(" ").upper()})


def parse_frame(frame: bytes) -> dict[str, Any]:
    if len(frame) != 32 or frame[:4] != SOF + b"\x00\x20":
        raise ServiceError("DATA_INTEGRITY", "RTC 帧头或长度错误", "rtc_decode")
    if not verify_crc_be(frame):
        raise ServiceError("DATA_INTEGRITY", "RTC 外层 CRC 错误", "rtc_decode")
    opcode, data = frame[4], frame[8:30]
    result: dict[str, Any] = {"opcode": opcode, "control": frame[5], "aux": _u(frame, 6),
                              "data": data, "raw_hex": frame.hex(" ").upper()}
    if opcode == 0x7F:
        result["nack"] = {"error_code": frame[5], "error": ERRORS.get(frame[5], "UNKNOWN"),
                          "rejected_opcode": frame[6], "detail": frame[7],
                          "state": RTC_STATES.get(data[0], "UNKNOWN"), "stage": data[1],
                          "parameter": _u(data, 2), "request_crc": _u(data, 4)}
    elif opcode == 0xE2:
        if frame[5] != 22:
            raise ServiceError("DATA_INTEGRITY", "RTC E2 有效长度必须为 22", "rtc_decode")
        result.update(event="ANTENNA_RX", sequence=_u(frame, 6), payload_hex=data.hex(" ").upper())
    elif opcode == 0xE0:
        result.update(event="DONE", reason=frame[5], trigger_mode=frame[6], stage=frame[7],
                      elapsed_us=_u(data, 0, 4), completed_groups=_u(data, 4, 4),
                      completed_points=_u(data, 8, 4), accepted_groups=_u(data, 12, 4),
                      valid_triggers=_u(data, 16, 4), flags=data[20],
                      state=RTC_STATES.get(data[21], "UNKNOWN"), **_status_flags(data[20]))
    elif opcode == 0xE1:
        result.update(event="FAULT", fault_code=frame[5], error=ERRORS.get(frame[5], "UNKNOWN"),
                      stage=frame[6], source=frame[7], elapsed_us=_u(data, 0, 4),
                      completed_groups=_u(data, 4, 4), completed_points=_u(data, 8, 4),
                      wave_address=_u(data, 12), vna_trigger_index=_u(data, 14),
                      valid_triggers=_u(data, 16, 4), flags=data[20],
                      state=RTC_STATES.get(data[21], "UNKNOWN"), **_status_flags(data[20]))
    return result


def parse_status(frame: bytes) -> dict[str, Any]:
    parsed = parse_frame(frame)
    if parsed["opcode"] != 0x83:
        raise ServiceError("DATA_INTEGRITY", "RTC GET_STATUS 应答 Opcode 错误", "rtc_decode")
    d = parsed["data"]
    return {"state_code": d[0], "state": RTC_STATES.get(d[0], "UNKNOWN"), "stage": d[1],
            "fault_code": d[2], "last_nack": d[3], "antenna_interface": d[4], "tr_mode": d[5],
            "wave_address": _u(d, 8), "vna_trigger_index": _u(d, 10), "elapsed_us": _u(d, 12, 4),
            "valid_wave_entries": _u(d, 16), "flags": d[18], "rdy": d[19],
            "last_result": d[20], "config_valid": d[21], **_status_flags(d[18])}


class RtcFrameDecoder:
    """Bounded serial stream framing; a valid payload may itself contain A5 5A."""
    def __init__(self, frame_timeout_s: float = 0.1):
        self.buffer = bytearray()
        self.frame_timeout_s = frame_timeout_s
        self.started_at: float | None = None
        self.bad_frames = 0
        self.discarded_bytes = 0

    def feed(self, data: bytes = b"", *, now: float | None = None) -> list[bytes]:
        now = time.monotonic() if now is None else now
        # Expire the old candidate before attaching late bytes to it.
        if self.started_at is not None and now - self.started_at >= self.frame_timeout_s:
            del self.buffer[:1]
            self.started_at = None
            self.bad_frames += 1
        self.buffer.extend(data)
        frames = []
        while self.buffer:
            start = self.buffer.find(SOF)
            if start < 0:
                keep = 1 if self.buffer[-1] == SOF[0] else 0
                self.discarded_bytes += len(self.buffer) - keep
                self.buffer[:] = self.buffer[-keep:] if keep else b""
                self.started_at = None
                break
            if start:
                self.discarded_bytes += start
                del self.buffer[:start]
                self.started_at = None
            if self.started_at is None:
                self.started_at = now
            if len(self.buffer) < 4:
                break
            if self.buffer[2:4] != b"\x00\x20":
                del self.buffer[:1]
                self.started_at = None
                self.bad_frames += 1
                continue
            if len(self.buffer) < 32:
                break
            candidate = bytes(self.buffer[:32])
            if not verify_crc_be(candidate):
                del self.buffer[:1]
                self.started_at = None
                self.bad_frames += 1
                continue
            frames.append(candidate)
            del self.buffer[:32]
            self.started_at = None
        return frames


@dataclass(slots=True)
class RtcEndpoint:
    port: str
    baudrate: int = 115200
    timeout_ms: int = 1000

    def __post_init__(self):
        if not isinstance(self.port, str) or not self.port.strip():
            raise invalid("RTC 需要串口名称", stage="rtc_connect", target="port")
        _integer(self.baudrate, 1, 4_000_000, "baudrate")
        _integer(self.timeout_ms, 10, 65535, "timeout_ms")


@dataclass
class _Pending:
    request: bytes
    ready: threading.Event = field(default_factory=threading.Event)
    response: bytes | None = None
    error: Exception | None = None


class RtcClient:
    """V1.0 8N1 serial client. No side-effect command is automatically replayed.

    Reader and activity monitor have their own threads, including while the async caller
    waits for a VNA buffer. A single action may overlap a serialized read-only
    request; each complete 32-byte write is protected against byte interleaving.
    """
    def __init__(self, endpoint: RtcEndpoint, *, on_event: Callable[[dict[str, Any]], None] | None = None,
                 serial_factory: Callable[..., Any] | None = None):
        self.endpoint, self.on_event, self.serial_factory = endpoint, on_event, serial_factory
        self.received_events: deque[dict[str, Any]] = deque(maxlen=256)
        self._serial: Any = None
        self._pending: dict[int, _Pending] = {}
        self._pending_lock = threading.Lock()
        self._write_lock, self._action_lock, self._query_lock = threading.Lock(), threading.Lock(), threading.Lock()
        self._shutdown = threading.Event()
        self._activity_wakeup = threading.Event()
        self._reader: threading.Thread | None = None
        self._activity_monitor: threading.Thread | None = None
        self._decoder = RtcFrameDecoder(max(0.1, 32 * 10 / endpoint.baudrate * 3))
        self._timing: dict[str, int] = {}
        self._capability: dict[str, Any] | None = None
        self.unknown_result: dict[str, Any] | None = None
        self.activity_monitor_error: dict[str, Any] | None = None
        self._transport_error: ServiceError | None = None
        self._activity_interval_s = 0.5
        self._activity_lock = threading.Lock()
        self._observed_activity = False
        self._active_requests = 0
        self._last_valid_request_at = time.monotonic()
        self._last_monitor_fault: tuple[Any, ...] | None = None
        self._antenna_rx_condition = threading.Condition()
        self._antenna_rx_sequence = 0
        self._antenna_rx_frames: deque[tuple[int, bytes]] = deque(maxlen=1024)

    async def connect(self) -> dict[str, Any]:
        if self._serial is not None:
            await self.close()
        self._shutdown.clear()
        self._activity_wakeup.clear()
        self._transport_error = None
        self.activity_monitor_error = None
        self.received_events.clear()
        with self._activity_lock:
            self._observed_activity = False
            self._active_requests = 0
            self._last_monitor_fault = None
        with self._antenna_rx_condition:
            self._antenna_rx_frames.clear()
        self._decoder = RtcFrameDecoder(max(0.1, 32 * 10 / self.endpoint.baudrate * 3))
        def open_port():
            if self.serial_factory is None:
                import serial
                factory = serial.Serial
            else:
                factory = self.serial_factory
            # Opening RS232 only; 8 data bits, no parity, 1 stop bit, no flow
            # control. write_timeout is seconds. Never emit reset/ARM on connect.
            self._serial = factory(port=self.endpoint.port, baudrate=self.endpoint.baudrate,
                                   bytesize=8, parity="N", stopbits=1, timeout=0.02,
                                   write_timeout=self.endpoint.timeout_ms / 1000,
                                   xonxoff=False, rtscts=False, dsrdtr=False)
        try:
            await asyncio.to_thread(open_port)
            self._reader = threading.Thread(target=self._receive_loop, daemon=True, name="rtc-rx")
            self._reader.start()
            info = await self.get_info()
            if (info["protocol_major"], info["protocol_minor"], info["device_type"]) != (1, 0, 1):
                raise ServiceError("DEVICE_FAULT", "RTC 不是固定 32 字节 V1.0 设备", "rtc_connect", details=info)
            capability = await self.get_capability()
            if capability["wave_entry_bytes"] != 22 or capability["antenna_frame_bytes"] != 22 or capability["clock_hz"] <= 0:
                raise ServiceError("DEVICE_FAULT", "RTC 能力或实际时钟无效", "rtc_connect", details=capability)
            await self.ping()
            await self.get_timing()
            status = await self.get_status()
            self._activity_monitor = threading.Thread(target=self._activity_monitor_loop, daemon=True, name="rtc-activity-monitor")
            self._activity_monitor.start()
            return {"protocol": "1.0", "firmware": info["firmware"], "status": status, "capability": capability}
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        # Closing transport does not send STOP and is not proof of output drain.
        # Callers must finish device shutdown before disconnecting.
        self._shutdown.set()
        self._activity_wakeup.set()
        with self._antenna_rx_condition:
            self._antenna_rx_condition.notify_all()
        transport, self._serial = self._serial, None
        if transport is not None:
            await asyncio.to_thread(transport.close)
        self._fail_pending(ServiceError("DEVICE_DISCONNECTED", "RTC 串口已断开", "rtc_receive"))
        for thread in (self._reader, self._activity_monitor):
            if thread is not None and thread is not threading.current_thread():
                await asyncio.to_thread(thread.join, 1.5)

    def _fail_pending(self, error: Exception):
        with self._pending_lock:
            for pending in self._pending.values():
                pending.error = error
                pending.ready.set()

    def _notify(self, event: dict[str, Any]):
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                # Observers must not block or terminate transport handling.
                pass

    def _receive_loop(self):
        try:
            while not self._shutdown.is_set():
                transport = self._serial
                if transport is None:
                    break
                for frame in self._decoder.feed(transport.read(256)):
                    self._notify({"event": "raw", "direction": "RX", "raw_hex": frame.hex(" ").upper()})
                    try:
                        parsed = parse_frame(frame)
                    except ServiceError:
                        continue
                    if "event" in parsed:
                        event = {key: value for key, value in parsed.items() if key != "data"}
                        if parsed["event"] == "ANTENNA_RX":
                            with self._antenna_rx_condition:
                                self._antenna_rx_sequence += 1
                                self._antenna_rx_frames.append((self._antenna_rx_sequence, parsed["data"]))
                                event["receive_cursor"] = self._antenna_rx_sequence
                                self._antenna_rx_condition.notify_all()
                        else:
                            self._observe_activity(parsed)
                        self.received_events.append(event)
                        event["direction"] = "RX"
                        self._notify(event)
                        continue
                    opcode = frame[6] if frame[4] == 0x7F else frame[4] & 0x7F
                    with self._pending_lock:
                        pending = self._pending.get(opcode)
                        if pending is not None and self._matches(pending.request, frame):
                            try:
                                _validate_response(frame)
                            except ServiceError:
                                pass  # The waiting transaction reports malformed ACK.
                            else:
                                if frame[4] != 0x7F:
                                    with self._activity_lock:
                                        self._last_valid_request_at = time.monotonic()
                                    self._observe_response(frame)
                                    self._activity_wakeup.set()
                            pending.response = frame
                            pending.ready.set()
        except Exception as exc:
            if not self._shutdown.is_set():
                error = ServiceError("DEVICE_DISCONNECTED", str(exc), "rtc_receive", self.endpoint.port)
                self._transport_error = error
                self.activity_monitor_error = error.as_dict()
                self._fail_pending(error)
                with self._antenna_rx_condition:
                    self._antenna_rx_condition.notify_all()
                self._notify({"event": "transport_error", "error": error.as_dict()})
                self._activity_wakeup.set()

    @staticmethod
    def _matches(request: bytes, response: bytes) -> bool:
        if response[4] == 0x7F:
            # CRC is diagnostic only; opcode is the protocol correlation field.
            return response[6] == request[4]
        if response[4] != request[4] | 0x80:
            return False
        if request[4] in {0x21, 0x24} and response[6:8] != request[6:8]:
            return False
        if request[4] == 0x0E and response[5:30] != request[5:30]:
            return False
        return True

    def _observe_activity(self, state: dict[str, Any]) -> None:
        active = state.get("state") in {"ARMED", "RUNNING", "STOPPING", "ANTENNA_DEBUG"} or any(
            state.get(key) for key in ("tx_busy", "group_in_flight", "tr_running", "tr_stopping", "io_busy"))
        with self._activity_lock:
            self._observed_activity = active
        self._activity_wakeup.set()

    def _observe_response(self, response: bytes) -> None:
        op, d = response[4], response[8:30]
        if op == 0x83:
            self.activity_monitor_error = None
            self._observe_activity(parse_status(response))
        elif op in {0x87, 0x88, 0x8B}:
            self._observe_activity({"state": RTC_STATES.get(d[0])})
        elif op in {0x95, 0x96}:
            self._observe_activity({"state": RTC_STATES.get(d[1]), "tr_running": d[0] in {1, 2, 3}})
        elif op == 0xB0:
            self._observe_activity({"state": RTC_STATES.get(d[5])})
        # STOP ACK only accepts a request. Do not overwrite a preceding DONE
        # event with the transitional STOPPING state carried by its late ACK.

    def _monitor_due(self) -> bool:
        with self._activity_lock:
            return ((self._observed_activity or self._active_requests > 0)
                    and time.monotonic() - self._last_valid_request_at >= self._activity_interval_s)

    def _activity_monitor_loop(self):
        # This reads live state and latched faults only while RTC outputs or an
        # ARM/TR/TX action may be active. Idle connections send no periodic traffic.
        # Any successful foreground request postpones this status read. The worker
        # still monitors during a blocking VNA read, without sending periodic PING.
        while not self._shutdown.is_set():
            with self._activity_lock:
                active = self._observed_activity or self._active_requests > 0
                delay = max(0.001, self._activity_interval_s - (time.monotonic() - self._last_valid_request_at)) if active else None
            self._activity_wakeup.wait(delay)
            self._activity_wakeup.clear()
            if self._shutdown.is_set() or self._transport_error is not None:
                break
            if not self._monitor_due():
                continue
            try:
                response = self._transact_blocking(build_fixed(0x03), monitor_only=True)
                if not response:
                    continue
                status = parse_status(response)
                self.activity_monitor_error = None
                if status["fault_code"] or status["state"] == "FAULT":
                    key = (status["fault_code"], status["stage"], status["state"])
                    if key != self._last_monitor_fault:
                        self._notify({"event": "FAULT", **status, "detected_by": "activity_status"})
                    self._last_monitor_fault = key
                else:
                    self._last_monitor_fault = None
            except ServiceError as exc:
                if self.activity_monitor_error is None and self._transport_error is None:
                    self._notify({"event": "transport_error", "error": exc.as_dict()})
                self.activity_monitor_error = exc.as_dict()
                # A failed monitor query is observable and is never a reason to
                # replay an action. Bound read-only monitoring even on fast errors.
                self._activity_wakeup.wait(self._activity_interval_s)
                self._activity_wakeup.clear()

    def antenna_rx_cursor(self) -> int:
        """Snapshot local RX order; RTC's 16-bit E2 sequence may wrap or reset."""
        with self._antenna_rx_condition:
            return self._antenna_rx_sequence

    async def wait_antenna_response(self, cursor: int, matcher: Callable[[bytes], bool], *, timeout_ms: int) -> bytes:
        """Collect only E2 received after the cursor; physical B0 is independent.

        Callers take the cursor before sending one request and serialize exchanges.
        The protocol has no transaction ID: an identical late reply from hardware
        cannot be distinguished from a new reply, so an uncertain command is never
        retried here. The caller must perform explicit recovery after a timeout.
        """
        _integer(timeout_ms, 1, 65535, "antenna_response_timeout_ms")
        return await asyncio.to_thread(self._wait_antenna_response, cursor, matcher, timeout_ms)

    def _wait_antenna_response(self, cursor: int, matcher: Callable[[bytes], bool], timeout_ms: int) -> bytes:
        deadline = time.monotonic() + timeout_ms / 1000
        with self._antenna_rx_condition:
            while True:
                if self._transport_error is not None:
                    raise self._transport_error
                if self._shutdown.is_set() or self._serial is None:
                    raise ServiceError("DEVICE_DISCONNECTED", "RTC 串口已断开", "rtc_antenna_response")
                if self._antenna_rx_frames and self._antenna_rx_frames[0][0] > cursor + 1:
                    raise ServiceError("DATA_INTEGRITY", "RTC 波控回包接收缓存溢出，不能确认应答完整", "rtc_antenna_response")
                for sequence, payload in self._antenna_rx_frames:
                    if sequence <= cursor:
                        continue
                    cursor = sequence
                    if matcher(payload):
                        return payload
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ServiceError("TIMEOUT", "RTC 已发送波控指令，但未收到匹配的波控应答", "rtc_antenna_response",
                                       side_effect_possible=True, next_action="核对波控状态；不要盲目重发原命令")
                self._antenna_rx_condition.wait(remaining)

    async def transact(self, request: bytes) -> bytes:
        # Queries may be explicitly requested again, but this adapter itself
        # sends each request only once; side effects are never replayed.
        return await asyncio.to_thread(self._transact_blocking, request)

    def _transact_blocking(self, request: bytes, *, monitor_only: bool = False) -> bytes:
        parse_frame(request)
        opcode = request[4]
        if opcode not in OPCODES.values():
            raise invalid("RTC V1.0 未定义此操作码", stage="rtc_send", opcode=opcode)
        query = opcode in QUERIES
        with self._query_lock if query else self._action_lock:
            # A foreground query may have renewed contact while this worker was
            # waiting for the query lock. Recheck before adding any wire traffic.
            if monitor_only and not self._monitor_due():
                return b""
            if self._transport_error is not None:
                raise ServiceError("DEVICE_DISCONNECTED", "RTC 接收链路已断开", "rtc_send", self.endpoint.port)
            if self._serial is None or self._shutdown.is_set():
                raise ServiceError("NOT_RUNNABLE", "RTC 尚未连接", "rtc_send", self.endpoint.port)
            if not query and self.unknown_result is not None:
                raise ServiceError("UNKNOWN_RESULT", "RTC 上一动作结果未知，请先查询核对", "rtc_send",
                                   details=self.unknown_result, side_effect_possible=True,
                                   next_action="核对状态、计数与 I/O 后明确确认结果，不能盲目重发")
            pending = _Pending(request)
            with self._pending_lock:
                self._pending[opcode] = pending
            active_request = opcode in {0x07, 0x08, 0x09, 0x0A, 0x15, 0x16, 0x30}
            if active_request:
                with self._activity_lock:
                    self._active_requests += 1
                    self._observed_activity = True
                self._activity_wakeup.set()
            try:
                with self._write_lock:
                    # A serial write can partially reach hardware before failing.
                    # One full frame is attempted once, never reset/retry a tail.
                    try:
                        written = self._serial.write(request)
                    except Exception:
                        self._notify({"event": "raw", "direction": "TX", "raw_hex": request.hex(" ").upper(), "write_uncertain": True})
                        raise
                    self._notify({"event": "raw", "direction": "TX", "raw_hex": request.hex(" ").upper(), "write_uncertain": written != len(request)})
                    if written != len(request):
                        raise ServiceError("DEVICE_FAULT", "RTC 串口未发送完整帧", "rtc_send")
                timeout_s = max(self.endpoint.timeout_ms, self._timing.get("ack_timeout_ms", 0)) / 1000
                if opcode == 0x30:
                    timeout_s = max(timeout_s, (self._timing.get("antenna_tx_timeout_ms", 0) + self.endpoint.timeout_ms) / 1000)
                if opcode == 0x07:
                    timeout_s = max(timeout_s, (self._timing.get("sync_io_timeout_ms", 0) + self.endpoint.timeout_ms) / 1000)
                if not pending.ready.wait(timeout_s):
                    raise ServiceError("TIMEOUT", "RTC 应答超时", "rtc_wait_ack", self.endpoint.port)
                if pending.error is not None:
                    raise pending.error
                assert pending.response is not None
                parsed = parse_frame(pending.response)
                _validate_response(pending.response)
                if "nack" in parsed:
                    nack = parsed["nack"]
                    raise ServiceError("DEVICE_FAULT", f"RTC 拒绝请求：{nack['error']}", "rtc_response", details=nack)
                return pending.response
            except Exception as exc:
                # A NACK is a confirmed rejection, all other failures after write
                # leave the side-effect result unknown until explicit reconciliation.
                confirmed_nack = (pending.response is not None and pending.response[4] == 0x7F
                                  and pending.response[5] not in {0x0A, 0x0B, 0x0C, 0x14, 0x15, 0x18}
                                  and isinstance(exc, ServiceError) and exc.stage == "rtc_response")
                if confirmed_nack and active_request:
                    self._observe_activity(parse_frame(pending.response)["nack"])
                if not query and not confirmed_nack:
                    self.unknown_result = {"opcode": opcode, "request_hex": request.hex(" ").upper()}
                    if isinstance(exc, ServiceError):
                        exc.side_effect_possible = True
                        exc.next_action = "请查询 RTC 状态、进度和 I/O，不要重发原动作"
                if isinstance(exc, ServiceError):
                    raise
                raise ServiceError("DEVICE_FAULT", str(exc), "rtc_send", self.endpoint.port,
                                   side_effect_possible=not query) from exc
            finally:
                if active_request:
                    with self._activity_lock:
                        self._active_requests -= 1
                    self._activity_wakeup.set()
                with self._pending_lock:
                    self._pending.pop(opcode, None)

    def acknowledge_unknown_result(self) -> None:
        """Caller explicitly reconciled status/counters/I/O; never called on timeout."""
        if self._action_lock.locked():
            raise ServiceError("BUSY", "RTC 动作仍在途", "rtc_reconcile")
        self.unknown_result = None

    async def _request(self, opcode: int, data: bytes = b"", *, aux: int = 0) -> bytes:
        return await self.transact(build_fixed(opcode, aux=aux, data=data))

    async def get_info(self) -> dict[str, Any]:
        d = (await self._request(1))[8:30]
        return {"protocol_major": d[0], "protocol_minor": d[1], "firmware": f"{d[2]}.{d[3]}.{_u(d, 4)}", "device_type": _u(d, 6)}

    async def get_capability(self) -> dict[str, Any]:
        d = (await self._request(2))[8:30]
        self._capability = {"max_wave_entries": _u(d, 0), "wave_entry_bytes": _u(d, 2), "clock_hz": _u(d, 4, 4),
                            "antenna_frame_bytes": _u(d, 8), "bulk_bytes": _u(d, 10), "capability_bits": _u(d, 12, 4)}
        return dict(self._capability)

    async def get_status(self) -> dict[str, Any]:
        return parse_status(await self._request(3)) | {"events": list(self.received_events),
                "unknown_result": self.unknown_result, "activity_monitor_error": self.activity_monitor_error}

    async def get_progress(self) -> dict[str, Any]:
        d = (await self._request(0x0D))[8:30]
        return {"accepted_groups": _u(d, 0, 4), "completed_groups": _u(d, 4, 4),
                "valid_triggers": _u(d, 8, 4), "completed_points": _u(d, 12, 4),
                "wave_address": _u(d, 16), "vna_trigger_index": _u(d, 18), "trigger_mode": d[20], "flags": d[21],
                "group_in_flight": bool(d[21] & 1), "fault": bool(d[21] & 2),
                "result_uncertain": bool(d[21] & 4), "io_busy": bool(d[21] & 8)}

    async def get_config(self) -> dict[str, Any]:
        d = (await self._request(0x0F))[8:30]
        return {"wave_count": _u(d, 2), "points_per_wave": _u(d, 4), "start_address": _u(d, 6),
                "trigger_mode": d[8], "antenna_interface": d[9], "tr_mode": d[10]}

    async def set_counts(self, wave_count: int, points_per_wave: int, start_address: int = 1) -> dict[str, Any]:
        _integer(wave_count, 0, 512, "wave_count")
        _integer(points_per_wave, 1, 65535, "points_per_wave")
        _integer(start_address, 1, 512, "start_address")
        if (wave_count == 0 and start_address != 1) or (wave_count and start_address + wave_count - 1 > 512):
            raise invalid("RTC 波位地址范围无效", stage="rtc_counts")
        data = b"\x00\x00" + b"".join(v.to_bytes(2, "big") for v in (wave_count, points_per_wave, start_address))
        await self._set_echo(4, data)
        return {"wave_count": wave_count, "points_per_wave": points_per_wave, "start_address": start_address}

    async def _set_echo(self, opcode: int, data: bytes) -> bytes:
        reply = await self._request(opcode, data)
        if reply[5:8] != bytes(3) or reply[8:30] != data.ljust(22, b"\x00"):
            self.unknown_result = {"opcode": opcode, "reason": "ACK_ECHO_MISMATCH"}
            raise ServiceError("DATA_INTEGRITY", "RTC 设置回显不一致", "rtc_config", side_effect_possible=True)
        return reply

    async def set_timing(self, *, pulse_qualification_us: int = 10, host_timeout_ms: int = 3000,
                         antenna_tx_timeout_ms: int = 1000, vna_ready_timeout_ms: int = 10000,
                         ack_timeout_ms: int = 1000, stable_wait_us: int = 0,
                         sync_io_timeout_ms: int = 1000) -> dict[str, int]:
        values = (pulse_qualification_us, host_timeout_ms, antenna_tx_timeout_ms,
                  vna_ready_timeout_ms, ack_timeout_ms, stable_wait_us, sync_io_timeout_ms)
        for name, value in zip(TIMING_FIELDS, values):
            _integer(value, 0 if name == "stable_wait_us" else 1, 65535, name)
        await self._set_echo(5, b"".join(v.to_bytes(2, "big") for v in values))
        self._timing = dict(zip(TIMING_FIELDS, values))
        self._activity_interval_s = host_timeout_ms / 4000
        self._activity_wakeup.set()
        return dict(self._timing)

    async def get_timing(self) -> dict[str, int]:
        d = (await self._request(0x10))[8:30]
        self._timing = {name: _u(d, 2 * index) for index, name in enumerate(TIMING_FIELDS)}
        if self._timing["host_timeout_ms"]:
            self._activity_interval_s = self._timing["host_timeout_ms"] / 4000
            self._activity_wakeup.set()
        return dict(self._timing)

    async def set_trigger_mode(self, mode: int | str) -> dict[str, int]:
        mode = {"continuous": 0, "turntable": 0, "software": 1, "step": 1}.get(str(mode).lower(), mode)
        _integer(mode, 0, 1, "trigger_mode")
        await self._set_echo(6, bytes([mode]))
        return {"trigger_mode": mode}

    async def _action(self, opcode: int) -> dict[str, Any]:
        d = (await self._request(opcode))[8:30]
        return {"state_code": d[0], "state": RTC_STATES.get(d[0], "UNKNOWN"), "stage": d[1]}

    async def arm(self): return await self._action(7)
    async def software_trigger(self): return await self._action(8)
    async def stop_graceful(self): return await self._action(9)
    async def stop_immediate(self): return await self._action(10)
    async def clear_fault(self): return await self._action(11)

    async def get_last_time(self) -> dict[str, int]:
        d = (await self._request(12))[8:30]
        return {"elapsed_us": _u(d, 0, 4), "last_result": d[4]}

    async def ping(self, data: bytes = b"") -> dict[str, Any]:
        reply = await self._request(14, data)
        return {"payload_hex": reply[8:30].hex(" ").upper()}

    async def set_tr_config(self, mode: int | str, period_us: float, high_us: float, delay_us: float = 1) -> dict[str, Any]:
        mode = {"TX": 1, "RX": 2}.get(str(mode).upper(), mode)
        _integer(mode, 1, 2, "tr_mode")
        values = (period_us, high_us, delay_us)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise invalid("RTC TR 参数必须为有限数值", stage="rtc_tr_config")
        capability = self._capability or await self.get_capability()
        clock_hz = capability["clock_hz"]
        if clock_hz <= 0:
            raise ServiceError("DEVICE_FAULT", "RTC 未提供有效实际时钟", "rtc_tr_config")
        t, h, delay = (round(v * clock_hz / 1_000_000) for v in values)
        _integer(t, 1, 0xFFFFFF, "TR周期计数")
        _integer(h, 1, 0xFFFF, "TR高宽计数")
        _integer(delay, 0, 0xFFFF, "VNA延时计数")
        pulse = math.ceil(clock_hz / 1_000_000)
        if (not 1 <= high_us <= 655 or not 0 <= delay_us <= 655
                or not clock_hz <= h * 1_000_000 <= 655 * clock_hz
                or delay * 1_000_000 > 655 * clock_hz or not 0 < h < t or
                10 * h > 3 * t or delay + pulse >= (h if mode == 1 else t - h)):
            raise invalid("RTC TR 高宽、占空比或触发相位窗口无效", stage="rtc_tr_config")
        data = bytes([mode, 0]) + t.to_bytes(3, "big") + h.to_bytes(2, "big") + delay.to_bytes(2, "big") + b"\x01"
        await self._set_echo(0x11, data)
        return await self.get_tr_config()

    async def get_tr_config(self) -> dict[str, Any]:
        d = (await self._request(0x12))[8:30]
        result = {"mode": {0: "UNCONFIGURED", 1: "TX", 2: "RX"}.get(d[0], "UNKNOWN"), "mode_code": d[0], "period_ticks": _u(d, 2, 3), "high_ticks": _u(d, 5),
                  "delay_ticks": _u(d, 7), "trigger_width_us": d[9], "tr_state": d[10], "tr_source": d[11]}
        if self._capability and self._capability["clock_hz"]:
            clock_hz = self._capability["clock_hz"]
            for name in ("period", "high", "delay"):
                result[f"{name}_us"] = result[f"{name}_ticks"] * 1_000_000 / clock_hz
            if result["mode_code"]:
                t, h, delay = (result[f"{name}_ticks"] for name in ("period", "high", "delay"))
                if (not clock_hz <= h * 1_000_000 <= 655 * clock_hz
                        or delay * 1_000_000 > 655 * clock_hz or not 0 < h < t
                        or 10 * h > 3 * t or result["trigger_width_us"] != 1
                        or delay + math.ceil(clock_hz / 1_000_000) >= (h if result["mode_code"] == 1 else t - h)):
                    raise ServiceError("DATA_INTEGRITY", "RTC TR 回读超出工程范围或相位窗口", "rtc_tr_config")
        return result

    async def get_antenna_interface(self) -> dict[str, int]:
        return {"antenna_interface": (await self._request(0x13))[8]}

    async def set_antenna_interface(self, interface: int = 0) -> dict[str, int]:
        _integer(interface, 0, 0, "antenna_interface")
        await self._set_echo(0x14, b"\x00")
        return {"antenna_interface": 0}

    async def _debug_tr(self, opcode: int) -> dict[str, Any]:
        d = (await self._request(opcode))[8:30]
        return {"tr_state": d[0], "state_code": d[1], "state": RTC_STATES.get(d[1], "UNKNOWN")}

    async def start_debug_tr(self): return await self._debug_tr(0x15)
    async def stop_debug_tr(self): return await self._debug_tr(0x16)

    async def write_wave_entry(self, address: int, antenna_frame: bytes) -> None:
        _integer(address, 1, 512, "wave_address")
        if len(antenna_frame) != 22:
            raise invalid("RTC 波位内容必须为完整 22 字节", stage="rtc_wave_write")
        reply = await self._request(0x21, antenna_frame, aux=address)
        if reply[8:30] != antenna_frame:
            self.unknown_result = {"opcode": 0x21, "address": address, "reason": "ACK_ECHO_MISMATCH"}
            raise ServiceError("DATA_INTEGRITY", "RTC 波位写入回显不一致", "rtc_wave_write", side_effect_possible=True)

    async def read_wave_entry(self, address: int) -> bytes:
        _integer(address, 1, 512, "wave_address")
        return (await self._request(0x24, aux=address))[8:30]

    async def transfer_antenna(self, antenna_frame: bytes) -> dict[str, Any]:
        if len(antenna_frame) != 22:
            raise invalid("RTC 转发内容必须为完整 22 字节", stage="rtc_antenna_transfer")
        # D5/AUX are zero. B0 proves physical TX/LOAD completion only; never
        # await E2, synthesize a response, or restart TR for this operation.
        d = (await self._request(0x30, antenna_frame))[8:30]
        if _u(d, 0) != 22 or d[4] != 0:
            self.unknown_result = {"opcode": 0x30, "reason": "TX_ACK_INVALID"}
            raise ServiceError("DATA_INTEGRITY", "RTC TX 完成应答无效", "rtc_antenna_transfer", side_effect_possible=True)
        return {"tx_state": "COMPLETED", "transmitted_length": _u(d, 0), "request_crc_fingerprint": d[2:4].hex().upper(),
                "antenna_interface": d[4], "state": RTC_STATES.get(d[5], "UNKNOWN")}

    async def get_antenna_io_status(self) -> dict[str, Any]:
        d = (await self._request(0x32))[8:30]
        return {"tx_state": d[0], "tx_error": d[1], "tx_length": _u(d, 2), "tx_request_crc": _u(d, 4),
                "rx_frames": _u(d, 6, 4), "rx_dropped_frames": _u(d, 10, 4), "debug_pulses": _u(d, 14, 4),
                "rx_sequence": _u(d, 18), "antenna_interface": d[20], "flags": d[21],
                "tx_busy": bool(d[21] & 1), "debug_tr_running": bool(d[21] & 2), "rx_overflow": bool(d[21] & 4)}

class _SimulatedSerial:
    """Offline firmware model, exercised through the real client wire decoder."""
    def __init__(self, *, point_delay_s: float = 0.001, clock_hz: int = 100_000_000, **_: Any):
        self.point_delay_s, self.clock_hz = point_delay_s, clock_hz
        self.lock = threading.RLock()
        self.incoming: queue.Queue[bytes] = queue.Queue()
        self.sent: list[bytes] = []
        self.closed = threading.Event()
        self.state, self.stage, self.fault, self.last_nack = 1, 0, 0, 0
        self.config_valid, self.rdy, self.last_result = 0, 1, 0
        self.rdy_behavior = "normal"
        self.rdy_trace: list[tuple[int, int]] = []
        self.counts = bytes(8)
        self.timing = bytes(14)
        self.tr = bytes(10)
        self.mode = 0
        self.waves: dict[int, bytes] = {}
        self.accepted_groups = self.completed_groups = self.valid_triggers = self.completed_points = 0
        self.wave_address = self.point_index = 0
        self.tr_state = self.tr_source = 0
        self.uncertain = False
        self.tx_state = self.tx_error = self.tx_length = self.tx_crc = 0
        self.rx_frames = self.rx_sequence = self.debug_pulses = 0
        self.tx_frames: list[bytes] = []
        self._queued_antenna_responses: deque[tuple[bytes, list[bytes]]] = deque()
        self._last_host = time.monotonic()
        self._group_started = 0.0
        self._elapsed = 0
        self._generation = 0
        self._stop_requested = False
        self._watchdog = threading.Thread(target=self._background, daemon=True, name="rtc-sim")
        self._watchdog.start()

    def read(self, length: int) -> bytes:
        try:
            return self.incoming.get(timeout=0.02)
        except queue.Empty:
            return b""

    def close(self):
        self.closed.set()
        with self.lock:
            self._generation += 1

    def _emit(self, opcode: int, data: bytes = b"", *, control: int = 0, aux: int = 0):
        self.incoming.put(build_fixed(opcode, control=control, aux=aux, data=data))

    def _idle(self):
        self.state = 3 if self.config_valid == 7 else 1
        self.stage = 0

    def _flags(self) -> int:
        return (int(self.tx_state == 1) | (int(self.state in {5, 6}) << 1) |
                (int(self.tr_state == 2) << 2) | (int(self.state == 9) << 3) |
                (int(self.uncertain) << 4) | (int(self.tr_state == 3) << 5) |
                (int(self.state in {5, 6}) << 6))

    def _status_data(self) -> bytes:
        return (bytes([self.state, self.stage, self.fault, self.last_nack, 0, self.tr[0]]) + bytes(2) +
                self.wave_address.to_bytes(2, "big") + self.point_index.to_bytes(2, "big") +
                self._elapsed_us().to_bytes(4, "big") + len(self.waves).to_bytes(2, "big") +
                bytes([self._flags(), self.rdy, self.last_result, self.config_valid]))

    def _progress_data(self) -> bytes:
        flags = int(self.state in {5, 6}) | (int(bool(self.fault)) << 1) | (int(self.uncertain) << 2) | (int(self.state in {5, 6}) << 3)
        return (b"".join(v.to_bytes(4, "big") for v in (self.accepted_groups, self.completed_groups, self.valid_triggers, self.completed_points)) +
                self.wave_address.to_bytes(2, "big") + self.point_index.to_bytes(2, "big") + bytes([self.mode, flags]))

    def _elapsed_us(self) -> int:
        return min(0xFFFFFFFF, int((time.monotonic() - self._group_started) * 1e6)) if self.state in {5, 6} else self._elapsed

    def _done(self, reason: int):
        self._elapsed = self._elapsed_us()
        self.state, self.stage, self.tr_state, self.tr_source = 7, 0, 0, 0
        self.last_result = reason + 1
        data = (b"".join(v.to_bytes(4, "big") for v in (self._elapsed, self.completed_groups, self.completed_points,
                                                         self.accepted_groups, self.valid_triggers)) + bytes([self._flags(), self.state]))
        self._emit(0xE0, data, control=reason, aux=self.mode << 8)

    def _fault(self, code: int, stage: int, source: int = 0):
        if self.fault:
            return
        self._elapsed = self._elapsed_us()
        self.fault, self.stage, self.state, self.last_result = code, stage, 8, 4
        self.tr_state = self.tr_source = 0
        self.uncertain = True
        self._generation += 1
        data = (b"".join(v.to_bytes(4, "big") for v in (self._elapsed, self.completed_groups, self.completed_points)) +
                self.wave_address.to_bytes(2, "big") + self.point_index.to_bytes(2, "big") +
                self.valid_triggers.to_bytes(4, "big") + bytes([self._flags(), 8]))
        self._emit(0xE1, data, control=code, aux=(stage << 8) | source)

    def _nack(self, request: bytes, error: int):
        self.last_nack = error
        self._emit(0x7F, bytes([self.state, self.stage]) + bytes(2) + request[-2:],
                   control=error, aux=request[4] << 8)

    def write(self, request: bytes) -> int:
        with self.lock:
            if self.closed.is_set():
                raise OSError("模拟串口已关闭")
            self.sent.append(request)
            parsed = parse_frame(request)
            op, data = request[4], parsed["data"]
            aux = parsed["aux"]
            self._last_host = time.monotonic()
            if op not in OPCODES.values():
                self._nack(request, 3)
                return 32
            if self.state == 8 and op not in QUERIES | {0x0B}:
                self._nack(request, 5)
                return 32
            reply = b""
            antenna_responses: list[bytes] = []
            if op == 1:
                reply = bytes([1, 0, 1, 0]) + bytes(2) + b"\x00\x01"
            elif op == 2:
                reply = b"\x02\x00\x00\x16" + self.clock_hz.to_bytes(4, "big") + b"\x00\x16\x00\x00" + (31).to_bytes(4, "big")
            elif op == 3:
                reply = self._status_data()
            elif op == 12:
                reply = self._elapsed_us().to_bytes(4, "big") + bytes([self.last_result])
            elif op == 13:
                reply = self._progress_data()
            elif op == 14:
                self._emit(0x8E, data, control=request[5], aux=aux)
                return 32
            elif op == 15:
                reply = self.counts + bytes([self.mode, 0, self.tr[0]])
            elif op == 16:
                reply = self.timing
            elif op == 18:
                reply = self.tr + bytes([self.tr_state, self.tr_source])
            elif op == 19:
                reply = b"\x00"
            elif op == 0x32:
                reply = (bytes([self.tx_state, self.tx_error]) + self.tx_length.to_bytes(2, "big") + self.tx_crc.to_bytes(2, "big") +
                         self.rx_frames.to_bytes(4, "big") + bytes(4) + self.debug_pulses.to_bytes(4, "big") +
                         self.rx_sequence.to_bytes(2, "big") + bytes([0, int(self.tx_state == 1) | (int(self.tr_source == 2) << 1)]))
            elif op in {4, 5, 6, 17, 20, 0x21, 0x24}:
                allowed = {1, 3, 7, 8} if op == 0x24 else {1, 3, 7}
                if self.state not in allowed or self.tr_state:
                    self._nack(request, 5)
                    return 32
                if op == 4:
                    self.counts, self.config_valid = data[:8], self.config_valid | 1
                elif op == 5:
                    self.timing, self.config_valid = data[:14], self.config_valid | 2
                elif op == 6:
                    self.mode = data[0]
                elif op == 17:
                    self.tr, self.config_valid = data[:10], self.config_valid | 4
                elif op in {0x21, 0x24}:
                    if not 1 <= aux <= 512:
                        self._nack(request, 7)
                        return 32
                    if op == 0x21:
                        self.waves[aux] = data
                    if aux not in self.waves:
                        self._nack(request, 8)
                        return 32
                    self._emit(op | 0x80, self.waves[aux], aux=aux)
                    return 32
                if op != 0x24:
                    self._idle()
                reply = data
            elif op == 7:
                if self.state not in {3, 4, 7} or self.config_valid != 7 or self.tr_source == 2:
                    self._nack(request, 5)
                    return 32
                b, a0 = _u(self.counts, 2), _u(self.counts, 6)
                if any(a0 + index not in self.waves for index in range(b)):
                    self._nack(request, 8)
                    return 32
                self.accepted_groups = self.completed_groups = self.valid_triggers = self.completed_points = 0
                self.wave_address = self.point_index = 0
                self.last_result, self.uncertain, self._stop_requested = 0, False, False
                self.state, self.stage, self.tr_state, self.tr_source = 4, 0, 2, 1
                reply = bytes([self.state, self.stage])
            elif op == 8:
                if self.state != 4 or self.mode != 1:
                    self._nack(request, 5)
                    return 32
                self._start_group()
                reply = bytes([5, self.stage])
            elif op in {9, 10}:
                if self.state not in {4, 5, 6}:
                    self._nack(request, 5)
                    return 32
                was_busy = self.state in {5, 6}
                self._stop_requested = True
                reply = bytes([6, 10])
                if op == 10:
                    self._generation += 1
                    self.uncertain = self.uncertain or was_busy
                    self._done(2)
                elif not was_busy:
                    self._done(1)
                else:
                    self.state = 6
            elif op == 11:
                if self.state != 8 or self.rdy != 1:
                    self._nack(request, 5)
                    return 32
                self.fault = self.config_valid = self.last_result = 0
                self.uncertain = False
                self._idle()
                reply = bytes([self.state, self.stage])
            elif op == 21:
                if self.state not in {1, 3, 7} or self.tr_state or self.config_valid & 6 != 6:
                    self._nack(request, 6)
                    return 32
                self.state, self.stage, self.tr_state, self.tr_source = 9, 11, 2, 2
                self.debug_pulses = 0
                reply = bytes([self.tr_state, self.state])
            elif op == 22:
                if self.tr_source == 1:
                    self._nack(request, 5)
                    return 32
                self.tr_state = self.tr_source = 0
                self._idle()
                reply = bytes([self.tr_state, self.state])
            elif op == 0x30:
                if self.state not in {1, 3, 7, 9} or self.config_valid & 2 == 0:
                    self._nack(request, 5)
                    return 32
                # Default transfer has no synthetic RX. A workflow/test may
                # explicitly queue profile-valid antenna responses for this TX.
                self.tx_frames.append(data)
                if self._queued_antenna_responses and self._queued_antenna_responses[0][0] == data:
                    _, antenna_responses = self._queued_antenna_responses.popleft()
                self.tx_state, self.tx_length, self.tx_crc = 2, 22, _u(request, 30)
                if self.tr_source != 2:
                    self._idle()
                reply = b"\x00\x16" + request[-2:] + bytes([0, self.state])
            self._emit(op | 0x80, reply)
            for antenna_response in antenna_responses:
                self.inject_antenna_rx(antenna_response)
            return 32

    def queue_antenna_response(self, request: bytes, responses: list[bytes]) -> None:
        if len(request) != 22 or not responses or any(len(frame) != 22 for frame in responses):
            raise ValueError("模拟波控应答需指定22字节请求和至少一个22字节回包")
        with self.lock:
            self._queued_antenna_responses.append((bytes(request), [bytes(frame) for frame in responses]))

    def _start_group(self):
        if self.accepted_groups == 0xFFFFFFFF:
            self._fault(0x16, 0)
            return
        self.accepted_groups += 1
        self.state, self.stage = 5, 4
        self._group_started = time.monotonic()
        self._generation += 1
        threading.Thread(target=self._run_group, args=(self._generation,), daemon=True, name="rtc-sim-group").start()

    def external_trigger(self) -> bool:
        with self.lock:
            if self.state in {5, 6} and not self._stop_requested:
                self._fault(0x13, self.stage)
                return False
            if self.state != 4 or self.mode != 0 or self._stop_requested:
                return False
            self._start_group()
            return True

    def _still_running(self, generation: int) -> bool:
        return generation == self._generation and not self.closed.is_set() and self.state in {5, 6}

    def _phase(self, generation: int, stage: int, level: int, fail: bool) -> bool:
        with self.lock:
            if not self._still_running(generation):
                return False
            self.stage = stage
            timeout_s = _u(self.timing, 6) / 1000
        deadline = time.monotonic() + timeout_s
        if fail:
            while time.monotonic() < deadline and not self.closed.wait(0.001):
                with self.lock:
                    if not self._still_running(generation):
                        return False
            with self.lock:
                if self._still_running(generation):
                    self._fault(0x0B, stage)
            return False
        if self.closed.wait(self.point_delay_s):
            return False
        with self.lock:
            if not self._still_running(generation):
                return False
            self.rdy = level
            self.rdy_trace.append((stage, level))
            if stage == 7:
                if self.valid_triggers == 0xFFFFFFFF:
                    self._fault(0x16, stage)
                    return False
                self.valid_triggers += 1
            if stage == 8:
                if self.completed_points == 0xFFFFFFFF:
                    self._fault(0x16, stage)
                    return False
                self.completed_points += 1
            return True

    def _run_group(self, generation: int):
        b, v, a0 = _u(self.counts, 2), _u(self.counts, 4), _u(self.counts, 6)
        for index in range(max(b, 1)):
            with self.lock:
                if not self._still_running(generation):
                    return
                self.wave_address = a0 + index if b else 0
                if b:
                    self.tx_frames.append(self.waves[self.wave_address])
            for point in range(1, v + 1):
                with self.lock:
                    self.point_index = point
                if not self._phase(generation, 4, 1, self.rdy_behavior == "low_at_start"):
                    return
                if not self._phase(generation, 7, 0, self.rdy_behavior == "no_low"):
                    return
                if not self._phase(generation, 8, 1, self.rdy_behavior == "no_high"):
                    return
        with self.lock:
            if not self._still_running(generation):
                return
            if self.completed_groups == 0xFFFFFFFF:
                self._fault(0x16, 9)
                return
            self.completed_groups += 1
            self._elapsed = self._elapsed_us()
            if self._stop_requested:
                self._done(1)
            elif self.mode == 1:
                self._done(0)
            else:
                self.state, self.stage = 4, 0  # Continuous groups are deliberately silent.

    def _background(self):
        while not self.closed.wait(0.005):
            with self.lock:
                active = self.state in {4, 5, 6, 9}
                timeout_ms = _u(self.timing, 2)
                if active and timeout_ms and (time.monotonic() - self._last_host) * 1000 >= timeout_ms:
                    self._fault(0x14, self.stage, 1 if self.tr_source == 2 else 0)
                if self.tr_source == 2 and self.tr_state == 2 and self.rdy == 1:
                    # One sampled debug cycle; RDY low skips with no queued retry.
                    self.debug_pulses = min(0xFFFFFFFF, self.debug_pulses + 1)

    def inject_antenna_rx(self, payload: bytes):
        if len(payload) != 22:
            raise ValueError("模拟 RX 必须为完整 22 字节")
        with self.lock:
            self.rx_frames = min(0xFFFFFFFF, self.rx_frames + 1)
            self.rx_sequence = (self.rx_sequence + 1) & 0xFFFF
            self._emit(0xE2, payload, control=22, aux=self.rx_sequence)


class SimulatedRtcClient(RtcClient):
    def __init__(self, endpoint: RtcEndpoint | None = None, *, on_event=None, point_delay_s: float = 0.001,
                 clock_hz: int = 100_000_000):
        self.firmware: _SimulatedSerial | None = None
        def factory(**kwargs):
            self.firmware = _SimulatedSerial(point_delay_s=point_delay_s, clock_hz=clock_hz, **kwargs)
            return self.firmware
        super().__init__(endpoint or RtcEndpoint("SIMULATED"), on_event=on_event, serial_factory=factory)

    async def external_trigger(self) -> bool:
        if self.firmware is None:
            raise ServiceError("NOT_RUNNABLE", "RTC 模拟器尚未连接", "rtc_simulated")
        return self.firmware.external_trigger()

    def inject_antenna_rx(self, payload: bytes) -> None:
        if self.firmware is None:
            raise ServiceError("NOT_RUNNABLE", "RTC 模拟器尚未连接", "rtc_simulated")
        self.firmware.inject_antenna_rx(payload)

    def queue_antenna_response(self, request: bytes, responses: list[bytes]) -> None:
        """Explicit simulated antenna fixture, emitted through E2 after matching TX."""
        if self.firmware is None:
            raise ServiceError("NOT_RUNNABLE", "RTC 模拟器尚未连接", "rtc_simulated")
        self.firmware.queue_antenna_response(request, responses)
