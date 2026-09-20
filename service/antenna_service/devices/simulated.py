from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Callable

import numpy as np

from antenna_service.devices.base import DeviceAdapter, RtcDeviceAdapter, azimuth_scan_geometry
from antenna_service.errors import invalid, ServiceError
from antenna_service.events import EventBus
from antenna_service.protocol.rtc import SimulatedRtcClient
from antenna_service.models import DeviceSource, DeviceState


class SimulatedBeamController(DeviceAdapter):
    def __init__(self) -> None:
        super().__init__("beam_controller", DeviceSource.SIMULATED)
        self.flash_memory: dict[tuple[int, int], bytes] = {}
        self.last_frame: bytes | None = None

    async def connect(self) -> dict[str, Any]:
        self.snapshot.identity = "SIM-BEAM-1"
        self.snapshot.update(state=DeviceState.READY, initialized=True)
        return self.snapshot.as_dict()

    async def disconnect(self) -> None:
        self.snapshot.update(state=DeviceState.DISCONNECTED)

    async def send_frame(
        self,
        frame: bytes,
        *,
        timeout_ms: int = 1000,
        response_opcode: int | None = None,
    ) -> bytes:
        await asyncio.sleep(min(timeout_ms / 1000, 0.005))
        self.last_frame = bytes(frame)
        return bytes(frame)

    async def send_only(self, frame: bytes) -> None:
        """Mirror the real manual-debug contract: write succeeds without synthesizing RX."""
        await asyncio.sleep(0.001)
        self.last_frame = bytes(frame)

    async def flash_write(self, tile_id: int, address: int, payload: bytes) -> None:
        await asyncio.sleep(0.001)
        self.flash_memory[(tile_id, address)] = bytes(payload)

    async def flash_read(self, tile_id: int, address: int, length: int) -> bytes:
        await asyncio.sleep(0.001)
        return self.flash_memory.get((tile_id, address), b"")[:length]


class SimulatedVna(DeviceAdapter):
    def __init__(self) -> None:
        super().__init__("vna", DeviceSource.SIMULATED)
        self.settings: dict[str, Any] = {}
        self.sample_counter = 0
        self._frequencies: np.ndarray | None = None
        self._buffer_plan: dict[str, Any] | None = None

    async def connect(self) -> dict[str, Any]:
        self.snapshot.identity = "SIMULATED,ANTENNA-VNA,0001,1.0"
        self.snapshot.update(state=DeviceState.READY)
        return self.snapshot.as_dict()

    async def disconnect(self) -> None:
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
        self._frequencies = np.asarray(frequencies_hz, dtype=float).copy()
        self._buffer_plan = None
        self.settings = {
            "s_parameter": s_parameter,
            "frequency_start_hz": float(frequencies_hz[0]),
            "frequency_stop_hz": float(frequencies_hz[-1]),
            "points": int(frequencies_hz.size),
            "if_bandwidth_hz": if_bandwidth_hz,
            "source_power_dbm": float(source_power_dbm),
            "source_port": int(s_parameter[2]),
            "averaging_enabled": bool(averaging_enabled),
            "averaging_count": int(averaging_count),
            "averaging_mode": "POINT" if trigger_mode == "EXTERNAL_POINT" else "SWEEP",
            "trigger_mode": trigger_mode,
        }
        self.snapshot.update(settings=self.settings)
        return dict(self.settings)

    async def prepare_buffered_acquisition(self, frequencies_hz: np.ndarray, sweep_count: int) -> dict[str, Any]:
        if (self.settings.get("trigger_mode") != "EXTERNAL_POINT" or self._frequencies is None
                or np.asarray(frequencies_hz).shape != self._frequencies.shape
                or not np.allclose(frequencies_hz, self._frequencies, rtol=1e-10, atol=1.0)):
            raise invalid("请先配置 RTC 外部逐点频率", stage="vna_buffer_prepare")
        if isinstance(sweep_count, bool) or not isinstance(sweep_count, int) or sweep_count < 1:
            raise invalid("缓冲扫频次数必须为正整数", stage="vna_buffer_prepare")
        count = int(self._frequencies.size)
        self._buffer_plan = {"sweep_count": sweep_count, "points_per_sweep": count, "triggers_per_sweep": count,
                             "memory_bytes": sweep_count * count * 8, "measurement_number": 1,
                             "buffer_kind": "REPEATED_SWEEP", "averaging_mode": "POINT", "ready_polarity": "HIGH"}
        self.snapshot.update(buffer=dict(self._buffer_plan))
        return dict(self._buffer_plan)

    async def read_buffered_acquisition(self, frequencies_hz: np.ndarray, sweep_count: int, *,
                                       completed_trigger_count: int,
                                       sample_contexts: list[dict[str, Any]] | None = None) -> np.ndarray:
        plan = self._buffer_plan
        if plan is None or plan["sweep_count"] != sweep_count:
            raise invalid("VNA 缓冲未准备或行规模改变", stage="vna_buffer_read")
        if (np.asarray(frequencies_hz).shape != self._frequencies.shape
                or not np.allclose(frequencies_hz, self._frequencies, rtol=1e-10, atol=1.0)
                or completed_trigger_count != sweep_count * plan["triggers_per_sweep"]):
            raise ServiceError("DATA_INTEGRITY", "RTC 完成点数或频率与缓冲计划不一致", "vna_buffer_read", "vna")
        contexts = sample_contexts if sample_contexts is not None else [{} for _ in range(sweep_count)]
        if len(contexts) != sweep_count:
            raise invalid("样本位置数量与缓冲记录数不一致", stage="vna_buffer_read")
        self._buffer_plan = None
        return np.stack([await self.acquire(frequencies_hz, **context) for context in contexts])

    async def abort_buffered_acquisition(self) -> None:
        self._buffer_plan = None

    async def acquire(
        self,
        frequencies_hz: np.ndarray,
        *,
        channel_element: int | None = None,
        azimuth_deg: float = 0,
        elevation_deg: float = 0,
    ) -> np.ndarray:
        await asyncio.sleep(0.002)
        self.sample_counter += 1
        if channel_element is not None:
            # Stable channel-to-channel differences exercise magnitude, phase and compensation.
            amplitude = 0.8 + 0.18 * math.cos(channel_element * 0.17)
            base_phase = channel_element * 0.31
        else:
            radius = math.hypot(azimuth_deg, elevation_deg)
            amplitude = max(0.001, abs(math.cos(math.radians(radius * 4.0))) ** 3)
            base_phase = math.radians(azimuth_deg * 2 + elevation_deg)
        normalized = (frequencies_hz - frequencies_hz[0]) / max(float(np.ptp(frequencies_hz)), 1.0)
        return amplitude * np.exp(1j * (base_phase + normalized * 0.2))


class SimulatedTurntable(DeviceAdapter):
    AXES = {1: "azimuth", 2: "elevation", 3: "polarization", 4: "feed", 7: "translation"}

    def __init__(self) -> None:
        super().__init__("turntable", DeviceSource.SIMULATED)
        self.positions = {axis: 0.0 for axis in self.AXES}
        self.velocities = {axis: 0.0 for axis in self.AXES}
        self._scan_cancelled = asyncio.Event()
        self._scan_motion: tuple[float, float, float, float] | None = None

    async def connect(self) -> dict[str, Any]:
        self.snapshot.identity = "SIM-PMAC-0"
        self.snapshot.update(state=DeviceState.READY, positions=self.positions, velocities=self.velocities)
        return self.snapshot.as_dict()

    async def disconnect(self) -> None:
        self.snapshot.update(state=DeviceState.DISCONNECTED)

    async def move_to(self, axis: int, target: float, speed: float) -> dict[str, Any]:
        if axis not in self.AXES:
            raise invalid("轴号必须为 1/2/3/4/7", stage="turntable_move", target=str(axis))
        if not math.isfinite(speed) or speed <= 0:
            raise invalid("转台移动速度必须大于 0，不能为负数", stage="turntable_move", target=str(axis), speed=speed)
        speed = round(float(speed), 4)
        if speed <= 0:
            raise invalid("转台移动速度最小为 0.0001", stage="turntable_move", target=str(axis))
        self.velocities[axis] = speed
        await asyncio.sleep(0.003)
        self.positions[axis] = target
        self.velocities[axis] = 0.0
        self.snapshot.update(positions=dict(self.positions), velocities=dict(self.velocities))
        return {"axis": axis, "position": target, "velocity": 0.0}

    def _update_scan_position(self) -> None:
        if self._scan_motion is not None:
            start, end, started_at, duration = self._scan_motion
            fraction = min(1.0, (time.perf_counter() - started_at) / duration)
            self.positions[1] = start + (end - start) * fraction

    async def scan_azimuth(self, start_deg: float, end_deg: float, step_deg: float, speed: float, *,
                           timeout_seconds: float = 120, position_tolerance: float = 0.01) -> dict[str, Any]:
        intervals, _ = azimuth_scan_geometry(start_deg, end_deg, step_deg, speed)
        if abs(self.positions[1] - start_deg) > position_tolerance or abs(self.velocities[1]) > 0.000001:
            raise invalid("连续扫描前方位轴须已在起点并静止", stage="turntable_scan")
        self._scan_cancelled.clear()
        duration = abs(end_deg - start_deg) / speed
        started_at = time.perf_counter()
        self._scan_motion = (start_deg, end_deg, started_at, duration)
        self.velocities[1] = math.copysign(speed, end_deg - start_deg)
        self.snapshot.update(pulse_output_disabled=False)
        try:
            while True:
                if self._scan_cancelled.is_set():
                    raise ServiceError("NOT_RUNNABLE", "连续扫描已被软件停止", "turntable_scan_wait", "1", side_effect_possible=True)
                elapsed = time.perf_counter() - started_at
                if timeout_seconds < duration and elapsed >= timeout_seconds:
                    raise ServiceError("TIMEOUT", "连续扫描未确认到达行末并静止", "turntable_scan_wait", "1", side_effect_possible=True)
                if elapsed >= duration:
                    break
                try:
                    # Windows event-loop timers may wake before their requested
                    # deadline. Only high-resolution elapsed travel time completes
                    # the row; an early timeout just recomputes the remaining wait.
                    await asyncio.wait_for(self._scan_cancelled.wait(), timeout=min(duration, timeout_seconds) - elapsed)
                except TimeoutError:
                    pass
            self.positions[1] = end_deg
        finally:
            self._update_scan_position()
            self._scan_motion = None
            self.velocities[1] = 0.0
            self.snapshot.update(pulse_output_disabled=True, positions=dict(self.positions), velocities=dict(self.velocities))
        telemetry = await self.read_all_axes()
        return {"axis": 1, "position": self.positions[1], "velocity": 0.0, "interval_count": intervals,
                "planned_point_count": intervals + 1, "endpoint_pulse_count_verified": False,
                "pulse_output_disabled": True, "position_source": "ROW_END_READBACK", **telemetry}

    async def read_all_axes(self) -> dict[str, Any]:
        """Return the same five-axis readback shape as the real PMAC adapter."""
        self._update_scan_position()
        return {
            "positions": {str(axis): value for axis, value in self.positions.items()},
            "velocities": {str(axis): value for axis, value in self.velocities.items()},
        }

    async def home(self, axis: int) -> dict[str, Any]:
        return await self.move_to(axis, 0.0, 1.0)

    async def stop(self, axis: int | str = "all") -> dict[str, Any]:
        axes = self.AXES if axis == "all" else [int(axis)]
        if 1 in axes:
            self._update_scan_position()
            self._scan_motion = None
            self._scan_cancelled.set()
        for item in axes:
            self.velocities[item] = 0.0
        self.snapshot.update(velocities=dict(self.velocities))
        return {"stopped": list(axes), "velocities": dict(self.velocities)}


class SimulatedRtc(RtcDeviceAdapter):
    def __init__(self, *, events: EventBus | None = None,
                 frame_decoder: Callable[[bytes], dict[str, Any] | None] | None = None) -> None:
        super().__init__(DeviceSource.SIMULATED, events=events, frame_decoder=frame_decoder)
        self.client = SimulatedRtcClient(on_event=self._on_client_event)

    def queue_antenna_response(self, request: bytes, responses: list[bytes]) -> None:
        """Explicit offline fixture only; real RTC never synthesizes antenna RX."""
        self.client.queue_antenna_response(request, responses)
