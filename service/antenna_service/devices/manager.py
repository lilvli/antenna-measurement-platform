from __future__ import annotations

import asyncio
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from antenna_service.devices.base import DeviceAdapter
from antenna_service.devices.real import RealBeamController, RealRtc, RealTurntable, RealVna
from antenna_service.devices.simulated import SimulatedBeamController, SimulatedRtc, SimulatedTurntable, SimulatedVna
from antenna_service.errors import ServiceError, invalid
from antenna_service.events import EventBus
from antenna_service.models import DeviceSource, DeviceState


class DeviceManager:
    DEVICE_IDS = {"beam_controller", "vna", "turntable", "rtc"}

    def __init__(
        self,
        events: EventBus,
        frame_decoder: Callable[[bytes], dict[str, Any] | None] | None = None,
    ) -> None:
        self.events = events
        self.frame_decoder = frame_decoder
        self.devices: dict[str, DeviceAdapter] = {}
        self._control_lock = asyncio.Lock()
        self._control_owner: str | None = None
        self.shutdown_requested = False
        self.device_revisions = {device_id: 0 for device_id in self.DEVICE_IDS}

    def note_configuration_change(self, device_id: str) -> None:
        self.device_revisions[device_id] = self.device_revisions.get(device_id, 0) + 1

    @property
    def control_owner(self) -> str | None:
        return self._control_owner

    async def _acquire_control(self, owner: str, stage: str) -> None:
        if self.shutdown_requested and stage != "shutdown":
            raise ServiceError("CONTROL_LOCKED", "软件正在停止任务并退出，暂不接受新设备操作", stage)
        if self._control_lock.locked():
            raise ServiceError(
                "CONTROL_LOCKED",
                "自动测试或其他设备操作正在占用测控资源",
                stage,
                self._control_owner,
            )
        await self._control_lock.acquire()
        self._control_owner = owner

    def _release_control(self, owner: str) -> None:
        if self._control_owner != owner:
            return
        self._control_owner = None
        self._control_lock.release()

    @asynccontextmanager
    async def manual_control(self, stage: str):
        owner = f"MANUAL:{stage}"
        await self._acquire_control(owner, stage)
        try:
            yield
        finally:
            self._release_control(owner)

    async def acquire_run_control(self, run_id: str) -> None:
        await self._acquire_control(f"RUN:{run_id}", "run_start")

    def release_run_control(self, run_id: str) -> None:
        self._release_control(f"RUN:{run_id}")

    async def connect(self, device_id: str, source: DeviceSource, parameters: dict[str, Any]) -> dict[str, Any]:
        async with self.manual_control("device_connect"):
            return await self._connect_unlocked(device_id, source, parameters)

    async def _connect_unlocked(self, device_id: str, source: DeviceSource, parameters: dict[str, Any]) -> dict[str, Any]:
        # 每台设备都单独创建适配器，source 不会影响其他设备。因此可以同时使用
        # “模拟波控机 + 真实 VNA + 真实转台”；更换同一设备来源前先关闭旧句柄，
        # 避免 COM 口、VISA 会话或 PMAC 连接被两个适配器同时占用。
        if device_id not in self.DEVICE_IDS:
            raise invalid("未知设备", stage="device_connect", target=device_id)
        self.note_configuration_change(device_id)
        if device_id in self.devices:
            await self.devices[device_id].disconnect()
        adapter = self._make_adapter(device_id, source, parameters)
        self.devices[device_id] = adapter
        try:
            snapshot = await adapter.connect()
        except Exception:
            await self.events.publish("device.status", device=(await adapter.status()))
            raise
        await self.events.publish("device.status", device=snapshot)
        return snapshot

    def _make_adapter(self, device_id: str, source: DeviceSource, parameters: dict[str, Any]) -> DeviceAdapter:
        if source == DeviceSource.SIMULATED:
            if device_id == "rtc":
                return SimulatedRtc(events=self.events, frame_decoder=self.frame_decoder)
            return {
                "beam_controller": SimulatedBeamController,
                "vna": SimulatedVna,
                "turntable": SimulatedTurntable,
                "rtc": SimulatedRtc,
            }[device_id]()
        if device_id == "beam_controller":
            port = str(parameters.get("port", "")).strip()
            if not port:
                raise invalid("真实波控机需要 COM 口", stage="device_connect", target=device_id)
            return RealBeamController(
                port,
                int(parameters.get("baud_rate", 115200)),
                events=self.events,
                frame_decoder=self.frame_decoder,
            )
        if device_id == "vna":
            resource = str(parameters.get("resource", "")).strip()
            if not resource:
                raise invalid("真实 VNA 需要 VISA 资源", stage="device_connect", target=device_id)
            # resource 原样交给系统 VISA 库，例如 TCPIP0::...::hislip0::INSTR；
            # backend 为空时使用 Windows 已注册的默认 VISA 实现，不在软件内伪造连接。
            return RealVna(resource, parameters.get("backend"), int(parameters.get("timeout_ms", 10000)))
        if device_id == "rtc":
            port = str(parameters.get("port", "")).strip()
            if not port:
                raise invalid("真实RTC需要COM口", stage="device_connect", target=device_id)
            return RealRtc(port, int(parameters.get("baud_rate", 115200)),
                           events=self.events, frame_decoder=self.frame_decoder)
        if device_id == "turntable":
            dll_path = str(parameters.get("dll_path", "")).strip()
            if not dll_path:
                # The turntable driver is an application resource. Users should never need to
                # browse to a DLL; packaged Electron passes its internal absolute resource path.
                dll_path = str(
                    Path(__file__).resolve().parents[3]
                    / "desktop"
                    / "resources"
                    / "turntable"
                    / "ImacFxDll.dll"
                )
            return RealTurntable(
                dll_path,
                device_number=int(parameters.get("device_number", 0)),
                controller_ip=str(parameters.get("controller_ip", "192.168.1.101")),
            )
        raise AssertionError("unreachable")

    async def disconnect(self, device_id: str) -> dict[str, Any]:
        async with self.manual_control("device_disconnect"):
            return await self._disconnect_unlocked(device_id)

    async def _disconnect_unlocked(self, device_id: str) -> dict[str, Any]:
        adapter = self.require(device_id, ready=False)
        self.note_configuration_change(device_id)
        await adapter.disconnect()
        snapshot = await adapter.status()
        await self.events.publish("device.status", device=snapshot)
        return snapshot

    def require(self, device_id: str, *, ready: bool = True) -> DeviceAdapter:
        adapter = self.devices.get(device_id)
        if adapter is None:
            raise ServiceError("NOT_RUNNABLE", f"设备 {device_id} 尚未连接", "devices", device_id)
        if ready and adapter.snapshot.state not in {DeviceState.READY, DeviceState.CONNECTED}:
            raise ServiceError(
                "NOT_RUNNABLE", f"设备 {device_id} 未就绪", "devices", device_id, adapter.snapshot.as_dict()
            )
        return adapter

    async def list_status(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for device_id in sorted(self.DEVICE_IDS):
            adapter = self.devices.get(device_id)
            if adapter:
                result.append(await adapter.status())
            else:
                result.append(
                    {
                        "device_id": device_id,
                        "source": None,
                        "state": DeviceState.DISCONNECTED.value,
                        "identity": None,
                        "details": {},
                    }
                )
        return result

    async def command(self, device_id: str, action: str, parameters: dict[str, Any]) -> Any:
        # A real turntable move/home keeps the global control lease until the axis is
        # confirmed in position.  Telemetry and the vendor software Stop are special:
        # their individual DLL calls are protected by RealTurntable._io_lock, so they
        # must remain available while a *manual* motion is waiting.  During an automatic
        # run only read-only telemetry is admitted; all manual side effects stay blocked.
        owner = self._control_owner
        rtc_queries = {"get_status", "get_progress", "get_tr_config", "get_antenna_io_status"}
        if device_id == "rtc" and action in rtc_queries:
            return await self._command_unlocked(device_id, action, parameters)
        if device_id == "turntable" and self._control_lock.locked():
            manual_motion = owner == "MANUAL:turntable_motion"
            run_active = bool(owner and owner.startswith("RUN:"))
            if (action == "read_axes" and (manual_motion or run_active)) or (action == "stop" and manual_motion):
                return await self._command_unlocked(device_id, action, parameters)

        stage = (
            "turntable_motion"
            if device_id == "turntable" and action in {"move_to", "home"}
            else f"turntable_{action}"
            if device_id == "turntable" and action in {"read_axes", "stop"}
            else "device_command"
        )
        async with self.manual_control(stage):
            if action not in {"read_axes", "get_status", "get_progress", "get_tr_config", "get_antenna_io_status"}:
                self.note_configuration_change(device_id)
            return await self._command_unlocked(device_id, action, parameters)

    async def _command_unlocked(self, device_id: str, action: str, parameters: dict[str, Any]) -> Any:
        adapter = self.require(device_id, ready=not (
            (device_id == "turntable" and action in {"read_axes", "stop"})
            or device_id == "rtc"
        ))
        if device_id == "turntable":
            # 结构化命令是界面与厂商 DLL 之间唯一的转台入口。所有数值先在此处
            # 归一化，再由 RealTurntable 做轴号、位置范围和速度上限的第二次校验。
            if action == "read_axes":
                return await getattr(adapter, "read_all_axes")() if hasattr(adapter, "read_all_axes") else await adapter.status()
            if action == "move_to":
                try:
                    speed = float(parameters["speed"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise invalid("转台移动速度必须是数字", stage="turntable_move", target=device_id) from exc
                if not math.isfinite(speed) or speed <= 0:
                    raise invalid("转台移动速度必须大于 0，不能为负数", stage="turntable_move", target=device_id)
                speed = round(speed, 4)
                if speed <= 0:
                    raise invalid("转台移动速度最小为 0.0001", stage="turntable_move", target=device_id)
                return await getattr(adapter, "move_to")(
                    int(parameters["axis"]), float(parameters["target"]), speed
                )
            if action == "home":
                return await getattr(adapter, "home")(int(parameters["axis"]))
            if action == "stop":
                return await getattr(adapter, "stop")(parameters.get("axis", "all"))
        if device_id == "rtc":
            if action == "configure_tr":
                try:
                    mode = str(parameters["mode"]).upper()
                    period_us = float(parameters["period_us"])
                    high_us = float(parameters["high_us"])
                    delay_us = float(parameters.get("delay_us", 1))
                except (KeyError, ValueError, TypeError) as exc:
                    raise invalid("TR参数不完整或不是有效数字", stage="rtc_configure_tr") from exc
                if mode not in {"TX", "RX"} or not all(math.isfinite(v) for v in (period_us, high_us, delay_us)):
                    raise invalid("TR模式必须为TX/RX且时间为有限数", stage="rtc_configure_tr")
                # Validate and set TR first; invalid values must not replace timing.
                # Initialize missing I/O deadlines only, without enabling output.
                await getattr(adapter, "set_tr_config")(mode, period_us, high_us, delay_us)
                timing = await getattr(adapter, "get_timing")()
                if not all(timing.get(key) for key in ("host_timeout_ms", "antenna_tx_timeout_ms",
                                                      "vna_ready_timeout_ms", "sync_io_timeout_ms")):
                    await getattr(adapter, "set_timing")()
                return await getattr(adapter, "get_tr_config")()
            if action in {"get_status", "get_progress", "get_tr_config", "get_antenna_io_status"}:
                return await getattr(adapter, action)()
            if action in {"start_debug_tr", "stop_debug_tr"}:
                await getattr(adapter, action)()
                timing = await getattr(adapter, "get_timing")()
                deadline = asyncio.get_running_loop().time() + max(1, timing["sync_io_timeout_ms"] / 1000) + 2
                expected = 2 if action == "start_debug_tr" else 0
                while True:
                    tr = await getattr(adapter, "get_tr_config")()
                    status = await getattr(adapter, "get_status")()
                    if status.get("fault_code") or status.get("state") == "FAULT":
                        raise ServiceError("DEVICE_FAULT", "RTC调试TR发生故障", "rtc_debug_tr", details=status,
                                           side_effect_possible=True)
                    if tr["tr_state"] == expected and not status.get("tr_stopping"):
                        return tr
                    if asyncio.get_running_loop().time() >= deadline:
                        raise ServiceError("TIMEOUT", "RTC调试TR实际启停状态未确认，请查询状态", "rtc_debug_tr",
                                           details=tr, side_effect_possible=True)
                    await asyncio.sleep(0.02)
            if action == "clear_fault":
                turntable = self.devices.get("turntable")
                if turntable is not None and turntable.snapshot.state != DeviceState.DISCONNECTED:
                    axes = await getattr(turntable, "read_all_axes")()
                    velocities = axes.get("velocities", {})
                    if not velocities or any(not math.isfinite(float(value)) or abs(float(value)) > 0.000001
                                             for value in velocities.values()):
                        raise ServiceError("NOT_RUNNABLE", "转台尚未确认静止，不能清除RTC故障", "rtc_clear_fault")
                status = await getattr(adapter, "get_status")()
                if (status.get("state") != "FAULT" or status.get("rdy") != 1
                        or any(status.get(key) for key in ("group_in_flight", "io_busy", "tx_busy", "tr_running"))):
                    raise ServiceError("NOT_RUNNABLE", "RTC故障恢复条件未满足，请确认RDY高且所有输出排空", "rtc_clear_fault")
                if adapter.client.unknown_result:
                    # Explicit user recovery, after observed FAULT and drain;
                    # never used to retry the original measurement command.
                    adapter.client.acknowledge_unknown_result()
                return await getattr(adapter, "clear_fault")()
        raise invalid("该设备不支持此结构化动作", stage="device_command", target=f"{device_id}.{action}")
