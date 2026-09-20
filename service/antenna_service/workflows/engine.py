from __future__ import annotations

import asyncio
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from antenna_service.coordinates import Channel, validate_profile_coordinate_pair
from antenna_service.devices.manager import DeviceManager
from antenna_service.errors import ServiceError
from antenna_service.events import EventBus
from antenna_service.models import DeviceState, RunPlan, RunState
from antenna_service.protocol.profile import AssetRegistry, Command, ProtocolProfile
from antenna_service.storage.hdf5_store import MeasurementStore, timestamped_path
from antenna_service.workflows.rtc import (
    RtcAcquisition, freeze_rtc_configuration, verify_frozen_rtc_configuration,
    compile_rtc_wave_table, wave_request_from_plan, read_rtc_wave_table, verify_rtc_wave_table,
)


@dataclass(slots=True)
class RunRecord:
    run_id: str
    plan: RunPlan
    state: RunState
    created_at: str
    total: int
    completed: int = 0
    output_path: str | None = None
    error: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    pause_gate: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    stop_requested: bool = False
    task: asyncio.Task[None] | None = field(default=None, repr=False)
    store: MeasurementStore | None = field(default=None, repr=False)
    samples: list[dict[str, Any]] = field(default_factory=list, repr=False)
    cleanup_pending: bool = False
    pause_requested: bool = False
    device_revisions: dict[str, int] = field(default_factory=dict)
    rtc_configuration: dict[str, Any] = field(default_factory=dict)
    rtc_session: RtcAcquisition | None = field(default=None, repr=False)
    rtc_wave_entries: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "state": self.state.value,
            "created_at": self.created_at,
            "total": self.total,
            "completed": self.completed,
            "progress": self.completed / self.total if self.total else 0,
            "output_path": self.output_path,
            "error": self.error,
            "result": self.result,
            "plan": self.plan.model_dump(),
            "cleanup_pending": self.cleanup_pending,
            "pause_requested": self.pause_requested,
            "rtc_configuration": self.rtc_configuration,
        }


def inclusive_axis(start: float, stop: float, step: float) -> np.ndarray:
    count = int(math.floor((stop - start) / step + 1e-9)) + 1
    values = start + np.arange(count, dtype=float) * step
    if values[-1] < stop - 1e-9:
        values = np.append(values, stop)
    return values


class RunEngine:
    def __init__(self, assets: AssetRegistry, devices: DeviceManager, events: EventBus) -> None:
        self.assets = assets
        self.devices = devices
        self.events = events
        self.runs: dict[str, RunRecord] = {}
        self._active_lock = asyncio.Lock()

    async def prepare(self, plan: RunPlan) -> RunRecord:
        async with self.devices.manual_control("run_prepare"):
            return await self._prepare_unlocked(plan)

    async def _prepare_unlocked(self, plan: RunPlan) -> RunRecord:
        profile = self.assets.profile(plan.profile_id)
        coordinates = self.assets.coordinate(plan.coordinate_id)
        validate_profile_coordinate_pair(profile, coordinates)
        if plan.polarization not in coordinates.polarizations:
            raise ServiceError("NOT_RUNNABLE", "坐标表没有所选极化", "run_prepare", plan.polarization)
        required_capability = "calibration" if plan.test_type == "CALIBRATION" else "pattern"
        if not profile.capabilities.get(required_capability):
            raise ServiceError("NOT_RUNNABLE", f"配置包不支持 {required_capability}", "run_prepare", profile.profile_id)

        # EXTERNAL_FIXED is acquisition-only: the beam may have been set from this
        # app's debug page or another controller, so its connection state must not
        # participate in PREPARE and the automatic run must not touch it.
        uses_rtc = plan.topology != "SOFTWARE_VNA_SWEEP"
        required_devices = ["vna"]
        if uses_rtc:
            required_devices.append("rtc")
        elif plan.beam_control_mode == "SOFTWARE_DIRECT":
            required_devices.append("beam_controller")
        if plan.test_type == "PATTERN":
            required_devices.append("turntable")
        for device_id in required_devices:
            self.devices.require(device_id)

        if plan.test_type == "CALIBRATION":
            # Disabled channels are explicit SKIPPED_DISABLED results and therefore count
            # towards completion even though no device command or VNA sweep is issued.
            total = len(coordinates.for_polarization(plan.polarization))
        else:
            total = int(
                inclusive_axis(plan.azimuth_start_deg, plan.azimuth_stop_deg, plan.azimuth_step_deg).size
                * inclusive_axis(plan.elevation_start_deg, plan.elevation_stop_deg, plan.elevation_step_deg).size
                * len(plan.beams)
            )
        if total == 0:
            raise ServiceError("NOT_RUNNABLE", "计划没有可执行测量单元", "run_prepare")
        rtc_configuration: dict[str, Any] = {}
        rtc_wave_entries: list[dict[str, Any]] = []
        if uses_rtc:
            rtc_configuration = await freeze_rtc_configuration(self.devices.require("rtc"), plan)
            rtc_source = self.devices.require("rtc").snapshot.source.value
            vna_source = self.devices.require("vna").snapshot.source.value
            turntable_source = self.devices.require("turntable").snapshot.source.value if plan.test_type == "PATTERN" else None
            # A simulator cannot generate a physical TTL trigger for real hardware.
            if rtc_source == "SIMULATED" and vna_source == "REAL":
                raise ServiceError("NOT_RUNNABLE", "真实矢网需要真实RTC提供硬件触发；模拟RTC请搭配模拟矢网", "run_prepare")
            if plan.topology == "RTC_CONTINUOUS" and rtc_source == "REAL" and turntable_source == "SIMULATED":
                raise ServiceError("NOT_RUNNABLE", "RTC连续模式需要真实转台提供位置脉冲；模拟转台请搭配模拟RTC", "run_prepare")
            row_points = inclusive_axis(plan.azimuth_start_deg, plan.azimuth_stop_deg, plan.azimuth_step_deg).size if plan.test_type == "PATTERN" else 1
            if row_points * len(plan.beams) * plan.frequency_points > 0xFFFFFFFF or total > 0x7FFFFFFF:
                raise ServiceError("NOT_RUNNABLE", "本轮计数超出RTC或测量文件编号范围", "run_prepare")

        if plan.test_type == "PATTERN" and plan.beam_control_mode == "SOFTWARE_DIRECT":
            # Compile every electronic beam before any real mechanical movement. A bad
            # angle/frequency mapping must fail in PREPARE, not after the turntable moves.
            frequencies = self._frequencies(plan)
            beam_command = profile.command_for_role("BEAM_SET")
            for beam in plan.beams:
                reference_frequency = beam.reference_frequency_hz or float((frequencies[0] + frequencies[-1]) / 2)
                profile.encode(
                    beam_command.command_id,
                    plan.array_id,
                    {
                        "off_axis_deg": beam.off_axis_deg,
                        "azimuth_deg": beam.azimuth_deg,
                        "frequency_ghz": reference_frequency / 1e9,
                        "signal_path": plan.signal_path,
                    },
                )

        # REAL readiness uses the profile's explicit initialization query. The returned
        # field, not a successful serial write, is the readiness evidence.
        transport = self.devices.require("beam_controller") if not uses_rtc and plan.beam_control_mode == "SOFTWARE_DIRECT" else None
        if transport is not None and transport.snapshot.source.value == "REAL" and profile.capabilities.get("initialization"):
            command = profile.command_for_role("INITIALIZE_QUERY")
            frame = profile.encode(command.command_id, plan.array_id, {})
            await self._send_antenna(plan, frame, command)

        if uses_rtc:
            if plan.test_type == "CALIBRATION":
                calibration_command = profile.command_for_role("CALIBRATION_WRITE")
                if calibration_command.response_mode != "SINGLE" or calibration_command.success_rule != "FRAME_EQUALS_REQUEST":
                    raise ServiceError("NOT_RUNNABLE", "RTC标校关闭确认需要单帧完整回显规则 FRAME_EQUALS_REQUEST", "run_prepare")
            if plan.beam_control_mode == "SOFTWARE_DIRECT":
                rtc_wave_entries = compile_rtc_wave_table(self.assets, wave_request_from_plan(plan))
                self.devices.note_configuration_change("rtc")
                table = await read_rtc_wave_table(self.devices.require("rtc"), rtc_wave_entries, write=True)
                rtc_configuration["wave_count"] = table["count"]
                rtc_configuration["waves_verified"] = table["verified"]
            else:
                rtc_configuration.update(wave_count=0, waves_verified=True)

        run_id = uuid.uuid4().hex
        record = RunRecord(
            run_id=run_id,
            plan=plan.model_copy(deep=True),
            state=RunState.PREPARED,
            created_at=datetime.now(timezone.utc).isoformat(),
            total=total,
            device_revisions={key: self.devices.device_revisions[key] for key in required_devices},
            rtc_configuration=rtc_configuration,
            rtc_wave_entries=rtc_wave_entries,
        )
        record.pause_gate.set()
        self.runs[run_id] = record
        await self.events.publish("run.status", run=record.public())
        return record

    def get(self, run_id: str) -> RunRecord:
        try:
            return self.runs[run_id]
        except KeyError as exc:
            raise ServiceError("NOT_FOUND", "运行不存在", "runs", run_id) from exc

    def current(self) -> RunRecord | None:
        owner = self.devices.control_owner
        if owner and owner.startswith("RUN:"):
            return self.runs.get(owner[4:])
        return next(reversed(self.runs.values()), None)

    def snapshot(self) -> dict[str, Any]:
        record = self.current()
        # No await: state, samples and sequence are one event-loop snapshot.
        return {
            "run": record.public() if record else None,
            "samples": list(record.samples) if record else [],
            "control_owner": self.devices.control_owner,
            "event_sequence": self.events.sequence,
        }

    async def _publish_sample(self, record: RunRecord, **payload: Any) -> None:
        event = await self.events.publish("run.sample", **payload)
        record.samples.append(event)

    async def _boundary(self, record: RunRecord) -> bool:
        if record.pause_requested and not record.stop_requested:
            record.pause_requested = False
            record.state = RunState.PAUSED
            await self.events.publish("run.status", run=record.public())
        await record.pause_gate.wait()
        return not record.stop_requested

    async def start(self, run_id: str) -> RunRecord:
        record = self.get(run_id)
        if record.state != RunState.PREPARED:
            raise ServiceError("INVALID_REQUEST", "只有已准备运行可以启动", "run_start", run_id)
        if self._active_lock.locked():
            raise ServiceError("CONTROL_LOCKED", "已有自动测试占用设备", "run_start", run_id)
        await self.devices.acquire_run_control(run_id)
        try:
            if any(self.devices.device_revisions.get(key) != revision for key, revision in record.device_revisions.items()):
                raise ServiceError("NOT_RUNNABLE", "设备配置或连接已改变，请重新准备测试", "run_start")
            if record.plan.topology != "SOFTWARE_VNA_SWEEP":
                await verify_frozen_rtc_configuration(self.devices.require("rtc"), record.rtc_configuration)
                await verify_rtc_wave_table(self.devices.require("rtc"), record.rtc_wave_entries)
            record.state = RunState.RUNNING
            record.task = asyncio.create_task(self._execute(record), name=f"antenna-run-{run_id}")
        except Exception:
            self.devices.release_run_control(run_id)
            raise
        await self.events.publish("run.status", run=record.public())
        return record

    async def pause(self, run_id: str) -> RunRecord:
        record = self.get(run_id)
        if record.state != RunState.RUNNING:
            raise ServiceError("INVALID_REQUEST", "当前运行不能暂停", "run_pause", run_id)
        record.pause_gate.clear()
        if record.plan.topology != "SOFTWARE_VNA_SWEEP":
            record.pause_requested = True
        else:
            record.state = RunState.PAUSED
        await self.events.publish("run.status", run=record.public())
        return record

    async def resume(self, run_id: str) -> RunRecord:
        record = self.get(run_id)
        if record.state != RunState.PAUSED:
            raise ServiceError("INVALID_REQUEST", "当前运行不能继续", "run_resume", run_id)
        record.state = RunState.RUNNING
        record.pause_requested = False
        record.pause_gate.set()
        await self.events.publish("run.status", run=record.public())
        return record

    async def stop(self, run_id: str) -> RunRecord:
        record = self.get(run_id)
        if record.state not in {RunState.RUNNING, RunState.PAUSED}:
            raise ServiceError("INVALID_REQUEST", "当前运行不能停止", "run_stop", run_id)
        record.stop_requested = True
        record.pause_requested = False
        record.state = RunState.STOPPING
        record.pause_gate.set()
        await self.events.publish("run.status", run=record.public())
        return record

    async def _execute(self, record: RunRecord) -> None:
        async with self._active_lock:
            try:
                if record.plan.test_type == "CALIBRATION":
                    if record.plan.topology != "SOFTWARE_VNA_SWEEP":
                        await self._calibrate_rtc(record)
                    else:
                        await self._calibrate(record)
                elif record.plan.topology != "SOFTWARE_VNA_SWEEP":
                    await self._pattern_rtc(record)
                else:
                    await self._pattern(record)
                # Pause/stop at the last atomic bundle also applies before completion/home.
                await self._boundary(record)
                if not record.stop_requested and record.completed != record.total:
                    raise ServiceError(
                        "DATA_INTEGRITY",
                        "测量项未全部确认，不能判定完成",
                        "run_complete",
                        record.run_id,
                        {"completed": record.completed, "total": record.total},
                    )
                terminal = RunState.STOPPED if record.stop_requested else RunState.COMPLETED
                assert record.store is not None
                record.result = record.store.finalize(terminal.value)
                record.state = terminal
                if terminal == RunState.COMPLETED and record.plan.test_type == "PATTERN":
                    # The closed-and-reopened HDF5 is the completion boundary. Publish
                    # COMPLETED before the independent post-run home; a home failure must
                    # not invalidate already verified measurement data.
                    record.cleanup_pending = True
                    await self.events.publish("run.status", run=record.public())
                    try:
                        home_result = await getattr(self.devices.require("turntable"), "home")(1)
                        cleanup = {"status": "SUCCESS", "axis": 1, "result": home_result}
                    except (Exception, asyncio.CancelledError) as home_exc:
                        cleanup_error = (
                            home_exc.as_dict()
                            if isinstance(home_exc, ServiceError)
                            else ServiceError("INTERNAL", str(home_exc) or "完成后的方位寻零被取消", "post_run_azimuth_home").as_dict()
                        )
                        cleanup = {"status": "FAILED", "axis": 1, "error": cleanup_error}
                        try:
                            await getattr(self.devices.require("turntable", ready=False), "stop")(1)
                        except Exception as stop_exc:
                            cleanup["stop_error"] = str(stop_exc)
                            await self._mark_turntable_unknown(str(stop_exc))
                    record.result["post_completion_azimuth_home"] = cleanup
                    await self.events.publish("run.post_complete", run=record.public(), cleanup=cleanup)
            except (Exception, asyncio.CancelledError) as exc:
                record.state = RunState.UNKNOWN if isinstance(exc, asyncio.CancelledError) or (isinstance(exc, ServiceError) and exc.side_effect_possible) else RunState.FAULTED
                if isinstance(exc, ServiceError):
                    record.error = exc.as_dict()
                else:
                    record.error = ServiceError("INTERNAL", str(exc) or "运行被取消", "run_execute").as_dict()
                if record.plan.test_type == "PATTERN":
                    # A motion timeout does not prove the axis stopped. Issue one software
                    # Stop and verify stationary readback before releasing the control lease.
                    try:
                        turntable = self.devices.require("turntable", ready=False)
                        await getattr(turntable, "stop")("all")
                    except Exception as stop_exc:
                        record.error["stop_error"] = str(stop_exc)
                        record.state = RunState.UNKNOWN
                        await self._mark_turntable_unknown(str(stop_exc))
                if record.rtc_session is not None:
                    try:
                        await record.rtc_session.close(failed=True)
                    except Exception as rtc_close_exc:
                        record.error["rtc_cleanup_error"] = str(rtc_close_exc)
                        record.state = RunState.UNKNOWN
                if record.store is not None and record.store.file.id.valid:
                    try:
                        record.result = record.store.finalize(record.state.value)
                    except Exception as finalize_exc:
                        record.error["finalize_error"] = str(finalize_exc)
            finally:
                if record.store is not None and record.store.file.id.valid:
                    record.store.file.close()
                record.store = None
                record.rtc_session = None
                record.pause_requested = False
                record.cleanup_pending = False
                self.devices.release_run_control(record.run_id)
                await self.events.publish("run.status", run=record.public())

    async def _mark_turntable_unknown(self, error: str) -> None:
        adapter = self.devices.devices.get("turntable")
        if adapter is not None:
            adapter.snapshot.update(state=DeviceState.UNKNOWN, error=error)
            await self.events.publish("device.status", device=await adapter.status())

    def _frequencies(self, plan: RunPlan) -> np.ndarray:
        return np.linspace(plan.frequency_start_hz, plan.frequency_stop_hz, plan.frequency_points)

    def _metadata(self, record: RunRecord, profile: ProtocolProfile, coordinates: Any) -> dict[str, Any]:
        selected = ["vna"] + (["turntable"] if record.plan.test_type == "PATTERN" else [])
        uses_rtc = record.plan.topology != "SOFTWARE_VNA_SWEEP"
        if uses_rtc:
            selected.insert(0, "rtc")
        elif record.plan.beam_control_mode == "SOFTWARE_DIRECT":
            selected.insert(0, "beam_controller")
        sources = {device_id: self.devices.require(device_id).snapshot.source.value for device_id in selected}
        device_evidence = {
            device_id: {
                "identity": self.devices.require(device_id).snapshot.identity,
                "details": dict(self.devices.require(device_id).snapshot.details),
            }
            for device_id in selected
        }
        evidence_sources = set(sources.values()) | {coordinates.evidence}
        evidence = next(iter(evidence_sources)) if len(evidence_sources) == 1 else "MIXED"
        return {
            "test_name": record.plan.name,
            "expected_units": record.total,
            "test_type": record.plan.test_type,
            "topology": record.plan.topology,
            "beam_control_mode": record.plan.beam_control_mode,
            "beam_command_sent": record.plan.beam_control_mode == "SOFTWARE_DIRECT",
            "beam_response_verified": not uses_rtc and record.plan.beam_control_mode == "SOFTWARE_DIRECT",
            "beam_state_source": ("RTC_TRANSMITTED_UNVERIFIED" if uses_rtc else "SOFTWARE_COMMAND_CONFIRMED")
            if record.plan.beam_control_mode == "SOFTWARE_DIRECT" else "USER_DECLARED_UNVERIFIED",
            "position_source": "TRIGGER_GRID" if record.plan.topology == "RTC_CONTINUOUS" else "READBACK",
            "signal_path": record.plan.signal_path,
            "polarization": record.plan.polarization,
            "s_parameter": record.plan.s_parameter,
            "profile_id": profile.profile_id,
            "profile_sha256": profile.file_sha256,
            "coordinate_id": coordinates.antenna_id,
            "coordinate_sha256": coordinates.file_sha256,
            "mapping_sha256": coordinates.mapping_sha256,
            "enabled_sha256": coordinates.enabled_sha256,
            "geometry_sha256": coordinates.geometry_sha256,
            "device_sources": sources,
            "device_evidence": device_evidence,
            "evidence": evidence,
            "plan_snapshot": record.plan.model_dump(),
        }

    async def _calibrate(self, record: RunRecord) -> MeasurementStore:
        plan = record.plan
        profile = self.assets.profile(plan.profile_id)
        coordinates = self.assets.coordinate(plan.coordinate_id)
        channels = coordinates.for_polarization(plan.polarization)
        frequencies = self._frequencies(plan)
        # Configure and read back the VNA before creating the result file. A bad SCPI
        # setup must not leave a misleading empty HDF5 run artifact.
        vna = self.devices.require("vna")
        await getattr(vna, "configure")(
            s_parameter=plan.s_parameter,
            frequencies_hz=frequencies,
            if_bandwidth_hz=plan.if_bandwidth_hz,
            source_power_dbm=plan.source_power_dbm,
            averaging_enabled=plan.averaging_enabled,
            averaging_count=plan.averaging_count,
            trigger_mode="INTERNAL_SINGLE",
        )
        path = timestamped_path(plan.output_directory, plan.base_filename, plan.test_type)
        record.output_path = str(path)
        store = MeasurementStore.create_calibration(
            path,
            run_id=record.run_id,
            metadata=self._metadata(record, profile, coordinates),
            frequencies_hz=frequencies,
            coordinates=coordinates,
            polarization=plan.polarization,
        )
        record.store = store
        command = profile.command_for_role("CALIBRATION_WRITE")
        for channel in channels:
            await record.pause_gate.wait()
            if record.stop_requested:
                break
            if not channel.enabled:
                record.completed += 1
                await self._publish_sample(
                    record,
                    run_id=record.run_id,
                    kind="CALIBRATION",
                    status="SKIPPED_DISABLED",
                    channel={"element": channel.element, "grid_row": channel.grid_row, "grid_column": channel.grid_column},
                    magnitude_db=None,
                    phase_deg=None,
                    completed=record.completed,
                    total=record.total,
                )
                await self.events.publish("run.status", run=record.public())
                continue
            opened = False
            measurement_error: Exception | None = None
            try:
                frame = profile.build_calibration_frame(
                    array_id=plan.array_id,
                    spi_no=channel.spi_no,
                    chip_no=channel.chip_no,
                    chip_channel_index=channel.chip_channel_index,
                    signal_path=plan.signal_path,
                )
                # From the send boundary onward the channel may be open even if its ACK
                # is lost. Never resend the opening command; finally sends ALL_OFF once.
                opened = True
                await self._send_antenna(plan, frame, command)
                if plan.settle_ms:
                    await asyncio.sleep(plan.settle_ms / 1000)
                values = await getattr(vna, "acquire")(frequencies, channel_element=channel.element)
                store.commit_channel(channel, values)
                record.completed += 1
                magnitudes_db = [20 * math.log10(abs(value)) if abs(value) > 0 else None for value in values]
                phases_deg = [
                    math.degrees(math.atan2(value.imag, value.real)) % 360 if abs(value) > 0 else None
                    for value in values
                ]
                await self._publish_sample(
                    record,
                    run_id=record.run_id,
                    kind="CALIBRATION",
                    status="CALIBRATED",
                    channel={"element": channel.element, "grid_row": channel.grid_row, "grid_column": channel.grid_column},
                    # Keep one scalar for compact consumers, while the full arrays let
                    # the desktop switch the live calibration heatmap between frequency
                    # points without repeating any VNA measurement.
                    magnitude_db=magnitudes_db[0],
                    phase_deg=phases_deg[0],
                    magnitudes_db=magnitudes_db,
                    phases_deg=phases_deg,
                    completed=record.completed,
                    total=record.total,
                )
                await self.events.publish("run.status", run=record.public())
            except (Exception, asyncio.CancelledError) as exc:
                measurement_error = exc
                store.fail_channel(channel, getattr(exc, "code", "FAILED"))
                raise
            finally:
                if opened:
                    # CALIBRATION_WRITE selects a whole chip control word. Explicitly send
                    # ALL_OFF after every measurement so a stopped or completed run cannot
                    # leave the last physical RF channel selected.
                    close_frame = profile.build_calibration_frame(
                        array_id=plan.array_id,
                        spi_no=channel.spi_no,
                        chip_no=channel.chip_no,
                        chip_channel_index=channel.chip_channel_index,
                        signal_path=plan.signal_path,
                        enabled=False,
                    )
                    try:
                        await self._send_antenna(plan, close_frame, command)
                    except Exception as close_exc:
                        store.log("ERROR", "calibration_close", "测量失败后的通道关闭未确认", {"error": str(close_exc)})
                        raise ServiceError(
                            "UNKNOWN", "标校通道关闭未确认，请检查阵面状态", "calibration_close",
                            record.run_id, {"measurement_error": str(measurement_error) if measurement_error else None,
                                            "close_error": str(close_exc)},
                            side_effect_possible=True, next_action="不要重发开通指令；请先核对阵面状态",
                        ) from close_exc
        return store

    async def _send_rtc_calibration_close(self, record: RunRecord, channel: Channel,
                                           profile: ProtocolProfile, command: Command) -> tuple[bytes, bytes]:
        rtc = self.devices.require("rtc", ready=False)
        request = profile.build_calibration_frame(
            array_id=record.plan.array_id, spi_no=channel.spi_no, chip_no=channel.chip_no,
            chip_channel_index=channel.chip_channel_index, signal_path=record.plan.signal_path, enabled=False,
        )
        status = await rtc.get_status()
        if (status.get("state") not in {"IDLE", "CONFIGURED", "COMPLETE"}
                or any(status.get(key) for key in ("tx_busy", "group_in_flight", "io_busy", "tr_running"))):
            # FAULT explicitly forbids opcode 30. Never clear a fault automatically
            # just to send a close, and never claim B0 proves the RF state changed.
            raise ServiceError("UNKNOWN", "RTC状态不允许发送关闭指令，请先现场核对阵面",
                               "rtc_calibration_close", details=status, side_effect_possible=True)
        if rtc.snapshot.source.value == "SIMULATED":
            # The explicit offline fixture follows the existing simulator's echo
            # behavior; physical RTC paths never synthesize or inject an E2 result.
            profile.validate_response(command, request, [request])
            rtc.queue_antenna_response(request, [request])

        def matches(response: bytes) -> bool:
            try:
                profile.validate_response(command, request, [response])
                return True
            except ServiceError:
                return False

        # send_frame captures a fresh RX cursor before one 30 and independently
        # waits for the E2 owned by this profile/request. No close retransmission.
        response = await rtc.send_frame(request, timeout_ms=command.timeout_ms,
                                        response_opcode=command.response_opcode, response_matcher=matches)
        frames = response if isinstance(response, list) else [response]
        profile.validate_response(command, request, frames)
        return request, frames[0]

    async def _calibrate_rtc(self, record: RunRecord) -> MeasurementStore:
        plan = record.plan
        profile, coordinates = self.assets.profile(plan.profile_id), self.assets.coordinate(plan.coordinate_id)
        channels = coordinates.for_polarization(plan.polarization)
        frequencies = self._frequencies(plan)
        frames = [bytes.fromhex(entry["frame_hex"]) for entry in record.rtc_wave_entries]
        addresses = {entry["element"]: entry["address"] for entry in record.rtc_wave_entries}
        rtc, vna = self.devices.require("rtc"), self.devices.require("vna")
        session = RtcAcquisition(rtc, vna, None, plan, record.rtc_configuration, frequencies, frames)
        record.rtc_session = session
        await session.configure()
        path = timestamped_path(plan.output_directory, plan.base_filename, plan.test_type)
        record.output_path = str(path)
        store = MeasurementStore.create_calibration(
            path, run_id=record.run_id, metadata=self._metadata(record, profile, coordinates),
            frequencies_hz=frequencies, coordinates=coordinates, polarization=plan.polarization,
        )
        record.store = store
        command = profile.command_for_role("CALIBRATION_WRITE")
        store.initialize_rtc(record.rtc_configuration, frames)
        store.file["rtc"].attrs["close_success_rule"] = command.success_rule
        for channel in channels:
            if channel.enabled:
                store.file["rtc/channel_measurements/wave_address"][channel.element] = addresses[channel.element]

        # Establish an off baseline for the chips participating in this run only.
        # This is per-SPI/chip/path ALL_OFF, not a claim that unknown array chips
        # are globally off. No pulse or measurement starts before these confirmations.
        chips = {}
        for channel in channels:
            if channel.enabled:
                chips.setdefault((channel.spi_no, channel.chip_no), channel)
        initial_frames = []
        for channel in chips.values():
            if not await self._boundary(record):
                break
            initial_request = profile.build_calibration_frame(
                array_id=plan.array_id, spi_no=channel.spi_no, chip_no=channel.chip_no,
                chip_channel_index=channel.chip_channel_index, signal_path=plan.signal_path, enabled=False)
            store.log("INFO", "rtc_initial_close", "关闭本次参与标校的芯片",
                      {"spi_no": channel.spi_no, "chip_no": channel.chip_no,
                       "request_hex": initial_request.hex(" ").upper()})
            try:
                request, response = await self._send_rtc_calibration_close(record, channel, profile, command)
                initial_frames.append(request)
                store.log("INFO", "rtc_initial_close", "芯片关闭已收到完整回显",
                          {"spi_no": channel.spi_no, "chip_no": channel.chip_no,
                           "request_hex": request.hex(" ").upper(), "response_hex": response.hex(" ").upper()})
            except (Exception, asyncio.CancelledError) as exc:
                store.log("ERROR", "rtc_initial_close", "芯片关闭未确认",
                          {"spi_no": channel.spi_no, "chip_no": channel.chip_no, "error": str(exc)})
                store.fail_channel(channel, "INITIAL_CLOSE_UNKNOWN")
                store.mark_rtc_channel_close(channel, "UNKNOWN")
                raise ServiceError("UNKNOWN", "RTC标校初始芯片关闭未确认，未开始测量",
                                   "rtc_calibration_initial_close", details={"error": str(exc)},
                                   side_effect_possible=True) from exc
        store.file["rtc"].create_dataset("initial_chip_close_frames",
            data=np.asarray([list(frame) for frame in initial_frames], dtype=np.uint8).reshape((-1, 22)))
        store.file.flush()

        for channel in channels:
            if not await self._boundary(record):
                break
            if not channel.enabled:
                record.completed += 1
                await self._publish_sample(
                    record, run_id=record.run_id, kind="CALIBRATION", status="SKIPPED_DISABLED",
                    channel={"element": channel.element, "grid_row": channel.grid_row, "grid_column": channel.grid_column},
                    magnitude_db=None, phase_deg=None, completed=record.completed, total=record.total,
                )
                await self.events.publish("run.status", run=record.public())
                continue
            may_be_open = False
            measurement_error = None
            values = None
            close_request = profile.build_calibration_frame(
                array_id=plan.array_id, spi_no=channel.spi_no, chip_no=channel.chip_no,
                chip_channel_index=channel.chip_channel_index, signal_path=plan.signal_path, enabled=False)
            try:
                await session.select_calibration_wave(addresses[channel.element])
                store.mark_rtc_channel_close(channel, "PENDING", request=close_request)
                # From this call onwards opcode 08 may have emitted the opening
                # wave even if its ACK is lost. Never resend 08 or the open frame.
                may_be_open = True
                spectra, evidence = await session.acquire_point([{"channel_element": channel.element}])
                values = spectra[0]
                store.commit_channel(channel, values)
                store.commit_rtc_channel(channel, evidence)
            except (Exception, asyncio.CancelledError) as exc:
                measurement_error = exc
                store.fail_channel(channel, getattr(exc, "code", "FAILED"))
                raise
            finally:
                if may_be_open:
                    try:
                        if measurement_error is not None:
                            await session.suspend(failed=True)
                        request, response = await self._send_rtc_calibration_close(record, channel, profile, command)
                        store.mark_rtc_channel_close(channel, "CONFIRMED", request=request, response=response)
                    except (Exception, asyncio.CancelledError) as close_exc:
                        store.mark_rtc_channel_close(channel, "UNKNOWN", request=close_request)
                        store.fail_channel(channel, "CLOSE_UNKNOWN")
                        store.log("ERROR", "rtc_calibration_close", "通道关闭未确认，停止后续通道",
                                  {"element": channel.element, "error": close_exc.as_dict() if isinstance(close_exc, ServiceError) else str(close_exc)})
                        raise ServiceError(
                            "UNKNOWN", "RTC标校通道关闭未确认，请核对阵面状态", "rtc_calibration_close",
                            str(channel.element), {
                                "measurement_error": measurement_error.as_dict() if isinstance(measurement_error, ServiceError)
                                                     else str(measurement_error) if measurement_error else None,
                                "close_error": close_exc.as_dict() if isinstance(close_exc, ServiceError) else str(close_exc)},
                            side_effect_possible=True, next_action="不要重发开通；RTC故障时先现场恢复，禁止自动清故障补发",
                        ) from close_exc
            assert values is not None
            record.completed += 1
            magnitudes = [20 * math.log10(abs(value)) if abs(value) > 0 else None for value in values]
            phases = [math.degrees(math.atan2(value.imag, value.real)) % 360 if abs(value) > 0 else None for value in values]
            await self._publish_sample(
                record, run_id=record.run_id, kind="CALIBRATION", status="CALIBRATED",
                channel={"element": channel.element, "grid_row": channel.grid_row, "grid_column": channel.grid_column},
                magnitude_db=magnitudes[0], phase_deg=phases[0], magnitudes_db=magnitudes, phases_deg=phases,
                completed=record.completed, total=record.total,
            )
            await self.events.publish("run.status", run=record.public())
        await session.close()
        return store

    async def _pattern(self, record: RunRecord) -> MeasurementStore:
        plan = record.plan
        profile = self.assets.profile(plan.profile_id)
        coordinates = self.assets.coordinate(plan.coordinate_id)
        frequencies = self._frequencies(plan)
        # Mechanical scan points are frozen from start/stop/step. The elevation
        # axis is the heatmap row and the azimuth axis is the heatmap column.
        azimuths = inclusive_axis(plan.azimuth_start_deg, plan.azimuth_stop_deg, plan.azimuth_step_deg)
        elevations = inclusive_axis(plan.elevation_start_deg, plan.elevation_stop_deg, plan.elevation_step_deg)
        turntable = self.devices.require("turntable")
        vna = self.devices.require("vna")
        beam_command = (
            profile.command_for_role("BEAM_SET")
            if plan.beam_control_mode == "SOFTWARE_DIRECT"
            else None
        )
        # Verify the complete VNA configuration before creating the HDF5 file or moving
        # either real axis. This keeps SCPI/configuration faults outside the motion phase.
        await getattr(vna, "configure")(
            s_parameter=plan.s_parameter,
            frequencies_hz=frequencies,
            if_bandwidth_hz=plan.if_bandwidth_hz,
            source_power_dbm=plan.source_power_dbm,
            averaging_enabled=plan.averaging_enabled,
            averaging_count=plan.averaging_count,
            trigger_mode="INTERNAL_SINGLE",
        )
        path = timestamped_path(plan.output_directory, plan.base_filename, plan.test_type)
        record.output_path = str(path)
        store = MeasurementStore.create_pattern(
            path,
            run_id=record.run_id,
            metadata=self._metadata(record, profile, coordinates),
            frequencies_hz=frequencies,
            beams=[
                {
                    "beam_id": beam.beam_id,
                    "off_axis_deg": beam.off_axis_deg,
                    "azimuth_deg": beam.azimuth_deg,
                    "reference_frequency_hz": beam.reference_frequency_hz or float((frequencies[0] + frequencies[-1]) / 2),
                }
                for beam in plan.beams
            ],
        )
        record.store = store
        # One complete VNA internal sweep is stored per spatial-point/beam bundle.
        # EXTERNAL_FIXED has one user-declared bundle and deliberately performs no
        # beam-controller call, regardless of whether that device is connected.
        point_id = 0
        bundle_id = 0
        stop_after_boundary = False
        for row_index, elevation in enumerate(elevations):
            await record.pause_gate.wait()
            if record.stop_requested:
                break
            # 真实设备顺序（混合模式同样适用）：先移动俯仰轴 2 并等待位置/速度
            # 回读确认，再把方位轴 1 回到本行起点。任何一步失败都不会继续采集。
            elevation_readback = await getattr(turntable, "move_to")(2, float(elevation), plan.move_speed_deg_s)
            await record.pause_gate.wait()
            if record.stop_requested:
                break
            # Every row begins at the same azimuth; this is intentionally not a snake scan.
            row_start_readback = await getattr(turntable, "move_to")(1, float(azimuths[0]), plan.move_speed_deg_s)
            for point_index, azimuth in enumerate(azimuths):
                await record.pause_gate.wait()
                if record.stop_requested:
                    stop_after_boundary = True
                    break
                # The row-start move already confirms point zero. Do not resend the same
                # absolute motion command; later points each move exactly once.
                azimuth_readback = row_start_readback if point_index == 0 else await getattr(turntable, "move_to")(
                    1,
                    float(azimuth),
                    plan.move_speed_deg_s,
                )
                for beam_index, beam in enumerate(plan.beams):
                    if not await self._boundary(record):
                        stop_after_boundary = True
                        break
                    if beam_command is not None:
                        reference_frequency = beam.reference_frequency_hz or float((frequencies[0] + frequencies[-1]) / 2)
                        frame = profile.encode(
                            beam_command.command_id,
                            plan.array_id,
                            {
                                "off_axis_deg": beam.off_axis_deg,
                                "azimuth_deg": beam.azimuth_deg,
                                "frequency_ghz": reference_frequency / 1e9,
                                "signal_path": plan.signal_path,
                            },
                        )
                        # 波控适配器可以是模拟器，而 turntable/vna 可以是真实设备。
                        # 软件直控时先确认波控应答，再触发一次 VNA 完整扫频。
                        await self._send_antenna(plan, frame, beam_command)
                    if plan.settle_ms:
                        await asyncio.sleep(plan.settle_ms / 1000)
                    values = await getattr(vna, "acquire")(
                        frequencies,
                        azimuth_deg=float(azimuth),
                        elevation_deg=float(elevation),
                    )
                    store.commit_point(
                        bundle_id=bundle_id,
                        point_id=point_id,
                        row_index=row_index,
                        point_index=point_index,
                        azimuth_deg=float(azimuth),
                        elevation_deg=float(elevation),
                        actual_azimuth_deg=float(azimuth_readback.get("position", azimuth)),
                        actual_elevation_deg=float(elevation_readback.get("position", elevation)),
                        beam_index=beam_index,
                        beam_id=beam.beam_id,
                        values=values,
                    )
                    record.completed += 1
                    center = values[len(values) // 2]
                    amplitude = abs(center)
                    phase = math.degrees(math.atan2(center.imag, center.real)) % 360 if amplitude > 0 else None
                    # Keep the confirmed per-frequency trace in the live event so the
                    # renderer can change frequency without re-running completed points.
                    magnitudes_db = [20 * math.log10(abs(value)) if abs(value) > 0 else None for value in values]
                    phases_deg = [
                        math.degrees(math.atan2(value.imag, value.real)) % 360 if abs(value) > 0 else None
                        for value in values
                    ]
                    await self._publish_sample(
                        record,
                        run_id=record.run_id,
                        kind="PATTERN",
                        status="COMPLETE",
                        point={"azimuth_deg": float(azimuth), "elevation_deg": float(elevation)},
                        beam={"beam_id": beam.beam_id, "beam_index": beam_index},
                        magnitude_db=20 * math.log10(amplitude) if amplitude > 0 else None,
                        phase_deg=phase,
                        magnitudes_db=magnitudes_db,
                        phases_deg=phases_deg,
                        completed=record.completed,
                        total=record.total,
                    )
                    await self.events.publish("run.status", run=record.public())
                    bundle_id += 1
                if stop_after_boundary:
                    break
                point_id += 1
            # Stop/pause prohibits further row-return motion. Only a continuing run
            # returns to the next row start; completion homes axis 1 after finalization.
            if not await self._boundary(record):
                break
            if row_index < len(elevations) - 1:
                await getattr(turntable, "move_to")(1, float(azimuths[0]), plan.move_speed_deg_s)
            if stop_after_boundary:
                break
        return store

    async def _pattern_rtc(self, record: RunRecord) -> MeasurementStore:
        plan = record.plan
        profile, coordinates = self.assets.profile(plan.profile_id), self.assets.coordinate(plan.coordinate_id)
        frequencies = self._frequencies(plan)
        azimuths = inclusive_axis(plan.azimuth_start_deg, plan.azimuth_stop_deg, plan.azimuth_step_deg)
        elevations = inclusive_axis(plan.elevation_start_deg, plan.elevation_stop_deg, plan.elevation_step_deg)
        frames = [bytes.fromhex(entry["frame_hex"]) for entry in record.rtc_wave_entries]
        rtc, vna, turntable = (self.devices.require(name) for name in ("rtc", "vna", "turntable"))
        session = RtcAcquisition(rtc, vna, turntable, plan, record.rtc_configuration, frequencies, frames)
        record.rtc_session = session
        await session.configure()
        path = timestamped_path(plan.output_directory, plan.base_filename, plan.test_type)
        record.output_path = str(path)
        store = MeasurementStore.create_pattern(
            path, run_id=record.run_id, metadata=self._metadata(record, profile, coordinates),
            frequencies_hz=frequencies,
            beams=[{"beam_id": beam.beam_id, "off_axis_deg": beam.off_axis_deg, "azimuth_deg": beam.azimuth_deg,
                    "reference_frequency_hz": beam.reference_frequency_hz or float((frequencies[0] + frequencies[-1]) / 2)}
                   for beam in plan.beams],
        )
        record.store = store
        store.initialize_rtc(record.rtc_configuration, frames)
        for row_index, elevation in enumerate(elevations):
            if not await self._boundary(record):
                break
            elevation_readback = await turntable.move_to(2, float(elevation), plan.move_speed_deg_s)
            if not await self._boundary(record):
                break
            start_readback = await turntable.move_to(1, float(azimuths[0]), plan.move_speed_deg_s)
            if plan.topology == "RTC_CONTINUOUS":
                # Once the row starts, normal pause/stop waits until all samples
                # have been read, checked and committed. Faults still stop promptly.
                if not await self._boundary(record):
                    break
                contexts = [{"azimuth_deg": float(azimuth), "elevation_deg": float(elevation)}
                            for azimuth in azimuths for _beam in plan.beams]
                values, evidence = await session.acquire_row(azimuths, contexts)
                store.commit_rtc_group(
                    row_index=row_index, first_point_index=0, point_count=len(azimuths),
                    evidence=evidence, start_azimuth=float(start_readback["position"]),
                    end_azimuth=float(evidence["motion"]["position"]),
                    actual_elevation=float(elevation_readback["position"]),
                )
                for point_index, azimuth in enumerate(azimuths):
                    for beam_index, beam in enumerate(plan.beams):
                        await self._commit_rtc_sample(record, row_index, point_index, float(azimuth), float(elevation),
                                                      math.nan, float(elevation_readback["position"]), beam_index,
                                                      values[point_index * len(plan.beams) + beam_index])
                # Disarm before pause or the next row's repositioning. A return
                # motion must never be accepted as another measurement group.
                await session.suspend()
                if not await self._boundary(record):
                    break
            else:
                for point_index, azimuth in enumerate(azimuths):
                    if not await self._boundary(record):
                        break
                    position = start_readback if point_index == 0 else await turntable.move_to(
                        1, float(azimuth), plan.move_speed_deg_s)
                    if not await self._boundary(record):
                        break
                    contexts = [{"azimuth_deg": float(azimuth), "elevation_deg": float(elevation)} for _ in plan.beams]
                    values, evidence = await session.acquire_point(contexts)
                    store.commit_rtc_group(
                        row_index=row_index, first_point_index=point_index, point_count=1,
                        evidence=evidence, start_azimuth=float(position["position"]),
                        end_azimuth=float(position["position"]), actual_elevation=float(elevation_readback["position"]),
                    )
                    for beam_index, beam in enumerate(plan.beams):
                        await self._commit_rtc_sample(record, row_index, point_index, float(azimuth), float(elevation),
                                                      float(position["position"]), float(elevation_readback["position"]),
                                                      beam_index, values[beam_index])
                if record.stop_requested:
                    break
        await session.close()
        return store

    async def _commit_rtc_sample(self, record: RunRecord, row_index: int, point_index: int,
                                 azimuth: float, elevation: float, actual_azimuth: float,
                                 actual_elevation: float, beam_index: int, values: np.ndarray) -> None:
        assert record.store is not None
        beam = record.plan.beams[beam_index]
        columns = inclusive_axis(record.plan.azimuth_start_deg, record.plan.azimuth_stop_deg,
                                 record.plan.azimuth_step_deg).size
        record.store.commit_point(
            bundle_id=record.completed, point_id=int(row_index * columns + point_index),
            row_index=row_index, point_index=point_index, azimuth_deg=azimuth, elevation_deg=elevation,
            actual_azimuth_deg=actual_azimuth, actual_elevation_deg=actual_elevation,
            beam_index=beam_index, beam_id=beam.beam_id, values=values,
        )
        record.completed += 1
        magnitudes = [20 * math.log10(abs(value)) if abs(value) > 0 else None for value in values]
        phases = [math.degrees(math.atan2(value.imag, value.real)) % 360 if abs(value) > 0 else None for value in values]
        await self._publish_sample(
            record, run_id=record.run_id, kind="PATTERN", status="COMPLETE",
            point={"azimuth_deg": azimuth, "elevation_deg": elevation},
            beam={"beam_id": beam.beam_id, "beam_index": beam_index},
            magnitude_db=magnitudes[len(values) // 2], phase_deg=phases[len(values) // 2],
            magnitudes_db=magnitudes, phases_deg=phases, completed=record.completed, total=record.total,
        )
        await self.events.publish("run.status", run=record.public())

    async def _send_antenna(self, plan: RunPlan, frame: bytes, command: Command) -> bytes | list[bytes]:
        transport_name = "SERIAL"
        await self.events.publish(
            "device.raw",
            device_id="beam_controller" if transport_name == "SERIAL" else "rtc",
            direction="TX",
            command_id=command.command_id,
            raw_hex=frame.hex(" ").upper(),
            transport=transport_name,
            context="AUTOMATIC_RUN",
        )
        beam = self.devices.require("beam_controller")
        profile = self.assets.profile(plan.profile_id)
        options: dict[str, Any] = {}
        if command.response_mode == "MULTI":
            rule = profile.response_rules.get(command.response_opcode if command.response_opcode is not None else command.opcode)
            if rule is None or rule.mode != "MULTI":
                raise ServiceError("NOT_RUNNABLE", "多帧指令缺少组帧规则", "antenna_response", command.command_id)
            if beam.snapshot.source.value == "SIMULATED":
                raise ServiceError("NOT_RUNNABLE", "当前模拟波控机没有此多帧应答样本", "antenna_response", command.command_id)
            options["response_rule"] = rule
        try:
            response = await getattr(beam, "send_frame")(
                frame,
                timeout_ms=command.timeout_ms,
                response_opcode=command.response_opcode,
                **options,
            )
            profile.validate_response(command, frame, response if isinstance(response, list) else [response])
        except ServiceError as exc:
            # A rejected ACK does not undo the already-transmitted hardware command.
            exc.side_effect_possible = True
            raise
        if beam.snapshot.source.value == "SIMULATED":
            # A real adapter publishes RX from its continuous physical reader. The simulator
            # has no reader, so reproduce that one event here without duplicating real frames.
            await self.events.publish(
                "device.raw",
                device_id="beam_controller" if transport_name == "SERIAL" else "rtc",
                direction="RX",
                command_id=command.command_id,
                raw_hex=response.hex(" ").upper(),
                parsed=self.assets.decode_matching_frame(response),
                transport=transport_name,
                context="AUTOMATIC_RUN",
            )
        return response
