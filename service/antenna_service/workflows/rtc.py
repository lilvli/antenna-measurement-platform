from __future__ import annotations

import asyncio
import math
import time
from typing import Any

import numpy as np

from antenna_service.errors import ServiceError
from antenna_service.models import DeviceSource, RunPlan, RtcWaveRequest
from antenna_service.coordinates import validate_profile_coordinate_pair


DEFAULT_TR = {"period_us": 100.0, "high_us": 20.0, "delay_us": 1.0}
DEFAULT_TIMING = {
    "pulse_qualification_us": 10, "host_timeout_ms": 3000,
    "antenna_tx_timeout_ms": 1000, "vna_ready_timeout_ms": 10000,
    "ack_timeout_ms": 1000, "stable_wait_us": 0, "sync_io_timeout_ms": 1000,
}


def wave_request_from_plan(plan: RunPlan) -> RtcWaveRequest:
    return RtcWaveRequest(
        profile_id=plan.profile_id, coordinate_id=plan.coordinate_id, test_type=plan.test_type,
        array_id=plan.array_id, signal_path=plan.signal_path, polarization=plan.polarization,
        reference_frequency_hz=(plan.frequency_start_hz + plan.frequency_stop_hz) / 2,
        beams=plan.beams,
    )


def compile_rtc_wave_table(assets: Any, request: RtcWaveRequest) -> list[dict[str, Any]]:
    profile = assets.profile(request.profile_id)
    entries: list[dict[str, Any]] = []
    if request.test_type == "CALIBRATION":
        if not request.coordinate_id:
            raise ServiceError("NOT_RUNNABLE", "标校波位预装需要当前坐标表", "rtc_wave_compile")
        coordinates = assets.coordinate(request.coordinate_id)
        validate_profile_coordinate_pair(profile, coordinates)
        if request.polarization not in coordinates.polarizations:
            raise ServiceError("NOT_RUNNABLE", "坐标表没有所选极化", "rtc_wave_compile")
        for channel in coordinates.for_polarization(request.polarization):
            if not channel.enabled:
                continue
            frame = profile.build_calibration_frame(
                array_id=request.array_id, spi_no=channel.spi_no, chip_no=channel.chip_no,
                chip_channel_index=channel.chip_channel_index, signal_path=request.signal_path,
            )
            entries.append({"address": len(entries) + 1, "element": channel.element,
                            "label": f"通道 {channel.element} / SPI {channel.spi_no} 芯片 {channel.chip_no}",
                            "frame_hex": frame.hex(" ").upper(), "status": "PREVIEW"})
    else:
        command = profile.command_for_role("BEAM_SET")
        for beam in request.beams:
            frame = profile.encode(command.command_id, request.array_id, {
                "off_axis_deg": beam.off_axis_deg, "azimuth_deg": beam.azimuth_deg,
                "frequency_ghz": (beam.reference_frequency_hz or request.reference_frequency_hz) / 1e9,
                "signal_path": request.signal_path,
            })
            entries.append({"address": len(entries) + 1, "label": beam.beam_id,
                            "frame_hex": frame.hex(" ").upper(), "status": "PREVIEW"})
    if len(entries) > 512 or any(len(bytes.fromhex(entry["frame_hex"])) != 22 for entry in entries):
        raise ServiceError("NOT_RUNNABLE", "RTC波位表最多512条，每条必须为完整22字节", "rtc_wave_compile")
    return entries


async def read_rtc_wave_table(rtc: Any, entries: list[dict[str, Any]], *, write: bool = False) -> dict[str, Any]:
    """One implementation for previewed preloading, run preparation and verification.

    Addresses are assigned by the compiler. Read-only mismatch is reviewable; writes
    update only differing/missing entries, then require exact 24 readback. No ARM,
    table commit/erase, TR start, antenna transfer, or physical movement occurs here.
    """
    status = await rtc.get_status()
    if write:
        check_rtc_status(status, idle=True)
    elif status.get("state") not in {"IDLE", "CONFIGURED", "COMPLETE", "FAULT"}:
        raise ServiceError("NOT_RUNNABLE", "RTC活动期间不允许读取波位表", "rtc_wave_read", details=status)
    capability = await rtc.get_capability()
    capacity = min(512, int(capability["max_wave_entries"]))
    if len(entries) > capacity:
        raise ServiceError("NOT_RUNNABLE", "本次波位数超过RTC实际容量", "rtc_wave_table")
    result = []
    written = 0
    for entry in entries:
        expected = bytes.fromhex(entry["frame_hex"])
        actual = None
        try:
            actual = await rtc.read_wave_entry(entry["address"])
        except ServiceError as exc:
            if (exc.details or {}).get("error_code") != 8:
                raise
        if write and actual != expected:
            await rtc.write_wave_entry(entry["address"], expected)
            written += 1
            actual = await rtc.read_wave_entry(entry["address"])
            if actual != expected:
                raise ServiceError("DATA_INTEGRITY", "RTC波位写后回读不一致", "rtc_wave_verify",
                                   str(entry["address"]), side_effect_possible=True)
        match = actual == expected
        result.append({**entry, "readback_hex": actual.hex(" ").upper() if actual is not None else None,
                       "status": "VERIFIED" if write and match else "MATCH" if match else "EMPTY" if actual is None else "MISMATCH"})
    return {"entries": result, "count": len(result), "capacity": capacity,
            "verified": all(item["status"] in {"VERIFIED", "MATCH"} for item in result),
            "written_count": written, "source": rtc.snapshot.source.value}


async def verify_rtc_wave_table(rtc: Any, entries: list[dict[str, Any]]) -> None:
    result = await read_rtc_wave_table(rtc, entries)
    if not result["verified"]:
        raise ServiceError("NOT_RUNNABLE", "RTC内部波位与准备计划不一致，请重新写入并准备",
                           "rtc_wave_verify", details={"mismatches": [entry["address"] for entry in result["entries"]
                                                                    if entry["status"] != "MATCH"]})


def _tr_fingerprint(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value.get(key) for key in ("mode", "period_ticks", "high_ticks", "delay_ticks", "tr_state", "tr_source")}


def check_rtc_status(status: dict[str, Any], *, idle: bool = False) -> None:
    if (status.get("state") == "FAULT" or status.get("fault_code", status.get("first_fault", 0))
            or status.get("result_uncertain") or status.get("activity_monitor_error")):
        raise ServiceError("DEVICE_FAULT", "RTC故障或执行结果不确定", "rtc_status",
                           details={"status": status}, side_effect_possible=True)
    if idle and (status.get("state") not in {"IDLE", "CONFIGURED", "COMPLETE"}
                 or any(status.get(key) for key in ("group_in_flight", "tx_busy", "io_busy", "tr_running"))
                 or status.get("unknown_result")):
        raise ServiceError("NOT_RUNNABLE", "请先停止RTC调试TR并确认发送和同步接口排空", "rtc_prepare",
                           details={"status": status})


async def freeze_rtc_configuration(rtc: Any, plan: RunPlan) -> dict[str, Any]:
    status = await rtc.get_status()
    check_rtc_status(status, idle=True)
    capability = await rtc.get_capability()
    tr = await rtc.get_tr_config()
    timing = await rtc.get_timing()
    required_bits = 0x19 | (1 if plan.beam_control_mode == "SOFTWARE_DIRECT" else 0)
    if (capability.get("capability_bits", 0) & required_bits) != required_bits:
        raise ServiceError("NOT_RUNNABLE", "RTC未声明本次测量所需RDY/连续计数能力", "rtc_prepare")
    if capability.get("clock_hz", 0) <= 0:
        raise ServiceError("NOT_RUNNABLE", "RTC实际时钟无效", "rtc_prepare")
    tr_configured = tr.get("mode") in {1, 2, "TX", "RX"}
    selected_tr = {key: float(tr[key]) if tr_configured else value for key, value in DEFAULT_TR.items()}
    selected_timing = {key: int(timing.get(key) or value) for key, value in DEFAULT_TIMING.items()}
    selected_timing["stable_wait_us"] = plan.settle_ms * 1000
    return {
        "tr": {"mode": plan.signal_path, **selected_tr},
        "timing": selected_timing, "clock_hz": capability["clock_hz"],
        "source_tr": _tr_fingerprint(tr), "source_timing": timing,
        "defaults_used": not tr_configured, "buffer_kind": "REPEATED_SWEEP",
    }


async def verify_frozen_rtc_configuration(rtc: Any, frozen: dict[str, Any]) -> None:
    status = await rtc.get_status()
    check_rtc_status(status, idle=True)
    if (_tr_fingerprint(await rtc.get_tr_config()) != frozen["source_tr"]
            or await rtc.get_timing() != frozen["source_timing"]
            or (await rtc.get_capability())["clock_hz"] != frozen["clock_hz"]):
        raise ServiceError("NOT_RUNNABLE", "RTC参数已改变，请重新准备测试", "rtc_prepare_stale")


class RtcAcquisition:
    """RTC/VNA group orchestration; plan loops, pause and HDF ownership stay in RunEngine."""

    def __init__(self, rtc: Any, vna: Any, turntable: Any, plan: RunPlan,
                 configuration: dict[str, Any], frequencies: np.ndarray, frames: list[bytes]):
        self.rtc, self.vna, self.turntable = rtc, vna, turntable
        self.plan, self.configuration = plan, configuration
        self.frequencies, self.frames = frequencies, frames
        self.beams = 1 if plan.test_type == "CALIBRATION" else max(len(frames), 1)
        self.points_per_wave = int(frequencies.size)
        self.last_buffer: dict[str, Any] = {}
        self._motion: asyncio.Task | None = None
        self._simulated_pulses: asyncio.Task | None = None
        self._started = False
        self._closed = False
        self._stop_sent = False

    async def configure(self) -> None:
        await verify_frozen_rtc_configuration(self.rtc, self.configuration)
        await self.vna.configure(
            s_parameter=self.plan.s_parameter, frequencies_hz=self.frequencies,
            if_bandwidth_hz=self.plan.if_bandwidth_hz, source_power_dbm=self.plan.source_power_dbm,
            averaging_enabled=self.plan.averaging_enabled, averaging_count=self.plan.averaging_count,
            trigger_mode="EXTERNAL_POINT",
        )
        # Configuration is sent once and read back; no output starts before ARM.
        self._started = True
        await self.rtc.set_timing(**self.configuration["timing"])
        self.configuration["tr_readback"] = await self.rtc.set_tr_config(**self.configuration["tr"])
        # Waves were written and verified during PREPARE. Start only rereads;
        # calibration selects one preloaded address at a time, never all channels.
        for address, frame in enumerate(self.frames, 1):
            if await self.rtc.read_wave_entry(address) != frame:
                raise ServiceError("DATA_INTEGRITY", "RTC波位与准备时不一致，未重复写入", "rtc_wave_verify",
                                   str(address), side_effect_possible=True)
        wave_count = min(1, len(self.frames)) if self.plan.test_type == "CALIBRATION" else len(self.frames)
        await self.rtc.set_counts(wave_count, self.points_per_wave)
        await self.rtc.set_trigger_mode("continuous" if self.plan.topology == "RTC_CONTINUOUS" else "software")
        config = await self.rtc.get_config()
        if (config["wave_count"] != wave_count or config["points_per_wave"] != self.points_per_wave
                or config["start_address"] != 1):
            raise ServiceError("DATA_INTEGRITY", "RTC计数设置回读不一致", "rtc_configuration")
        timing = await self.rtc.get_timing()
        if timing != self.configuration["timing"]:
            raise ServiceError("DATA_INTEGRITY", "RTC时序回读不一致", "rtc_configuration")

    async def select_calibration_wave(self, address: int) -> None:
        if self.plan.test_type != "CALIBRATION" or not 1 <= address <= len(self.frames):
            raise ServiceError("INVALID_REQUEST", "RTC标校地址无效", "rtc_calibration_select")
        check_rtc_status(await self.rtc.get_status(), idle=True)
        await self.rtc.set_counts(1, self.points_per_wave, address)
        config = await self.rtc.get_config()
        if (config["wave_count"], config["points_per_wave"], config["start_address"]) != (1, self.points_per_wave, address):
            raise ServiceError("DATA_INTEGRITY", "RTC标校地址/次数回读不一致", "rtc_calibration_select")

    def _group_timeout(self) -> float:
        # Bounds include all selected beams, per-point waits and J30J settle.
        # Simulated acquisition uses accelerated time; the same count checks remain.
        if self.rtc.snapshot.source == DeviceSource.SIMULATED:
            return max(5.0, self.beams * self.points_per_wave * 0.01 + 2)
        point_timeout = self.configuration["timing"]["vna_ready_timeout_ms"] / 1000
        tx_timeout = self.configuration["timing"]["antenna_tx_timeout_ms"] / 1000
        settings = self.vna.snapshot.details.get("settings", {})
        sweep_seconds = float(settings.get("sweep_time_seconds", 0))
        if math.isfinite(sweep_seconds) and sweep_seconds > 0:
            return max(5.0, self.beams * (sweep_seconds * self.plan.averaging_count * 2
                       + tx_timeout + self.points_per_wave * self.configuration["tr"]["period_us"] / 1e6) + 5)
        return self.beams * (tx_timeout + self.points_per_wave * point_timeout * 3) + 5

    async def _prepare_group_buffer(self, groups: int) -> None:
        if groups * self.beams * self.points_per_wave > 0xFFFFFFFF:
            raise ServiceError("NOT_RUNNABLE", "本轮点数超出RTC的uint32计数范围", "rtc_buffer_prepare")
        self.last_buffer = await self.vna.prepare_buffered_acquisition(self.frequencies, groups * self.beams)
        if self.last_buffer["triggers_per_sweep"] != self.points_per_wave:
            raise ServiceError("NOT_RUNNABLE", "矢网实际触发次数与RTC每波位次数不一致", "rtc_buffer_prepare")
        await self._arm_once()

    async def _arm_once(self) -> None:
        try:
            await self.rtc.arm()
        except ServiceError as exc:
            if not exc.side_effect_possible:
                raise
            status = await self.rtc.get_status()
            progress = await self.rtc.get_progress()
            check_rtc_status(status)
            if not self._arm_ready(status, progress):
                raise
            # Only the observed new ARMED + zero counters resolves this ARM. No resend.
            self.rtc.client.acknowledge_unknown_result()
        status, progress = await self._status_progress()
        if not self._arm_ready(status, progress):
            raise ServiceError("DATA_INTEGRITY", "RTC尚未确认ARM就绪或已收到计划外脉冲", "rtc_arm",
                               details={"status": status, "progress": progress}, side_effect_possible=True)
        self._stop_sent = False

    @staticmethod
    def _arm_ready(status: dict[str, Any], progress: dict[str, Any]) -> bool:
        return (status.get("state") == "ARMED" and status.get("stage") != 1
                and status.get("tr_running")
                and not any(status.get(key) for key in ("io_busy", "tx_busy", "group_in_flight", "result_uncertain", "fault_code"))
                and not any(progress.get(key) for key in ("io_busy", "group_in_flight", "fault", "result_uncertain"))
                and not any(progress[key] for key in ("accepted_groups", "completed_groups", "valid_triggers", "completed_points")))

    async def _status_progress(self) -> tuple[dict[str, Any], dict[str, Any]]:
        status = await self.rtc.get_status()
        check_rtc_status(status)
        progress = await self.rtc.get_progress()
        if any(progress.get(key) for key in ("fault", "result_uncertain")):
            raise ServiceError("DEVICE_FAULT", "RTC采集故障或结果不确定", "rtc_measurement",
                               details=progress, side_effect_possible=True)
        return status, progress

    async def _wait_groups(self, groups: int, *, timeout: float | None = None, input_finished: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
        deadline = time.monotonic() + (timeout if timeout is not None else self._group_timeout())
        expected_points = groups * self.beams * self.points_per_wave
        while True:
            status, progress = await self._status_progress()
            if (progress["accepted_groups"] > groups or progress["completed_groups"] > groups
                    or progress["valid_triggers"] > expected_points or progress["completed_points"] > expected_points):
                raise ServiceError("DATA_INTEGRITY", "RTC接受了计划之外的触发", "rtc_counts", details=progress,
                                   side_effect_possible=True)
            drained = not any(status.get(key) for key in ("group_in_flight", "tx_busy", "io_busy"))
            drained = drained and not progress.get("group_in_flight") and not progress.get("io_busy")
            if drained and progress["accepted_groups"] == progress["completed_groups"] == groups:
                if progress["valid_triggers"] != expected_points or progress["completed_points"] != expected_points:
                    raise ServiceError("DATA_INTEGRITY", "RTC有效触发/完成点数与计划不一致", "rtc_counts",
                                       details=progress, side_effect_possible=True)
                return status, progress
            if input_finished and drained and progress["accepted_groups"] == progress["completed_groups"] < groups:
                raise ServiceError("DATA_INTEGRITY", "行运动已结束，但RTC实际触发组数少于计划；未补点或重发",
                                   "rtc_counts", details=progress, side_effect_possible=True)
            if time.monotonic() >= deadline:
                raise ServiceError("TIMEOUT", "RTC未在期限内完成本组/本行并排空", "rtc_wait",
                                   details=progress, side_effect_possible=True)
            await asyncio.sleep(0.02)

    async def acquire_point(self, context: list[dict[str, float]]) -> tuple[np.ndarray, dict[str, Any]]:
        await self._prepare_group_buffer(1)
        try:
            await self.rtc.software_trigger()
        except ServiceError as exc:
            if not exc.side_effect_possible:
                raise
            # ACK loss is resolved from actual accepted/completed counters, never a
            # second SOFTWARE_TRIGGER. A request not proved accepted remains unknown.
            status, progress = await self._status_progress()
            if progress["accepted_groups"] != 1:
                raise
        status, progress = await self._wait_groups(1)
        if status.get("state") != "COMPLETE" or status.get("tr_running"):
            raise ServiceError("DATA_INTEGRITY", "走停组完成后RTC未停止TR", "rtc_group_complete")
        if self.rtc.client.unknown_result:
            self.rtc.client.acknowledge_unknown_result()
        values = await self.vna.read_buffered_acquisition(
            self.frequencies, self.beams, completed_trigger_count=progress["completed_points"],
            sample_contexts=context,
        )
        return values, {"progress": progress, "status": status, "buffer": dict(self.last_buffer)}

    async def _feed_simulated_pulses(self, groups: int) -> None:
        # Only a simulated RTC receives software-injected external edges. REAL
        # serial hardware never receives fabricated progress or a software substitute.
        loop = asyncio.get_running_loop()
        origin = loop.time()
        interval = self.plan.azimuth_step_deg / self.plan.move_speed_deg_s
        for group in range(groups):
            await asyncio.sleep(max(0.0, origin + group * interval - loop.time()))
            if not await self.rtc.client.external_trigger():
                raise ServiceError("DEVICE_FAULT", "模拟RTC忙时收到新的位置脉冲，未排队或补测",
                                   "rtc_simulated_trigger", side_effect_possible=True)

    async def acquire_row(self, azimuths: np.ndarray, contexts: list[dict[str, float]]) -> tuple[np.ndarray, dict[str, Any]]:
        groups = int(azimuths.size)
        await self._prepare_group_buffer(groups)
        travel_seconds = abs(float(azimuths[-1] - azimuths[0])) / self.plan.move_speed_deg_s
        self._motion = asyncio.create_task(self.turntable.scan_azimuth(
            float(azimuths[0]), float(azimuths[-1]), self.plan.azimuth_step_deg,
            self.plan.move_speed_deg_s, timeout_seconds=travel_seconds + 30,
        ))
        if self.rtc.snapshot.source == DeviceSource.SIMULATED:
            self._simulated_pulses = asyncio.create_task(self._feed_simulated_pulses(groups))
        try:
            deadline = time.monotonic() + travel_seconds + groups * self._group_timeout() + 30
            while not self._motion.done() or (self._simulated_pulses is not None and not self._simulated_pulses.done()):
                await self._status_progress()
                if self._motion.done() and self._motion.exception() is not None:
                    await self._motion
                if self._simulated_pulses is not None and self._simulated_pulses.done():
                    await self._simulated_pulses
                if time.monotonic() >= deadline:
                    raise ServiceError("TIMEOUT", "RTC连续行运动或采集未结束", "rtc_row", side_effect_possible=True)
                await asyncio.sleep(0.02)
            motion = await self._motion
            if not motion.get("pulse_output_disabled"):
                raise ServiceError("DATA_INTEGRITY", "转台未确认关闭行扫描脉冲，不能读取本行",
                                   "rtc_row", details=motion, side_effect_possible=True)
            if self._simulated_pulses is not None:
                await self._simulated_pulses
            status, progress = await self._wait_groups(groups, input_finished=True)
            # scan_azimuth has confirmed end position and three zero-speed readings;
            # only now is it valid to treat the row as having no more trigger pulses.
            values = await self.vna.read_buffered_acquisition(
                self.frequencies, groups * self.beams, completed_trigger_count=progress["completed_points"],
                sample_contexts=contexts,
            )
            return values, {"progress": progress, "status": status, "buffer": dict(self.last_buffer),
                            "motion": motion}
        finally:
            for task in (self._motion, self._simulated_pulses):
                if task is not None and not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            self._motion = self._simulated_pulses = None

    async def suspend(self, *, failed: bool = False) -> None:
        """Close the group entrance at a point/row boundary without resending STOP."""
        deadline = time.monotonic() + self.configuration["timing"]["sync_io_timeout_ms"] / 1000 + 2
        while True:
            status = await self.rtc.get_status()
            if status.get("stage") != 1:
                break
            if time.monotonic() >= deadline:
                raise ServiceError("TIMEOUT", "RTC仍在准备阶段，无法确认安全停止", "rtc_cleanup", side_effect_possible=True)
            await asyncio.sleep(0.02)
        unknown = self.rtc.client.unknown_result
        if unknown:
            opcode = unknown.get("opcode")
            if opcode in {9, 10}:
                # The previously sent STOP may already be draining. Read only;
                # re-entering cleanup never sends it a second time.
                self._stop_sent = True
            elif opcode in {7, 8}:
                progress = await self.rtc.get_progress()
                accepted = self._arm_ready(status, progress) if opcode == 7 else (
                    progress["accepted_groups"] == 1 and not progress.get("result_uncertain")
                    and status.get("state") in {"RUNNING", "COMPLETE"})
                if not accepted and status.get("state") != "FAULT":
                    raise ServiceError("UNKNOWN_RESULT", "RTC请求结果尚未确认，未追加停止命令", "rtc_cleanup",
                                       details=unknown, side_effect_possible=True)
                if accepted:
                    self.rtc.client.acknowledge_unknown_result()
            elif status.get("state") not in {"IDLE", "CONFIGURED", "COMPLETE", "FAULT"}:
                raise ServiceError("UNKNOWN_RESULT", "RTC配置结果尚未确认，不能追加动作", "rtc_cleanup",
                                   details=unknown, side_effect_possible=True)
        if status.get("state") in {"ARMED", "RUNNING", "STOPPING"} and not self._stop_sent:
            self._stop_sent = True
            try:
                await (self.rtc.stop_immediate() if failed else self.rtc.stop_graceful())
            except ServiceError as exc:
                if not exc.side_effect_possible:
                    raise
                # ACK loss: leave the unknown result in place until output drain
                # and a matching COMPLETE stop result have actually been observed.
        while True:
            status = await self.rtc.get_status()
            drained = not any(status.get(key) for key in ("group_in_flight", "io_busy", "tx_busy", "tr_running"))
            if drained and status.get("state") in {"IDLE", "CONFIGURED", "COMPLETE", "FAULT"}:
                pending = self.rtc.client.unknown_result
                if pending and pending.get("opcode") in {9, 10} and status.get("state") == "COMPLETE":
                    result = status.get("last_result")
                    if result == (2 if pending["opcode"] == 9 else 3):
                        self.rtc.client.acknowledge_unknown_result()
                return
            if time.monotonic() >= deadline:
                raise ServiceError("TIMEOUT", "RTC停止后未确认输出排空", "rtc_cleanup", side_effect_possible=True)
            await asyncio.sleep(0.02)

    async def close(self, *, failed: bool = False) -> None:
        if not self._started or self._closed:
            return
        errors: dict[str, str] = {}
        try:
            await self.suspend(failed=failed)
        except Exception as exc:
            errors["rtc"] = str(exc)
        try:
            await self.vna.abort_buffered_acquisition()
        except Exception as exc:
            errors["vna"] = str(exc)
        if errors:
            raise ServiceError("DEVICE_FAULT", "RTC测试收尾未确认", "rtc_cleanup", details=errors,
                               side_effect_possible=True)
        self._closed = True
