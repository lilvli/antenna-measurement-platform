from __future__ import annotations

import asyncio
import re
import time

import numpy as np
import pytest

from antenna_service.devices.real import RealTurntable, RealVna
from antenna_service.devices.simulated import SimulatedRtc, SimulatedTurntable, SimulatedVna
from antenna_service.errors import ServiceError
from antenna_service.events import EventBus
from test_real_device_contracts import FakeImac, FakeVnaInstrument


FREQUENCIES = np.array([8e9, 8.005e9, 8.01e9])


class BufferedInstrument(FakeVnaInstrument):
    def __init__(self):
        super().__init__()
        self.correction = "OFF"
        self.channel2_mode = "HOLD"
        self.additional_parameter = None
        self.records = {1: [1, 2, 3, 4, 5, 6], 2: [10, 20, 30, 40, 50, 60]}
        self.memory_error = False
        self.atba_readback = None

    def query(self, command):
        self.queries.append(command)
        commands = {
            "SENS1:AVER:MODE?": "SENS1:AVER:MODE ", "SENS1:SWE:MODE?": "SENS1:SWE:MODE ",
            "SENS1:SWE:TRIG:MODE?": "SENS1:SWE:TRIG:MODE ", "TRIG:SOUR?": "TRIG:SOUR ",
            "TRIG:SCOP?": "TRIG:SCOP ", "TRIG:ROUT:INP?": "TRIG:ROUT:INP ",
            "TRIG:TYPE?": "TRIG:TYPE ", "TRIG:SLOP?": "TRIG:SLOP ", "TRIG:READ:POL?": "TRIG:READ:POL ",
            "SYST:DATA:MEM:MEAS7:REP?": "SYST:DATA:MEM:MEAS7:REP ",
            "CONT:SIGN:TRIG:ATBA?": "CONT:SIGN:TRIG:ATBA ",
        }
        if command == "CONT:SIGN:TRIG:ATBA?" and self.atba_readback is not None:
            return self.atba_readback
        if command in commands:
            prefix = commands[command]
            return next(item[len(prefix):] for item in reversed(self.writes) if item.startswith(prefix))
        if command == "SYST:CHAN:CAT?":
            return '"1,2"'
        if command == "SENS2:SWE:MODE?":
            return self.channel2_mode
        if command == "SYST:MEAS:CAT? 1":
            return '"7"'
        if command == "CALC1:PAR:CAT:EXT? DEF" and self.additional_parameter:
            return f'"ANTENNA_MEAS,S11,user,{self.additional_parameter}"'
        if command == "CALC1:MEAS7:CORR:STAT?":
            return "0" if self.correction == "OFF" else "1"
        if command == "CALC1:MEAS7:CORR:TYPE?":
            return self.correction
        if command == "CALC1:MEAS7:CORR:IND?":
            return "MAST"
        if command == "TRIG:STAT:READ? MEAS":
            return "1"
        if command == "SYST:DATA:MEM:SIZE?":
            return "0" if self.memory_error else str(int(self.query("SYST:DATA:MEM:MEAS7:REP?")) * 3 * 8)
        return super().query(command)

    def query_ascii_values(self, command, *, container=np.array):
        match = re.fullmatch(r"SYST:DATA:MEM:READ:MEAS7:REP(\d+)\? RI", command)
        if match:
            self.queries.append(command)
            return container(self.records[int(match.group(1))])
        return super().query_ascii_values(command, container=container)


async def configured_vna(*, averaging=False):
    vna = RealVna("MOCK")
    instrument = BufferedInstrument()
    vna._instrument = instrument
    await vna.configure(s_parameter="S11", frequencies_hz=FREQUENCIES,
                        if_bandwidth_hz=1000, source_power_dbm=-10,
                        averaging_enabled=averaging, averaging_count=4 if averaging else 1,
                        trigger_mode="EXTERNAL_POINT")
    return vna, instrument


@pytest.mark.asyncio
async def test_buffer_preserves_repeat_order_measurement_identity_and_point_averaging():
    vna, instrument = await configured_vna(averaging=True)
    plan = await vna.prepare_buffered_acquisition(FREQUENCIES, 2)
    assert plan["triggers_per_sweep"] == 3  # Four internal averages remain one trigger per frequency.
    assert plan["memory_bytes"] == 48
    assert "SENS1:SWE:MODE CONT" in instrument.writes
    assert not any("SWE:GRO:COUN" in command for command in instrument.writes)
    result = await vna.read_buffered_acquisition(FREQUENCIES, 2, completed_trigger_count=6)
    np.testing.assert_array_equal(result, [[1 + 2j, 3 + 4j, 5 + 6j], [10 + 20j, 30 + 40j, 50 + 60j]])
    assert instrument.writes.index("SENS1:SWE:MODE CONT") < len(instrument.writes) - 1
    assert not any("CORR OFF" in command or "PAR:DEL" in command or "MEM:RES" in command for command in instrument.writes)
    with pytest.raises(ServiceError):
        await vna.read_buffered_acquisition(FREQUENCIES, 2, completed_trigger_count=6)


@pytest.mark.asyncio
@pytest.mark.parametrize("correction,accepted", [("OFF", True), ("Response(S11)", True),
                                               ("Full 1 Port(1)", True), ("Full 2 Port(1,2)", False)])
async def test_buffer_reads_actual_calibration_and_does_not_silently_disable_it(correction, accepted):
    vna, instrument = await configured_vna()
    instrument.correction = correction
    if accepted:
        plan = await vna.prepare_buffered_acquisition(FREQUENCIES, 2)
        assert plan["correction_types"]["ANTENNA_MEAS"] == correction
    else:
        with pytest.raises(ServiceError, match="缓冲准备失败"):
            await vna.prepare_buffered_acquisition(FREQUENCIES, 2)
        assert "SENS1:SWE:MODE CONT" not in instrument.writes
    assert not any("CORR OFF" in command for command in instrument.writes)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["count", "nan", "short", "identity", "frequency"])
async def test_bad_buffer_is_never_returned_as_completed_data(failure):
    vna, instrument = await configured_vna()
    await vna.prepare_buffered_acquisition(FREQUENCIES, 2)
    if failure == "nan":
        instrument.records[2][-1] = float("nan")
    elif failure == "short":
        instrument.records[2].pop()
    elif failure == "identity":
        instrument.measurement_name = "USER_TRACE"
    with pytest.raises(ServiceError):
        await vna.read_buffered_acquisition(FREQUENCIES + (1000 if failure == "frequency" else 0), 2,
                                           completed_trigger_count=5 if failure == "count" else 6)


@pytest.mark.asyncio
async def test_allocation_failure_and_other_triggerable_channel_prevent_continuous_start():
    for failure in ("memory", "channel", "source"):
        vna, instrument = await configured_vna()
        instrument.memory_error = failure == "memory"
        instrument.channel2_mode = "CONT" if failure == "channel" else "HOLD"
        instrument.additional_parameter = "S22" if failure == "source" else None
        with pytest.raises(ServiceError):
            await vna.prepare_buffered_acquisition(FREQUENCIES, 2)
        assert "SENS1:SWE:MODE CONT" not in instrument.writes


@pytest.mark.asyncio
async def test_buffer_capacity_has_no_arbitrary_ceiling_and_abort_does_not_read_data():
    vna, instrument = await configured_vna()
    plan = await vna.prepare_buffered_acquisition(FREQUENCIES, 20_000_000)
    assert plan["sweep_count"] == 20_000_000
    await vna.abort_buffered_acquisition()
    assert instrument.writes[-2] == "SENS1:SWE:MODE HOLD"
    assert instrument.writes[-1].startswith('SYST:DATA:MEM:CLOS "ANTENNA_')
    assert not any("MEM:READ:" in command for command in instrument.queries)


class ScanImac(FakeImac):
    def __init__(self):
        super().__init__()
        self.scan_calls = []
        self.stop_calls = []
        self.disable_calls = []

    def SetTableEquEnable(self, device, axis, enabled):
        self.disable_calls.append((device, axis, enabled))
        self.m_values[7104] = self.m_values[7105] = int(enabled)

    def MoveToPosByType(self, *args):
        self.scan_calls.append(args)
        self.positions[1] = args[3] + 2  # Vendor's 0.0002 degree overrun.
        self.m_values[7104] = self.m_values[7105] = 1

    def Stop(self, device, axis):
        self.stop_calls.append((device, axis))


@pytest.mark.asyncio
async def test_scan_uses_scaled_existing_parameters_and_returns_row_end_evidence(tmp_path):
    turntable = RealTurntable(str(tmp_path / "ImacFxDll.dll"))
    imac = ScanImac()
    turntable._imac = imac
    result = await turntable.scan_azimuth(0, 2, 0.5, 1.2345)
    assert imac.scan_calls == [(0, 1, 0, 20000, 5000, 12345, 12345, 0.0, 0)]
    assert result["planned_point_count"] == 5
    assert result["interval_count"] == 4
    assert result["position"] == pytest.approx(2.0002)
    assert result["position_source"] == "ROW_END_READBACK"
    assert result["endpoint_pulse_count_verified"] is False
    assert result["pulse_output_disabled"] is True
    assert result["pulse_enable_readback"] == {"7104": 0, "7105": 0}
    assert imac.disable_calls == [(0, 1, False)]


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["not_at_start", "busy", "fractional_interval", "single_point", "speed_precision"])
async def test_scan_preflight_rejects_unsafe_or_unrepresentable_request_without_motion(tmp_path, problem):
    turntable = RealTurntable(str(tmp_path / "ImacFxDll.dll"))
    imac = ScanImac()
    turntable._imac = imac
    imac.positions[1] = 10000 if problem == "not_at_start" else 0
    imac.m_values[7107] = int(problem == "busy")
    with pytest.raises(ServiceError):
        await turntable.scan_azimuth(0, 0 if problem == "single_point" else 2,
                                    0.3 if problem == "fractional_interval" else 0.5,
                                    1.23456 if problem == "speed_precision" else 1)
    assert imac.scan_calls == []
    assert imac.disable_calls == []


@pytest.mark.asyncio
async def test_simulated_buffer_labels_are_separate_from_completion_counts():
    vna = SimulatedVna()
    await vna.configure(s_parameter="S11", frequencies_hz=FREQUENCIES, if_bandwidth_hz=1000,
                        source_power_dbm=-10, averaging_enabled=False, averaging_count=1, trigger_mode="EXTERNAL_POINT")
    await vna.prepare_buffered_acquisition(FREQUENCIES, 2)
    with pytest.raises(ServiceError):
        await vna.read_buffered_acquisition(FREQUENCIES, 2, completed_trigger_count=3)
    data = await vna.read_buffered_acquisition(FREQUENCIES, 2, completed_trigger_count=6,
                                             sample_contexts=[{"azimuth_deg": 0}, {"azimuth_deg": 10}])
    assert data.shape == (2, 3)
    assert not np.allclose(data[0], data[1])


@pytest.mark.asyncio
async def test_rtc_packet_log_does_not_duplicate_type_event_and_antenna_rx_is_independent():
    events = EventBus()
    queue = events.subscribe()
    rtc = SimulatedRtc(events=events)
    raw = "A5 5A"
    await rtc._publish_client_event({"event": "raw", "direction": "RX", "raw_hex": raw})
    await rtc._publish_client_event({"event": "ANTENNA_RX", "direction": "RX", "raw_hex": raw, "payload_hex": "AA55"})
    messages = []
    while not queue.empty():
        messages.append(queue.get_nowait())
    assert sum(message.get("context") == "RTC_FRAME" for message in messages) == 1
    assert sum(message.get("context") == "ANTENNA_RECEIVE" for message in messages) == 1
    assert sum(message["type"] == "device.rtc" for message in messages) == 1


@pytest.mark.asyncio
async def test_rtc_disconnect_stops_debug_output_and_waits_for_drain():
    rtc = SimulatedRtc()
    await rtc.connect()
    await rtc.set_timing()
    await rtc.set_tr_config("TX", 1000, 100, 1)
    await rtc.start_debug_tr()
    assert (await rtc.get_status())["debug_context"]
    await rtc.disconnect()
    assert rtc.snapshot.state.value == "DISCONNECTED"
    assert rtc.client.firmware.tr_state == 0


@pytest.mark.asyncio
async def test_rtc_disconnect_stops_armed_formal_output_using_formal_stop():
    rtc = SimulatedRtc()
    await rtc.connect()
    await rtc.set_timing()
    await rtc.set_tr_config("TX", 1000, 100, 1)
    await rtc.set_counts(0, 1)
    await rtc.set_trigger_mode("continuous")
    await rtc.arm()
    await rtc.disconnect()
    assert rtc.snapshot.state.value == "DISCONNECTED"
    assert rtc.client.firmware.tr_state == 0


@pytest.mark.asyncio
async def test_short_scan_does_not_finish_on_stale_start_or_moving_end_position(tmp_path):
    class StaleOrMovingImac(ScanImac):
        def __init__(self, moving):
            super().__init__()
            self.moving = moving

        def MoveToPosByType(self, *args):
            self.scan_calls.append(args)
            if self.moving:
                self.positions[1] = args[3]

        def GetVelStr(self, _device, axis):
            return "1" if self.moving and axis == 1 and self.scan_calls else "0"

    for moving in (False, True):
        turntable = RealTurntable(str(tmp_path / "ImacFxDll.dll"))
        imac = StaleOrMovingImac(moving)
        turntable._imac = imac
        with pytest.raises(ServiceError, match="连续扫描未确认"):
            await turntable.scan_azimuth(0, 0.001, 0.0005, 1, timeout_seconds=0.03)
        assert len(imac.scan_calls) == 1


@pytest.mark.asyncio
async def test_simulated_scan_obeys_duration_and_stop_cancels_without_finishing_row():
    turntable = SimulatedTurntable()
    started = time.perf_counter()
    result = await turntable.scan_azimuth(0, 0.1, 0.05, 1)
    assert time.perf_counter() - started >= 0.1
    assert result["pulse_output_disabled"]
    task = asyncio.create_task(turntable.scan_azimuth(0.1, 10.1, 1, 1))
    await asyncio.sleep(0.03)
    telemetry = await turntable.read_all_axes()
    assert 0.1 < telemetry["positions"]["1"] < 10.1
    started = time.perf_counter()
    await turntable.stop(1)
    with pytest.raises(ServiceError, match="连续扫描已被软件停止"):
        await asyncio.wait_for(task, timeout=0.2)
    assert time.perf_counter() - started < 0.2
    assert turntable.velocities[1] == 0
    assert turntable.positions[1] < 10.1


@pytest.mark.asyncio
async def test_scan_disable_failure_reports_unknown_and_never_retries(tmp_path):
    class FailedDisable(ScanImac):
        def SetTableEquEnable(self, device, axis, enabled):
            self.disable_calls.append((device, axis, enabled))
            # Pretend driver returns success but actual enable bits remain set.

    turntable = RealTurntable(str(tmp_path / "ImacFxDll.dll"))
    imac = FailedDisable()
    turntable._imac = imac
    with pytest.raises(ServiceError) as failure:
        await turntable.scan_azimuth(0, 2, 1, 1)
    assert failure.value.code == "UNKNOWN"
    assert len(imac.disable_calls) == len(imac.scan_calls) == 1
    assert turntable.snapshot.state.value == "UNKNOWN"


@pytest.mark.asyncio
async def test_cancelled_scan_disables_pulse_output_once(tmp_path):
    class MovingScan(ScanImac):
        def GetVelStr(self, _device, axis):
            return "10000" if self.scan_calls and axis == 1 else "0"

    turntable = RealTurntable(str(tmp_path / "ImacFxDll.dll"))
    imac = MovingScan()
    turntable._imac = imac
    task = asyncio.create_task(turntable.scan_azimuth(0, 2, 1, 1))
    while not imac.scan_calls:
        await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert imac.disable_calls == [(0, 1, False)]
    assert imac.m_values[7104] == imac.m_values[7105] == 0


@pytest.mark.asyncio
async def test_simulated_scan_early_timer_wakeup_does_not_complete_travel(monkeypatch):
    real_wait_for = asyncio.wait_for
    calls = 0

    async def early_wait_for(awaitable, *, timeout):
        nonlocal calls
        calls += 1
        if calls <= 2:
            # Simulate a timer expiration before any actual travel time elapsed.
            awaitable.close()
            raise TimeoutError
        return await real_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(asyncio, "wait_for", early_wait_for)
    turntable = SimulatedTurntable()
    started = time.perf_counter()
    result = await turntable.scan_azimuth(0, 0.05, 0.025, 1)
    assert calls >= 3
    assert time.perf_counter() - started >= 0.05
    assert result["position"] == pytest.approx(0.05)
    assert result["pulse_output_disabled"]


@pytest.mark.asyncio
@pytest.mark.parametrize("readback", ["1", "nan", "0.5"])
async def test_buffer_refuses_to_arm_when_early_trigger_memory_is_not_disabled(readback):
    vna, instrument = await configured_vna()
    instrument.atba_readback = readback
    with pytest.raises(ServiceError, match="缓冲准备失败"):
        await vna.prepare_buffered_acquisition(FREQUENCIES, 2)
    assert instrument.writes.count("CONT:SIGN:TRIG:ATBA 0") == 1
    assert "SENS1:SWE:MODE CONT" not in instrument.writes


@pytest.mark.asyncio
async def test_buffer_disables_early_trigger_memory_and_autoscales_only_after_first_complete_row():
    vna, instrument = await configured_vna()
    instrument.atba_readback = "+0"
    autoscale = lambda: [item for item in instrument.writes if item.endswith(":Y:AUTO")]
    assert autoscale() == []
    plan = await vna.prepare_buffered_acquisition(FREQUENCIES, 2)
    assert plan["accept_trigger_before_armed"] is False
    assert autoscale() == []
    await vna.read_buffered_acquisition(FREQUENCIES, 2, completed_trigger_count=6)
    assert len(autoscale()) == 1
    await vna.prepare_buffered_acquisition(FREQUENCIES, 2)
    await vna.read_buffered_acquisition(FREQUENCIES, 2, completed_trigger_count=6)
    assert len(autoscale()) == 1


@pytest.mark.asyncio
async def test_rtc_status_preserves_fault_and_unknown_until_readback_is_clean():
    responses = iter([
        {"state": "ARMED", "activity_monitor_error": {"code": "TIMEOUT"}},
        {"state": "DONE", "unknown_result": {"opcode": 8}},
        {"state": "DONE", "result_uncertain": True},
        {"state": "FAULT", "fault_code": 5, "result_uncertain": True},
        {"state": "READY", "fault_code": 0, "activity_monitor_error": None,
         "unknown_result": None, "result_uncertain": False},
    ])

    class StatusClient:
        async def get_status(self):
            return next(responses)

    rtc = SimulatedRtc()
    rtc.client = StatusClient()
    for expected in ("FAULT", "UNKNOWN", "UNKNOWN", "FAULT", "READY"):
        await rtc.get_status()
        assert rtc.snapshot.state.value == expected
