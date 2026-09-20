from __future__ import annotations

import asyncio
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import pytest_asyncio
from pydantic import ValidationError

from antenna_service.devices.manager import DeviceManager
from antenna_service.errors import ServiceError
from antenna_service.events import EventBus
from antenna_service.models import BeamDefinition, DeviceSource, RunPlan
from antenna_service.storage.hdf5_store import inspect_data_file
from antenna_service.workflows.engine import RunEngine
from antenna_service.workflows.rtc import DEFAULT_TIMING, RtcAcquisition


@pytest_asyncio.fixture
async def rtc_setup(loaded_assets):
    assets, profile, coordinates = loaded_assets
    events = EventBus()
    devices = DeviceManager(events, frame_decoder=assets.decode_matching_frame)
    for device in ("rtc", "vna", "turntable"):
        await devices.connect(device, DeviceSource.SIMULATED, {})
    engine = RunEngine(assets, devices, events)
    yield engine, devices, profile, coordinates
    record = engine.current()
    if record is not None and record.task is not None and not record.task.done():
        record.task.cancel()
        await asyncio.gather(record.task, return_exceptions=True)
    for adapter in devices.devices.values():
        await adapter.disconnect()


def plan_for(tmp_path, profile, coordinates, **changes):
    values = dict(test_type="PATTERN", topology="RTC_STOP_AND_GO", profile_id=profile.asset_id,
                  coordinate_id=coordinates.asset_id, output_directory=str(tmp_path), base_filename="rtc_run",
                  frequency_start_hz=8e9, frequency_stop_hz=8.2e9, frequency_points=3,
                  azimuth_start_deg=0, azimuth_stop_deg=2, azimuth_step_deg=1,
                  elevation_start_deg=0, elevation_stop_deg=0, move_speed_deg_s=20, settle_ms=0,
                  beams=[BeamDefinition(beam_id="B1"), BeamDefinition(beam_id="B2", off_axis_deg=10)])
    values.update(changes)
    return RunPlan(**values)


@pytest.mark.asyncio
@pytest.mark.parametrize("topology", ["RTC_STOP_AND_GO", "RTC_CONTINUOUS"])
async def test_rtc_pattern_full_group_or_row_retains_counts_and_source(tmp_path, rtc_setup, topology):
    engine, devices, profile, coordinates = rtc_setup
    plan = plan_for(tmp_path, profile, coordinates, topology=topology,
                    elevation_stop_deg=1, elevation_step_deg=1)
    record = await engine.prepare(plan)
    assert record.total == 12
    assert record.rtc_configuration["tr"] == {"mode": "TX", "period_us": 100, "high_us": 20, "delay_us": 1}
    await engine.start(record.run_id)
    await record.task
    assert record.state.value == "COMPLETED", record.error
    assert record.completed == 12
    assert "beam_controller" not in devices.devices
    with h5py.File(record.output_path, "r") as file:
        assert file["measurements/real"].shape == (12, 3)
        assert file["rtc/wave_frames"].shape == (2, 22)
        assert file["metadata"].attrs["evidence"] == "SIMULATED"
        assert not file["metadata"].attrs["beam_response_verified"]
        acquisitions = file["rtc/acquisitions"]
        if topology == "RTC_CONTINUOUS":
            assert acquisitions["accepted_groups"][...].tolist() == [3, 3]
            assert acquisitions["completed_points"][...].tolist() == [18, 18]
            assert np.all(np.isnan(file["spatial_points/actual_azimuth_deg"][...]))
            assert file["metadata"].attrs["position_source"] == "TRIGGER_GRID"
        else:
            assert acquisitions["completed_points"][...].tolist() == [6] * 6
            assert np.all(np.isfinite(file["spatial_points/actual_azimuth_deg"][...]))
    assert inspect_data_file(record.output_path)["schema"]["status"] == "COMPLETED"
    assert (await devices.require("rtc").get_status())["state"] == "COMPLETE"


@pytest.mark.asyncio
async def test_rtc_external_fixed_writes_no_wave_entries(tmp_path, rtc_setup):
    engine, devices, profile, coordinates = rtc_setup
    plan = plan_for(tmp_path, profile, coordinates, beam_control_mode="EXTERNAL_FIXED",
                    beams=[BeamDefinition()], azimuth_stop_deg=0)
    record = await engine.prepare(plan)
    await engine.start(record.run_id)
    await record.task
    assert record.state.value == "COMPLETED", record.error
    with h5py.File(record.output_path, "r") as file:
        assert file["rtc/wave_frames"].shape == (0, 22)
        assert file["metadata"].attrs["beam_state_source"] == "USER_DECLARED_UNVERIFIED"
        assert file["rtc/acquisitions/completed_points"][0] == 3


@pytest.mark.asyncio
async def test_continuous_pause_waits_for_read_commit_and_disarm_without_return(tmp_path, rtc_setup):
    engine, devices, profile, coordinates = rtc_setup
    plan = plan_for(tmp_path, profile, coordinates, topology="RTC_CONTINUOUS", elevation_stop_deg=1)
    record = await engine.prepare(plan)
    vna = devices.require("vna")
    read = vna.read_buffered_acquisition
    paused_once = False

    async def pause_during_row_read(*args, **kwargs):
        nonlocal paused_once
        if not paused_once:
            paused_once = True
            await engine.pause(record.run_id)
            assert record.state.value == "RUNNING" and record.pause_requested
        return await read(*args, **kwargs)

    vna.read_buffered_acquisition = pause_during_row_read
    await engine.start(record.run_id)
    for _ in range(250):
        if record.state.value == "PAUSED" or record.task.done():
            break
        await asyncio.sleep(0.02)
    assert record.state.value == "PAUSED", record.error
    assert record.completed == 6
    assert devices.require("turntable").positions[1] == 2
    rtc_status = await devices.require("rtc").get_status()
    assert rtc_status["state"] == "COMPLETE" and not rtc_status["tr_running"]
    with h5py.File(record.output_path, "r") as file:
        assert len(file["spatial_points/bundle_id"]) == 6
    await engine.resume(record.run_id)
    await record.task
    assert record.state.value == "COMPLETED", record.error


@pytest.mark.asyncio
async def test_rtc_bad_count_never_reads_buffer_or_marks_complete(tmp_path, rtc_setup):
    engine, devices, profile, coordinates = rtc_setup
    record = await engine.prepare(plan_for(tmp_path, profile, coordinates, azimuth_stop_deg=0))
    rtc, vna = devices.require("rtc"), devices.require("vna")
    original = rtc.get_progress
    called = False

    async def wrong_count():
        result = await original()
        if result["completed_groups"]:
            result["completed_points"] -= 1
        return result

    async def forbidden_read(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("Must reject RTC counters before VNA read")

    rtc.get_progress = wrong_count
    vna.read_buffered_acquisition = forbidden_read
    await engine.start(record.run_id)
    await record.task
    assert record.state.value in {"UNKNOWN", "FAULTED"}
    assert record.completed == 0
    assert not called
    with h5py.File(record.output_path, "r") as file:
        assert file.attrs["status"] != "COMPLETED"
        assert file["spatial_points/bundle_id"].shape == (0,)


@pytest.mark.asyncio
async def test_rtc_tr_change_invalidates_prepared_plan(tmp_path, rtc_setup):
    engine, devices, profile, coordinates = rtc_setup
    record = await engine.prepare(plan_for(tmp_path, profile, coordinates))
    await devices.command("rtc", "configure_tr", {"mode": "TX", "period_us": 120, "high_us": 20, "delay_us": 1})
    with pytest.raises(ServiceError, match="重新准备"):
        await engine.start(record.run_id)
    assert record.task is None
    assert devices.control_owner is None


@pytest.mark.asyncio
async def test_continuous_dense_pulses_fault_instead_of_queuing(tmp_path, rtc_setup):
    engine, devices, profile, coordinates = rtc_setup
    record = await engine.prepare(plan_for(tmp_path, profile, coordinates, topology="RTC_CONTINUOUS",
                                           azimuth_stop_deg=0.0002, azimuth_step_deg=0.0001))
    await engine.start(record.run_id)
    await record.task
    assert record.state.value in {"UNKNOWN", "FAULTED"}
    assert record.completed == 0
    assert (await devices.require("rtc", ready=False).get_status())["fault_code"] == 0x13
    assert all(v == 0 for v in devices.require("turntable", ready=False).velocities.values())


def test_rtc_continuous_rejects_single_or_uneven_row_and_deferred_calibration(tmp_path, loaded_assets):
    _, profile, coordinates = loaded_assets
    for changes in ({"azimuth_stop_deg": 0}, {"azimuth_stop_deg": 1, "azimuth_step_deg": 0.3},
                    {"test_type": "CALIBRATION"}):
        with pytest.raises(ValidationError):
            plan_for(tmp_path, profile, coordinates, topology="RTC_CONTINUOUS", **changes)


@pytest.mark.asyncio
async def test_cleanup_unknown_stop_only_queries_and_does_not_resend():
    calls = []
    client = SimpleNamespace(unknown_result={"opcode": 9})
    client.acknowledge_unknown_result = lambda: setattr(client, "unknown_result", None)
    states = iter([
        {"state": "STOPPING", "stage": 10, "tr_running": True},
        {"state": "COMPLETE", "stage": 0, "tr_running": False, "last_result": 2},
    ])

    async def status():
        return next(states)

    async def stop():
        calls.append("STOP")

    rtc = SimpleNamespace(client=client, get_status=status, stop_graceful=stop, stop_immediate=stop)
    session = RtcAcquisition(rtc, None, None, SimpleNamespace(test_type="PATTERN"), {"timing": DEFAULT_TIMING}, np.array([8e9]), [])
    await session.suspend()
    assert calls == []
    assert client.unknown_result is None


def test_arm_preparation_state_is_not_ready_even_with_zero_counters():
    status = {"state": "ARMED", "stage": 1, "tr_running": False, "io_busy": True}
    progress = {"accepted_groups": 0, "completed_groups": 0, "valid_triggers": 0, "completed_points": 0}
    assert not RtcAcquisition._arm_ready(status, progress)



@pytest.mark.asyncio
async def test_rtc_debug_api_tx_rx_and_tr_are_independent(rtc_setup, monkeypatch):
    import httpx
    from antenna_service import api as api_module

    engine, devices, profile, coordinates = rtc_setup
    for name, value in (("assets", engine.assets), ("devices", devices), ("events", engine.events), ("runs", engine)):
        monkeypatch.setattr(api_module, name, value)
    queue = engine.events.subscribe()
    command = profile.command_for_role("INITIALIZE_QUERY")
    frame = profile.encode(command.command_id, 0, {})
    rtc = devices.require("rtc")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_module.app), base_url="http://test") as http:
        response = await http.post("/api/devices/beam_controller/send", json={
            "profile_id": profile.asset_id, "command_id": command.command_id,
            "array_id": 0, "parameters": {}, "transport": "RTC",
        })
        assert response.status_code == 200, response.text
        assert response.json()["written"]
        assert len(bytes.fromhex(response.json()["rtc_frame_hex"])) == 32
        assert rtc.client.firmware.tx_frames == [frame]
        assert (await rtc.get_antenna_io_status())["rx_frames"] == 0
        rtc.client.inject_antenna_rx(frame)
        for _ in range(50):
            await asyncio.sleep(0.01)
            if (await rtc.get_antenna_io_status())["rx_frames"]:
                break
        received = []
        while not queue.empty():
            received.append(queue.get_nowait())
        rx = [event for event in received if event["type"] == "device.raw"
              and event.get("transport") == "RTC_ANTENNA" and event.get("direction") == "RX"]
        assert len(rx) == 1 and rx[0]["raw_hex"] == frame.hex(" ").upper()
        response = await http.post("/api/devices/rtc/command", json={
            "action": "configure_tr", "parameters": {"mode": "RX", "period_us": 100, "high_us": 20, "delay_us": 1},
        })
        assert response.status_code == 200, response.text
        assert response.json()["tr_state"] == 0
        response = await http.post("/api/devices/rtc/command", json={"action": "start_debug_tr", "parameters": {}})
        assert response.status_code == 200, response.text
        assert response.json()["tr_state"] == 2
        response = await http.post("/api/devices/rtc/command", json={"action": "stop_debug_tr", "parameters": {}})
        assert response.status_code == 200, response.text
        assert response.json()["tr_state"] == 0
    assert rtc.client.firmware.tx_frames == [frame]
    assert devices.require("vna").sample_counter == 0


@pytest.mark.asyncio
async def test_rtc_reads_remain_available_but_writes_locked_during_run(rtc_setup):
    _, devices, _, _ = rtc_setup
    await devices.acquire_run_control("test")
    try:
        assert (await devices.command("rtc", "get_status", {}))["state"] == "IDLE"
        with pytest.raises(ServiceError, match="占用"):
            await devices.command("rtc", "configure_tr", {"mode": "TX", "period_us": 100, "high_us": 20})
    finally:
        devices.release_run_control("test")


@pytest.mark.asyncio
async def test_unrelated_beam_connection_does_not_invalidate_rtc_plan(tmp_path, rtc_setup):
    engine, devices, profile, coordinates = rtc_setup
    record = await engine.prepare(plan_for(tmp_path, profile, coordinates, azimuth_stop_deg=0))
    await devices.connect("beam_controller", DeviceSource.SIMULATED, {})
    await engine.start(record.run_id)
    await record.task
    assert record.state.value == "COMPLETED", record.error
    assert devices.require("beam_controller").last_frame is None


@pytest.mark.asyncio
async def test_rtc_continuous_stop_finishes_current_row_and_does_not_return(tmp_path, rtc_setup):
    engine, devices, profile, coordinates = rtc_setup
    record = await engine.prepare(plan_for(tmp_path, profile, coordinates, topology="RTC_CONTINUOUS",
                                          elevation_stop_deg=1))
    vna = devices.require("vna")
    original = vna.read_buffered_acquisition

    async def stop_when_reading_row(*args, **kwargs):
        await engine.stop(record.run_id)
        return await original(*args, **kwargs)

    vna.read_buffered_acquisition = stop_when_reading_row
    await engine.start(record.run_id)
    await record.task
    assert record.state.value == "STOPPED", record.error
    assert record.completed == 6
    assert devices.require("turntable").positions[1] == 2
    with h5py.File(record.output_path, "r") as file:
        assert file.attrs["status"] == "STOPPED"
        assert file["rtc/acquisitions/point_count"][...].tolist() == [3]
    assert (await devices.require("rtc").get_status())["state"] == "COMPLETE"

