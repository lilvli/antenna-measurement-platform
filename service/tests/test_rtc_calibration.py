from __future__ import annotations

import asyncio
from dataclasses import replace

import h5py
import httpx
import numpy as np
import pytest
import pytest_asyncio

from antenna_service.devices.manager import DeviceManager
from antenna_service.events import EventBus
from antenna_service.errors import ServiceError
from antenna_service.models import DeviceSource, RunPlan, RtcWaveRequest
from antenna_service.storage.hdf5_store import inspect_data_file
from antenna_service.workflows.engine import RunEngine
from antenna_service.workflows.rtc import compile_rtc_wave_table


@pytest_asyncio.fixture
async def calibration_setup(loaded_assets):
    assets, profile, coordinates = loaded_assets
    # Preserve coordinate numbering and exercise disabled rows plus a chip switch.
    coordinates.channels = [replace(channel, enabled=channel.element in {0, 1, 4}) for channel in coordinates.channels]
    events = EventBus()
    devices = DeviceManager(events, frame_decoder=assets.decode_matching_frame)
    for device_id in ("rtc", "vna"):
        await devices.connect(device_id, DeviceSource.SIMULATED, {})
    engine = RunEngine(assets, devices, events)
    yield engine, devices, profile, coordinates
    record = engine.current()
    if record and record.task and not record.task.done():
        record.task.cancel()
        await asyncio.gather(record.task, return_exceptions=True)
    for adapter in devices.devices.values():
        await adapter.disconnect()


def calibration_plan(tmp_path, profile, coordinates, **changes):
    arguments = dict(test_type="CALIBRATION", topology="RTC_STOP_AND_GO",
                     profile_id=profile.asset_id, coordinate_id=coordinates.asset_id,
                     output_directory=str(tmp_path), base_filename="rtc_cal",
                     frequency_start_hz=8e9, frequency_stop_hz=8.2e9, frequency_points=3, settle_ms=0)
    arguments.update(changes)
    return RunPlan(**arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize("signal_path", ["TX", "RX"])
async def test_rtc_channel_calibration_preloads_then_closes_each_channel(tmp_path, calibration_setup, signal_path):
    engine, devices, profile, coordinates = calibration_setup
    rtc, vna = devices.require("rtc"), devices.require("vna")
    queue = engine.events.subscribe()
    record = await engine.prepare(calibration_plan(tmp_path, profile, coordinates, signal_path=signal_path))
    assert record.rtc_configuration["wave_count"] == 3
    assert record.rtc_configuration["waves_verified"]
    assert not rtc.client.firmware.tx_frames and vna.sample_counter == 0
    assert "turntable" not in devices.devices
    while not queue.empty():
        queue.get_nowait()
    await engine.start(record.run_id)
    await record.task
    assert record.state.value == "COMPLETED", record.error
    assert record.completed == len(coordinates.channels) == 256
    assert vna.sample_counter == 3
    emitted = []
    while not queue.empty():
        emitted.append(queue.get_nowait())
    opcodes = [bytes.fromhex(event["raw_hex"])[4] for event in emitted
               if event["type"] == "device.raw" and event.get("context") == "RTC_FRAME"
               and event.get("direction") == "TX"]
    assert 0x21 not in opcodes, "Start must not rewrite already verified waves"
    enabled = [channel for channel in coordinates.channels if channel.enabled]
    initial = list({(channel.spi_no, channel.chip_no): channel for channel in enabled}.values())
    expected = []
    for channel in initial:
        expected.append(profile.build_calibration_frame(array_id=0, spi_no=channel.spi_no,
                        chip_no=channel.chip_no, chip_channel_index=channel.chip_channel_index,
                        signal_path=signal_path, enabled=False))
    for channel in enabled:
        for opening in (True, False):
            expected.append(profile.build_calibration_frame(array_id=0, spi_no=channel.spi_no,
                            chip_no=channel.chip_no, chip_channel_index=channel.chip_channel_index,
                            signal_path=signal_path, enabled=opening))
    assert rtc.client.firmware.tx_frames == expected
    with h5py.File(record.output_path, "r") as file:
        group = file["rtc/channel_measurements"]
        flags = file["channels/enabled"][...]
        assert group["wave_address"][flags].tolist() == [1, 2, 3]
        assert group["completed_groups"][flags].tolist() == [1, 1, 1]
        assert group["completed_points"][flags].tolist() == [3, 3, 3]
        assert set(group["close_status"].asstr()[flags]) == {"CONFIRMED"}
        assert set(group["close_status"].asstr()[~flags]) == {"SKIPPED_DISABLED"}
        assert np.array_equal(group["close_request"][flags], group["close_response"][flags])
        assert np.all(np.isnan(file["measurements/real"][~flags]))
        assert file["rtc/wave_frames"].shape == (3, 22)
    assert inspect_data_file(record.output_path)["checksum_verified"]


@pytest.mark.asyncio
async def test_close_without_matching_e2_stops_next_channel_and_preserves_data(tmp_path, calibration_setup):
    engine, devices, profile, coordinates = calibration_setup
    rtc = devices.require("rtc")
    original = rtc.queue_antenna_response
    calls = 0

    def missing_after_initial(request, responses):
        nonlocal calls
        calls += 1
        if calls <= 2:  # Two participating chips have an initial close.
            original(request, responses)

    rtc.queue_antenna_response = missing_after_initial
    record = await engine.prepare(calibration_plan(tmp_path, profile, coordinates))
    await engine.start(record.run_id)
    await record.task
    assert record.state.value == "UNKNOWN"
    assert devices.require("vna").sample_counter == 1
    assert calls == 3, "No close resend and no next-channel request"
    with h5py.File(record.output_path, "r") as file:
        assert file["channels/result_status"].asstr()[0] == "FAILED"
        assert file["channels/error_code"].asstr()[0] == "CLOSE_UNKNOWN"
        assert file["rtc/channel_measurements/close_status"].asstr()[0] == "UNKNOWN"
        assert file["channels/result_status"].asstr()[1] == "PENDING"
        assert np.all(np.isfinite(file["measurements/real"][0]))
        assert file.attrs["status"] == "UNKNOWN"


@pytest.mark.asyncio
async def test_calibration_pause_occurs_after_close_confirmation(tmp_path, calibration_setup):
    engine, devices, profile, coordinates = calibration_setup
    vna = devices.require("vna")
    record = await engine.prepare(calibration_plan(tmp_path, profile, coordinates))
    original = vna.read_buffered_acquisition
    paused = False

    async def pause_read(*args, **kwargs):
        nonlocal paused
        if not paused:
            paused = True
            await engine.pause(record.run_id)
        return await original(*args, **kwargs)

    vna.read_buffered_acquisition = pause_read
    await engine.start(record.run_id)
    for _ in range(250):
        if record.state.value == "PAUSED" or record.task.done():
            break
        await asyncio.sleep(0.02)
    assert record.state.value == "PAUSED", record.error
    assert vna.sample_counter == 1
    with h5py.File(record.output_path, "r") as file:
        assert file["rtc/channel_measurements/close_status"].asstr()[0] == "CONFIRMED"
    await engine.stop(record.run_id)
    await record.task
    assert record.state.value == "STOPPED"
    assert vna.sample_counter == 1


@pytest.mark.asyncio
async def test_calibration_rejects_weak_close_success_rule(tmp_path, calibration_setup):
    engine, _, profile, coordinates = calibration_setup
    command = profile.command_for_role("CALIBRATION_WRITE")
    profile.commands[command.command_id] = replace(command, success_rule="VALID_RESPONSE")
    with pytest.raises(ServiceError, match="FRAME_EQUALS_REQUEST"):
        await engine.prepare(calibration_plan(tmp_path, profile, coordinates))


@pytest.mark.asyncio
async def test_changed_rtc_wave_prevents_start_without_rewriting(tmp_path, calibration_setup):
    engine, devices, profile, coordinates = calibration_setup
    record = await engine.prepare(calibration_plan(tmp_path, profile, coordinates))
    rtc = devices.require("rtc")
    wrong = bytes.fromhex(record.rtc_wave_entries[1]["frame_hex"])
    await rtc.write_wave_entry(1, wrong)
    with pytest.raises(ServiceError, match="重新写入"):
        await engine.start(record.run_id)
    assert record.task is None
    assert await rtc.read_wave_entry(1) == wrong
    assert not rtc.client.firmware.tx_frames


@pytest.mark.asyncio
async def test_independent_wave_preload_needs_no_vna_turntable_or_output(loaded_assets, monkeypatch):
    from antenna_service import api as api_module
    assets, profile, coordinates = loaded_assets
    events = EventBus()
    devices = DeviceManager(events)
    monkeypatch.setattr(api_module, "assets", assets)
    monkeypatch.setattr(api_module, "devices", devices)
    payload = {"profile_id": profile.asset_id, "test_type": "PATTERN", "array_id": 7,
               "reference_frequency_hz": 8.1e9, "beams": [
                   {"beam_id": "A", "off_axis_deg": 0, "azimuth_deg": 0},
                   {"beam_id": "B", "off_axis_deg": 10, "azimuth_deg": 20}]}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_module.app), base_url="http://test") as http:
        preview = await http.post("/api/devices/rtc/waves/preview", json=payload)
        assert preview.status_code == 200
        assert preview.json()["count"] == 2 and devices.devices == {}
        await devices.connect("rtc", DeviceSource.SIMULATED, {})
        try:
            empty = await http.post("/api/devices/rtc/waves/read", json=payload)
            assert empty.status_code == 200, empty.text
            assert all(item["status"] == "EMPTY" for item in empty.json()["entries"])
            first = await http.post("/api/devices/rtc/waves/write", json=payload)
            assert first.status_code == 200, first.text
            assert first.json()["verified"] and first.json()["written_count"] == 2
            second = await http.post("/api/devices/rtc/waves/write", json=payload)
            assert second.status_code == 200
            assert second.json()["written_count"] == 0
            actual = await http.post("/api/devices/rtc/waves/read", json=payload)
            assert actual.json()["verified"]
            assert all(entry["status"] == "MATCH" for entry in actual.json()["entries"])
            assert not devices.require("rtc").client.firmware.tx_frames
            assert (await devices.require("rtc").get_status())["tr_running"] is False
        finally:
            await devices.require("rtc", ready=False).disconnect()


def test_calibration_wave_preview_auto_assigns_enabled_channels(loaded_assets):
    assets, profile, coordinates = loaded_assets
    coordinates.channels = [replace(channel, enabled=channel.element in {1, 4, 255}) for channel in coordinates.channels]
    entries = compile_rtc_wave_table(assets, RtcWaveRequest(
        profile_id=profile.asset_id, coordinate_id=coordinates.asset_id, test_type="CALIBRATION", signal_path="RX"))
    assert [(entry["address"], entry["element"]) for entry in entries] == [(1, 1), (2, 4), (3, 255)]
    assert all(len(bytes.fromhex(entry["frame_hex"])) == 22 for entry in entries)



@pytest.mark.asyncio
async def test_rtc_fault_does_not_clear_fault_or_send_forbidden_close(tmp_path, calibration_setup):
    engine, devices, profile, coordinates = calibration_setup
    rtc, vna = devices.require("rtc"), devices.require("vna")
    record = await engine.prepare(calibration_plan(tmp_path, profile, coordinates))
    async def fail_after_point(*args, **kwargs):
        rtc.client.firmware._fault(0x0B, 8)
        raise ServiceError("DEVICE_FAULT", "矢网读取失败", "vna_buffer_read", side_effect_possible=True)
    vna.read_buffered_acquisition = fail_after_point
    await engine.start(record.run_id)
    await record.task
    assert record.state.value == "UNKNOWN"
    assert (await rtc.get_status())["state"] == "FAULT"
    assert (await rtc.get_status())["fault_code"] == 0x0B
    assert len(rtc.client.firmware.tx_frames) == 3  # initial chip closes + first opening only
    with h5py.File(record.output_path, "r") as file:
        assert file["rtc/channel_measurements/close_status"].asstr()[0] == "UNKNOWN"
        assert file["channels/result_status"].asstr()[1] == "PENDING"


@pytest.mark.asyncio
async def test_full_256_channel_rtc_calibration_needs_no_turntable(tmp_path, loaded_assets):
    assets, profile, coordinates = loaded_assets
    events = EventBus()
    devices = DeviceManager(events)
    for device in ("rtc", "vna"):
        await devices.connect(device, DeviceSource.SIMULATED, {})
    engine = RunEngine(assets, devices, events)
    try:
        plan = calibration_plan(tmp_path, profile, coordinates, frequency_stop_hz=8e9, frequency_points=1)
        record = await engine.prepare(plan)
        assert record.rtc_configuration["wave_count"] == 256
        await engine.start(record.run_id)
        await record.task
        assert record.state.value == "COMPLETED", record.error
        assert record.completed == 256
        assert devices.require("vna").sample_counter == 256
        with h5py.File(record.output_path, "r") as file:
            assert file["rtc/wave_frames"].shape == (256, 22)
            assert file["measurements/real"].shape == (256, 1)
            assert set(file["rtc/channel_measurements/close_status"].asstr()[...]) == {"CONFIRMED"}
    finally:
        for adapter in devices.devices.values():
            await adapter.disconnect()

