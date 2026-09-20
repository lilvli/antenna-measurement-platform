from __future__ import annotations

import asyncio
from dataclasses import replace

import h5py
import httpx
import pytest

from antenna_service import api as api_module
from antenna_service.devices.manager import DeviceManager
from antenna_service.devices.real import RealTurntable
from antenna_service.errors import ServiceError
from antenna_service.events import EventBus
from antenna_service.models import BeamDefinition, DeviceSource, DeviceState, RunPlan
from antenna_service.protocol.crc import append_crc_be
from antenna_service.protocol.profile import ResponseAssembly, ResponseField, ResponseRule
from antenna_service.workflows.engine import RunEngine


async def setup_run(loaded_assets, tmp_path, **changes):
    assets, profile, coordinates = loaded_assets
    events = EventBus()
    devices = DeviceManager(events)
    for name in ("beam_controller", "vna", "turntable"):
        await devices.connect(name, DeviceSource.SIMULATED, {})
    engine = RunEngine(assets, devices, events)
    options = dict(test_type="PATTERN", profile_id=profile.asset_id,
                   coordinate_id=coordinates.asset_id, output_directory=str(tmp_path),
                   azimuth_start_deg=0, azimuth_stop_deg=1, settle_ms=0)
    record = await engine.prepare(RunPlan(**(options | changes)))
    return engine, devices, record


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["CALIBRATION", "PATTERN"])
async def test_measurement_fault_persists_terminal_file_and_releases_handle(loaded_assets, tmp_path, kind):
    engine, devices, record = await setup_run(loaded_assets, tmp_path, test_type=kind)
    vna = devices.require("vna")
    acquire = vna.acquire
    async def fail_second(*args, **kwargs):
        if vna.sample_counter:
            raise ServiceError("DEVICE_FAULT", "injected failure", "test")
        return await acquire(*args, **kwargs)
    vna.acquire = fail_second
    await engine.start(record.run_id)
    await record.task
    assert record.state.value == "FAULTED"
    assert record.completed == 1
    assert record.result["status"] == "FAULTED"
    assert record.store is None and devices.control_owner is None
    with h5py.File(record.output_path, "r+") as file:
        assert file.attrs["status"] == "FAULTED"
        assert file.attrs["file_role"] == "RUN_TERMINAL"


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["pause", "stop"])
@pytest.mark.parametrize("beam_count", [1, 3])
async def test_controls_apply_between_beams_including_last_bundle(loaded_assets, tmp_path, action, beam_count):
    engine, devices, record = await setup_run(
        loaded_assets, tmp_path, azimuth_stop_deg=0,
        beams=[BeamDefinition(beam_id=str(n)) for n in range(beam_count)],
    )
    vna = devices.require("vna")
    acquire = vna.acquire
    async def request_after_first(*args, **kwargs):
        result = await acquire(*args, **kwargs)
        if vna.sample_counter == 1:
            await getattr(engine, action)(record.run_id)
        return result
    vna.acquire = request_after_first
    await engine.start(record.run_id)
    if action == "pause":
        for _ in range(100):
            if record.completed == 1:
                break
            await asyncio.sleep(.005)
        await asyncio.sleep(.02)
        assert record.state.value == "PAUSED" and not record.task.done()
        assert vna.sample_counter == 1
        await engine.stop(record.run_id)
    await asyncio.wait_for(record.task, 3)
    assert record.completed == 1 and record.state.value == "STOPPED"
    assert "post_completion_azimuth_home" not in record.result


@pytest.mark.asyncio
async def test_open_ack_timeout_still_sends_one_all_off(loaded_assets, tmp_path):
    engine, devices, record = await setup_run(loaded_assets, tmp_path, test_type="CALIBRATION")
    _, profile, coordinates = loaded_assets
    beam = devices.require("beam_controller")
    frames = []
    async def lost_open_ack(frame, **kwargs):
        frames.append(frame)
        if len(frames) == 1:
            raise ServiceError("TIMEOUT", "opening ACK lost", side_effect_possible=True)
        return frame
    beam.send_frame = lost_open_ack
    await engine.start(record.run_id)
    await record.task
    channel = coordinates.for_polarization("H")[0]
    expected_close = profile.build_calibration_frame(array_id=0, spi_no=channel.spi_no, chip_no=channel.chip_no,
        chip_channel_index=channel.chip_channel_index, signal_path="TX", enabled=False)
    assert len(frames) == 2 and frames[1] == expected_close
    assert record.state.value == "UNKNOWN" and record.result["status"] == "UNKNOWN"


@pytest.mark.asyncio
async def test_current_snapshot_recovers_all_samples_and_finished_state(loaded_assets, tmp_path):
    engine, devices, record = await setup_run(loaded_assets, tmp_path)
    await engine.start(record.run_id)
    await record.task
    snapshot = engine.snapshot()
    assert snapshot["run"]["state"] == "COMPLETED"
    assert snapshot["run"]["cleanup_pending"] is False
    assert len(snapshot["samples"]) == 2
    assert all(item["sequence"] <= snapshot["event_sequence"] for item in snapshot["samples"])
    assert snapshot["control_owner"] is None


@pytest.mark.asyncio
async def test_cancelled_post_completion_home_preserves_verified_data(loaded_assets, tmp_path):
    engine, devices, record = await setup_run(loaded_assets, tmp_path, azimuth_stop_deg=0)
    async def cancel_home(axis):
        raise asyncio.CancelledError()
    devices.require("turntable").home = cancel_home
    await engine.start(record.run_id)
    await record.task
    assert record.state.value == record.result["status"] == "COMPLETED"
    assert record.result["post_completion_azimuth_home"]["status"] == "FAILED"
    assert record.cleanup_pending is False
    with h5py.File(record.output_path, "r") as file:
        assert file.attrs["status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_shutdown_waits_for_control_then_disconnects_and_cancel_unfreezes(loaded_assets, tmp_path, monkeypatch):
    engine, devices, record = await setup_run(loaded_assets, tmp_path)
    monkeypatch.setattr(api_module, "runs", engine)
    monkeypatch.setattr(api_module, "devices", devices)
    monkeypatch.setattr(api_module, "events", engine.events)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_module.app), base_url="http://test") as client:
        async with devices.manual_control("flash_write"):
            response = await client.post("/api/shutdown/prepare")
            assert response.json()["ready"] is False
            assert devices.shutdown_requested
        with pytest.raises(ServiceError, match="正在停止任务"):
            await devices.connect("vna", DeviceSource.SIMULATED, {})
        response = await client.post("/api/shutdown/prepare")
        assert response.json()["ready"] is True
        assert all(item.snapshot.state.value == "DISCONNECTED" for item in devices.devices.values())
        await client.post("/api/shutdown/cancel")
        assert not devices.shutdown_requested


def multi_frame(index, total=2, value=7, array=3):
    return append_crc_be(bytes([0xAA, 0x55, 0, 22, 0x1A, array, 0, total, index, value]) + bytes(10))


@pytest.mark.asyncio
async def test_shutdown_allows_disconnecting_failed_connection_without_sending_stop(loaded_assets, tmp_path, monkeypatch):
    engine, devices, _ = await setup_run(loaded_assets, tmp_path)
    failed = RealTurntable(str(tmp_path / "missing.dll"))
    failed.snapshot.update(state=DeviceState.FAULT)
    devices.devices["turntable"] = failed
    monkeypatch.setattr(api_module, "runs", engine)
    monkeypatch.setattr(api_module, "devices", devices)
    monkeypatch.setattr(api_module, "events", engine.events)
    result = await api_module.prepare_shutdown()
    assert result["ready"] is True
    assert failed.snapshot.state == DeviceState.DISCONNECTED


def test_multiframe_assembly_complete_missing_duplicate_order_and_identity(loaded_assets):
    _, profile, _ = loaded_assets
    rule = ResponseRule("MULTI", "QUERY", 0x1A, "MULTI", 0, 2, 3, 4, 3, True, 1500)
    profile.response_rules[0x1A] = rule
    profile.response_fields["MULTI"] = [ResponseField("MULTI", n, "reading", "读数", 1, 1, "uint8", 1, 0, None, None, None) for n in (1, 2)]
    decoded = profile.decode_response([multi_frame(1, value=7), multi_frame(2, value=9)])
    assert decoded["fields"]["1:reading"]["value"] == 7
    assert decoded["fields"]["2:reading"]["value"] == 9
    for frames in ([multi_frame(1)], [multi_frame(1), multi_frame(1)],
                   [multi_frame(2), multi_frame(1)], [multi_frame(1), multi_frame(2, array=4)]):
        with pytest.raises(ServiceError):
            profile.decode_response(frames)
    assembly = ResponseAssembly(replace(rule, ordered=False))
    assert not assembly.add(multi_frame(2))
    assert assembly.add(multi_frame(1))
    assert assembly.frames == [multi_frame(1), multi_frame(2)]


def test_field_equals_compares_configured_wire_not_hardcoded_enum_label(loaded_assets):
    _, profile, _ = loaded_assets
    original = profile.command_for_role("INITIALIZE_QUERY")
    command = replace(original, success_value=b"\x01")
    request = profile.encode(command.command_id, 3, {})
    response = bytearray(request[:-2])
    response[7] = 1
    profile.validate_response(command, request, [append_crc_be(response)])
    response[7] = 0
    with pytest.raises(ServiceError, match="成功条件"):
        profile.validate_response(command, request, [append_crc_be(response)])


@pytest.mark.asyncio
async def test_slow_event_consumer_is_forced_to_resynchronize():
    events = EventBus()
    queue = events.subscribe()
    for _ in range(501):
        await events.publish("run.status", run={"state": "RUNNING"})
    assert queue.get_nowait()["type"] == "stream.resync"
    assert queue.get_nowait()["sequence"] == 501
