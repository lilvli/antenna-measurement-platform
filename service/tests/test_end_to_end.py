from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from antenna_service.devices.manager import DeviceManager
from antenna_service.errors import ServiceError
from antenna_service.events import EventBus
from antenna_service.models import (
    BeamDefinition,
    CompensationRequest,
    DeviceSource,
    FlashPrepareRequest,
    FlashWriteRequest,
    RunPlan,
)
from antenna_service.storage.hdf5_store import inspect_data_file, read_data_view
from antenna_service.workflows.compensation import CompensationService
from antenna_service.workflows.engine import RunEngine
from antenna_service.workflows.flash import FlashService


async def _run_calibration(engine, profile, coordinates, output: Path, signal_path: str) -> str:
    plan = RunPlan(
        name=f"{signal_path}-H",
        test_type="CALIBRATION",
        signal_path=signal_path,
        polarization="H",
        profile_id=profile.asset_id,
        coordinate_id=coordinates.asset_id,
        output_directory=str(output),
        base_filename=f"{signal_path.lower()}_h",
        frequency_start_hz=8e9,
        frequency_stop_hz=10e9,
        frequency_points=3,
        source_power_dbm=-17.5,
        averaging_enabled=True,
        averaging_count=3,
        settle_ms=0,
    )
    record = await engine.prepare(plan)
    await engine.start(record.run_id)
    assert record.task is not None
    await record.task
    assert record.state.value == "COMPLETED"
    assert record.completed == 256
    assert record.output_path
    assert Path(record.output_path).suffix == ".hdf5"
    assert "_标校_" in Path(record.output_path).name
    assert h5py.is_hdf5(record.output_path)
    return record.output_path


@pytest.mark.asyncio
async def test_simulated_pattern_uses_vna_beam_turntable_and_hdf5(tmp_path, loaded_assets):
    registry, profile, coordinates = loaded_assets
    events = EventBus()
    queue = events.subscribe()
    devices = DeviceManager(events)
    await devices.connect("beam_controller", DeviceSource.SIMULATED, {})
    await devices.connect("vna", DeviceSource.SIMULATED, {})
    await devices.connect("turntable", DeviceSource.SIMULATED, {})
    engine = RunEngine(registry, devices, events)
    plan = RunPlan(
        test_type="PATTERN",
        profile_id=profile.asset_id,
        coordinate_id=coordinates.asset_id,
        output_directory=str(tmp_path),
        base_filename="beam_scan",
        frequency_start_hz=8e9,
        frequency_stop_hz=8.2e9,
        frequency_points=3,
        source_power_dbm=-17.5,
        averaging_enabled=True,
        averaging_count=3,
        azimuth_start_deg=0,
        azimuth_stop_deg=0,
        azimuth_step_deg=1,
        elevation_start_deg=0,
        elevation_stop_deg=1,
        elevation_step_deg=1,
        beams=[
            BeamDefinition(beam_id="BEAM-1", off_axis_deg=0, azimuth_deg=0),
            BeamDefinition(beam_id="BEAM-2", off_axis_deg=10, azimuth_deg=30),
        ],
        settle_ms=0,
    )
    record = await engine.prepare(plan)
    assert record.total == 4
    await engine.start(record.run_id)
    assert record.task is not None
    await record.task
    assert record.state.value == "COMPLETED"
    assert record.output_path and "_方向图_" in Path(record.output_path).name
    assert h5py.is_hdf5(record.output_path)
    with h5py.File(record.output_path, "r") as file:
        assert file.attrs["status"] == "COMPLETED"
        assert file.attrs["schema_version"] == "3.1"
        assert file["spatial_points/point_id"].shape == (4,)
        assert file["measurements/real"].shape == (4, 3)
        assert file["beam_definitions/beam_id"].asstr()[0] == "BEAM-1"
        assert file["spatial_points/beam_id"].asstr()[0] == "BEAM-1"
        assert file["spatial_points/actual_azimuth_deg"].shape == (4,)
        assert file["spatial_points/azimuth_deg"][...].tolist() == [0.0, 0.0, 0.0, 0.0]
        assert file["spatial_points/elevation_deg"][...].tolist() == [0.0, 0.0, 1.0, 1.0]
    queued = []
    while not queue.empty():
        queued.append(queue.get_nowait())
    samples = [event for event in queued if event["type"] == "run.sample"]
    assert len(samples) == 4
    assert all(len(event["magnitudes_db"]) == 3 for event in samples)
    # One BEAM_SET and one complete three-point VNA sweep are used per mechanical point.
    assert len([event for event in queued if event["type"] == "device.raw" and event["direction"] == "TX"]) == 4
    simulated_vna = devices.require("vna")
    assert simulated_vna.sample_counter == 4
    assert simulated_vna.settings["points"] == 3
    assert simulated_vna.settings["source_power_dbm"] == -17.5
    assert simulated_vna.settings["averaging_enabled"] is True
    assert simulated_vna.settings["averaging_count"] == 3
    simulated_beam = devices.require("beam_controller")
    assert simulated_beam.last_frame is not None
    assert simulated_beam.last_frame[13] == 0xFF
    inspected = inspect_data_file(record.output_path)
    assert inspected["analysis"]["kind"] == "PATTERN"
    assert len(inspected["analysis"]["frequencies_hz"]) == 3
    history_view = read_data_view(record.output_path, frequency_index=1, beam_index=1)
    assert history_view["frequency_hz"] == pytest.approx(8.1e9)
    assert [(point["elevation_deg"], point["azimuth_deg"]) for point in history_view["points"]] == [(0.0, 0.0), (1.0, 0.0)]
    assert history_view["beam"]["beam_id"] == "BEAM-2"
    all_directions_view = read_data_view(record.output_path, frequency_index=1, beam_index=-1)
    assert len(all_directions_view["points"]) == 4
    assert {point["beam_id"] for point in all_directions_view["points"]} == {"BEAM-1", "BEAM-2"}


@pytest.mark.asyncio
async def test_pattern_is_completed_before_only_azimuth_is_homed(tmp_path, loaded_assets):
    registry, profile, coordinates = loaded_assets
    events = EventBus()
    devices = DeviceManager(events)
    await devices.connect("beam_controller", DeviceSource.SIMULATED, {})
    await devices.connect("vna", DeviceSource.SIMULATED, {})
    await devices.connect("turntable", DeviceSource.SIMULATED, {})
    engine = RunEngine(registry, devices, events)
    plan = RunPlan(
        test_type="PATTERN",
        profile_id=profile.asset_id,
        coordinate_id=coordinates.asset_id,
        output_directory=str(tmp_path),
        base_filename="post_complete_home",
        azimuth_start_deg=-1,
        azimuth_stop_deg=-1,
        elevation_start_deg=-2,
        elevation_stop_deg=-2,
        settle_ms=0,
    )
    record = await engine.prepare(plan)
    turntable = devices.require("turntable")
    original_home = turntable.home

    async def verify_completed_then_home(axis: int):
        assert record.state.value == "COMPLETED"
        assert record.result is not None
        assert record.output_path is not None
        with h5py.File(record.output_path, "r") as file:
            assert file.attrs["status"] == "COMPLETED"
        return await original_home(axis)

    turntable.home = verify_completed_then_home
    await engine.start(record.run_id)
    assert record.task is not None
    await record.task

    assert record.state.value == "COMPLETED"
    assert record.result["post_completion_azimuth_home"]["status"] == "SUCCESS"
    assert turntable.positions[1] == 0.0
    assert turntable.positions[2] == -2.0


def test_vna_average_plan_requires_consistent_state_and_count(tmp_path, loaded_assets):
    _, profile, coordinates = loaded_assets
    common = {
        "test_type": "CALIBRATION",
        "profile_id": profile.asset_id,
        "coordinate_id": coordinates.asset_id,
        "output_directory": str(tmp_path),
    }
    with pytest.raises(ValueError, match="至少为 2"):
        RunPlan(averaging_enabled=True, averaging_count=1, **common)
    with pytest.raises(ValueError, match="必须为 1"):
        RunPlan(averaging_enabled=False, averaging_count=2, **common)


@pytest.mark.asyncio
async def test_external_fixed_pattern_ignores_beam_connection_and_only_acquires(tmp_path, loaded_assets):
    registry, profile, coordinates = loaded_assets
    events = EventBus()
    queue = events.subscribe()
    devices = DeviceManager(events)
    await devices.connect("vna", DeviceSource.SIMULATED, {})
    await devices.connect("turntable", DeviceSource.SIMULATED, {})
    engine = RunEngine(registry, devices, events)
    plan = RunPlan(
        test_type="PATTERN",
        beam_control_mode="EXTERNAL_FIXED",
        profile_id=profile.asset_id,
        coordinate_id=coordinates.asset_id,
        output_directory=str(tmp_path),
        base_filename="external_fixed",
        frequency_start_hz=8e9,
        frequency_stop_hz=8.2e9,
        frequency_points=3,
        azimuth_start_deg=0,
        azimuth_stop_deg=1,
        azimuth_step_deg=1,
        elevation_start_deg=0,
        elevation_stop_deg=0,
        elevation_step_deg=1,
        beams=[BeamDefinition(beam_id="EXTERNAL-FIXED", off_axis_deg=12, azimuth_deg=35)],
        settle_ms=0,
    )

    # PREPARE succeeds with no beam-controller adapter at all.
    record = await engine.prepare(plan)
    assert record.total == 2

    # Connecting it afterwards must also be harmless: the automatic run leaves the
    # pre-existing debug command marker untouched and publishes no beam TX event.
    await devices.connect("beam_controller", DeviceSource.SIMULATED, {})
    beam = devices.require("beam_controller")
    beam.last_frame = b"debug-command-marker"
    await engine.start(record.run_id)
    assert record.task is not None
    await record.task

    assert record.state.value == "COMPLETED"
    assert record.completed == 2
    assert beam.last_frame == b"debug-command-marker"
    assert devices.require("vna").sample_counter == 2
    queued = []
    while not queue.empty():
        queued.append(queue.get_nowait())
    assert not [event for event in queued if event["type"] == "device.raw" and event.get("direction") == "TX"]

    assert record.output_path
    inspected = inspect_data_file(record.output_path)
    assert inspected["metadata"]["beam_control_mode"] == "EXTERNAL_FIXED"
    assert inspected["metadata"]["beam_command_sent"] is False
    assert inspected["metadata"]["beam_response_verified"] is False
    assert inspected["metadata"]["beam_state_source"] == "USER_DECLARED_UNVERIFIED"
    assert set(inspected["metadata"]["device_sources"]) == {"vna", "turntable"}
    assert inspected["analysis"]["beams"] == [
        {
            "beam_index": 0,
            "beam_id": "EXTERNAL-FIXED",
            "off_axis_deg": 12.0,
            "azimuth_deg": 35.0,
        }
    ]


def test_external_fixed_mode_is_pattern_only_and_single_beam(tmp_path, loaded_assets):
    _, profile, coordinates = loaded_assets
    common = {
        "profile_id": profile.asset_id,
        "coordinate_id": coordinates.asset_id,
        "output_directory": str(tmp_path),
        "beam_control_mode": "EXTERNAL_FIXED",
    }
    with pytest.raises(ValueError, match="只用于方向图"):
        RunPlan(test_type="CALIBRATION", **common)
    with pytest.raises(ValueError, match="只能记录一个"):
        RunPlan(
            test_type="PATTERN",
            beams=[BeamDefinition(beam_id="A"), BeamDefinition(beam_id="B")],
            **common,
        )


@pytest.mark.asyncio
async def test_simulated_calibration_compensation_and_flash(tmp_path, loaded_assets):
    registry, profile, coordinates = loaded_assets
    events = EventBus()
    queue = events.subscribe()
    devices = DeviceManager(events)
    await devices.connect("beam_controller", DeviceSource.SIMULATED, {})
    await devices.connect("vna", DeviceSource.SIMULATED, {})
    engine = RunEngine(registry, devices, events)
    tx_cal = await _run_calibration(engine, profile, coordinates, tmp_path, "TX")
    rx_cal = await _run_calibration(engine, profile, coordinates, tmp_path, "RX")
    queued = []
    while not queue.empty():
        queued.append(queue.get_nowait())
    calibrated_samples = [event for event in queued if event["type"] == "run.sample" and event.get("status") == "CALIBRATED"]
    assert calibrated_samples
    assert all(len(event["magnitudes_db"]) == 3 and len(event["phases_deg"]) == 3 for event in calibrated_samples)
    last_channel = coordinates.for_polarization("H")[-1]
    expected_closed = profile.build_calibration_frame(
        array_id=0,
        spi_no=last_channel.spi_no,
        chip_no=last_channel.chip_no,
        chip_channel_index=last_channel.chip_channel_index,
        signal_path="RX",
        enabled=False,
    )
    assert devices.require("beam_controller").last_frame == expected_closed
    with h5py.File(tx_cal, "r") as file:
        assert file.attrs["schema_version"] == "3.2"
        assert set(file["channels/result_status"].asstr()[...]) <= {"CALIBRATED", "SKIPPED_DISABLED"}
    calibration_view = read_data_view(tx_cal, frequency_index=0)
    assert calibration_view["kind"] == "CALIBRATION"
    assert len(calibration_view["channels"]) == 256
    assert calibration_view["max_amplitude_difference_db"] > 0

    compensation = CompensationService()
    tx_comp = compensation.generate(
        CompensationRequest(
            signal_path="TX",
            coordinate_id=coordinates.asset_id,
            calibration_files=[tx_cal],
            frequency_indices=[2, 0],
            output_path=str(tmp_path / "tx_compensation.hdf5"),
        ),
        coordinates,
    )
    rx_comp = compensation.generate(
        CompensationRequest(
            signal_path="RX",
            coordinate_id=coordinates.asset_id,
            calibration_files=[rx_cal],
            frequency_indices=[2, 0],
            output_path=str(tmp_path / "rx_compensation.hdf5"),
        ),
        coordinates,
    )
    assert tx_comp["tables"]["compensation"] == 512
    assert h5py.is_hdf5(tx_comp["path"])
    with h5py.File(tx_comp["path"], "r") as file:
        assert file.attrs["schema_name"] == "antenna-compensation"
        assert file.attrs["schema_version"] == "2.1"
        assert file["frequencies_hz"][...].tolist() == [8e9, 10e9]
        assert file["provenance/source_frequency_index"][...].tolist() == [0, 2]
        assert file["compensation/phase_code"].shape == (256, 2)
        assert file["aperture_weighting"].attrs["algorithm"] == "NONE"
        assert np.allclose(file["aperture_weighting/linear_weight"][...], 1.0)
        assert np.allclose(file["compensation/calibration_attenuation_db"][...], file["compensation/attenuation_db"][...], atol=0.251)

    weighted_comp = compensation.generate(
        CompensationRequest(
            signal_path="TX",
            coordinate_id=coordinates.asset_id,
            calibration_files=[tx_cal],
            frequency_indices=[0],
            output_path=str(tmp_path / "tx_taylor_compensation.hdf5"),
            sidelobe_algorithm="TAYLOR",
            sidelobe_axes="BOTH",
            taylor_nbar=4,
            taylor_sll_db=30,
        ),
        coordinates,
    )
    assert weighted_comp["aperture_weighting"]["algorithm"] == "TAYLOR"
    with h5py.File(weighted_comp["path"], "r") as file:
        weights = file["aperture_weighting/linear_weight"][...]
        taper = file["aperture_weighting/taper_attenuation_db"][...]
        assert weights.shape == (256,)
        assert np.nanmax(weights) == pytest.approx(1.0)
        assert np.nanmax(taper) > 20.0
        assert np.all(file["compensation/attenuation_code"][...] <= 63)

    with pytest.raises(ServiceError, match="所需通道衰减超过硬件上限"):
        compensation.generate(
            CompensationRequest(
                signal_path="TX",
                coordinate_id=coordinates.asset_id,
                calibration_files=[tx_cal],
                frequency_indices=[0],
                output_path=str(tmp_path / "tx_hamming_compensation.hdf5"),
                sidelobe_algorithm="HAMMING",
                sidelobe_axes="BOTH",
            ),
            coordinates,
        )
    flash = FlashService()
    package = flash.prepare(
        FlashPrepareRequest(
            coordinate_id=coordinates.asset_id,
            tx_compensation_file=tx_comp["path"],
            rx_compensation_file=rx_comp["path"],
            output_path=str(tmp_path / "antenna_flash_package.hdf5"),
            array_id=1,
            start_addresses={
                "ARRAY_ID": 0x0000,
                "SWITCH_TABLE": 0x1000,
                "COORDINATE": 0x2000,
                "TX_COMPENSATION": 0x4000,
                "RX_COMPENSATION": 0x6000,
            },
        ),
        coordinates,
    )
    assert len(package["items"]) == 5
    assert h5py.is_hdf5(package["path"])
    with h5py.File(package["path"], "r") as file:
        assert file.attrs["schema_name"] == "antenna-flash-package"
        assert file.attrs["schema_version"] == "2.0"
        assert set(file["flash_items"].keys()) == {
            "ARRAY_ID", "SWITCH_TABLE", "COORDINATE", "TX_COMPENSATION", "RX_COMPENSATION"
        }
        assert file.attrs["tile_id"] == 1
        assert file.attrs["payload_layout"] == "FLASH_INTERNAL_OFFICIAL_V1"
        assert file["flash_items/SWITCH_TABLE/payload"].dtype == "uint8"
        array_item = file["flash_items/ARRAY_ID"]
        array_payload = array_item["payload"][...].tolist()
        assert array_item.attrs["effective_length"] == 1
        assert array_item.attrs["occupied_length"] == 256
        assert array_payload[0] == 1
        assert set(array_payload[1:]) == {0}
        switch_item = file["flash_items/SWITCH_TABLE"]
        assert switch_item.attrs["effective_length"] == 32
        assert switch_item.attrs["occupied_length"] == 256
        assert switch_item["payload"][:32].tolist() == [255] * 32
        ordered_coordinates = sorted(
            coordinates.channels,
            key=lambda channel: (channel.polarization, channel.chip_no, channel.chip_channel_index, channel.spi_no),
        )
        expected_coordinates = bytes(channel.grid_row for channel in ordered_coordinates) + bytes(
            channel.grid_column for channel in ordered_coordinates
        )
        coordinate_item = file["flash_items/COORDINATE"]
        assert coordinate_item.attrs["effective_length"] == len(expected_coordinates) == 512
        assert coordinate_item["payload"][...].tobytes() == expected_coordinates
        tx_flash_payload = file["flash_items/TX_COMPENSATION/payload"][...].tobytes()
        tx_effective_length = int(file["flash_items/TX_COMPENSATION"].attrs["effective_length"])
    with h5py.File(tx_comp["path"], "r") as source:
        source_order = sorted(
            range(source["channels/polarization"].shape[0]),
            key=lambda index: (
                source["channels/polarization"].asstr()[index],
                int(source["channels/chip_no"][index]),
                int(source["channels/chip_channel_index"][index]),
                int(source["channels/spi_no"][index]),
            ),
        )
        expected_tx = b"".join(
            bytes(int(source["compensation/phase_code"][index, frequency_index]) for index in source_order)
            + bytes(int(source["compensation/attenuation_code"][index, frequency_index]) for index in source_order)
            for frequency_index in (0, 1)
        )
    assert tx_effective_length == len(expected_tx) == 1024
    assert tx_flash_payload == expected_tx
    assert package["items"][1]["start_address"] == 0x1000
    assert package["items"][1]["end_address"] == 0x1000 + package["items"][1]["occupied_length"] - 1
    result = await flash.write_item(
        FlashWriteRequest(
            package_path=package["path"],
            item_name="SWITCH_TABLE",
            device_id="beam_controller",
        ),
        devices,
    )
    assert result["status"] == "SUCCESS"
