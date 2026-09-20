from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from openpyxl import load_workbook
from pydantic import ValidationError

from antenna_service.coordinates import CoordinateLoader
from antenna_service.errors import ServiceError
from antenna_service.models import CompensationRequest, FlashPrepareRequest, FlashWriteRequest
from antenna_service.storage.hdf5_store import MeasurementStore, inspect_data_file, read_data_view
from antenna_service.storage import hdf5_store
from antenna_service.workflows.compensation import CompensationService
from antenna_service.workflows.flash import FlashService


@pytest.fixture()
def small_coordinates(loaded_assets):
    coordinates = loaded_assets[2]
    channels = [replace(channel, grid_row=0, grid_column=index, enabled=index != 0)
                for index, channel in enumerate(coordinates.channels[:4])]
    return replace(coordinates, channels=channels)


def _store(path, coordinates, signal_path="TX"):
    metadata = {
        "signal_path": signal_path, "polarization": "H", "evidence": "SIMULATED",
        "coordinate_sha256": coordinates.file_sha256, "mapping_sha256": coordinates.mapping_sha256,
        "enabled_sha256": coordinates.enabled_sha256, "geometry_sha256": coordinates.geometry_sha256,
    }
    return MeasurementStore.create_calibration(path, run_id="integrity-test", metadata=metadata,
        frequencies_hz=np.array([8e9, 9e9]), coordinates=coordinates, polarization="H")


def _completed_calibration(path, coordinates, signal_path="TX"):
    store = _store(path, coordinates, signal_path)
    for channel in coordinates.channels:
        if channel.enabled:
            store.commit_channel(channel, np.array([1j, 1 + 1j]))
    store.finalize("COMPLETED")
    return path


def _compensation(path, calibration, coordinates, signal_path="TX"):
    request = CompensationRequest(signal_path=signal_path, coordinate_id=coordinates.asset_id,
        calibration_files=[str(calibration)], frequency_indices=[1, 0], output_path=str(path))
    return CompensationService().generate(request, coordinates)


@pytest.fixture()
def flash_package(tmp_path, small_coordinates):
    sources = {}
    for signal_path in ("TX", "RX"):
        calibration = _completed_calibration(tmp_path / f"{signal_path}-cal.hdf5", small_coordinates, signal_path)
        sources[signal_path] = _compensation(tmp_path / f"{signal_path}-comp.hdf5", calibration, small_coordinates, signal_path)["path"]
    result = FlashService().prepare(FlashPrepareRequest(coordinate_id=small_coordinates.asset_id,
        tx_compensation_file=sources["TX"], rx_compensation_file=sources["RX"],
        output_path=str(tmp_path / "flash.hdf5")), small_coordinates)
    return result


@pytest.mark.parametrize("cell,value", [("C9", 0.9), ("D9", -0.9), ("E9", True), ("F9", 0.5), ("G9", 0.9)])
def test_coordinate_integer_fields_reject_truncation(tmp_path, loaded_assets, cell, value):
    workbook = load_workbook(loaded_assets[2].path)
    workbook["通道坐标表"][cell] = value
    path = tmp_path / "bad-integer.xlsx"
    workbook.save(path)
    workbook.close()
    with pytest.raises(ServiceError, match="必须是整数"):
        CoordinateLoader().load(str(path))


@pytest.mark.parametrize("field,value", [("phase_step_deg", 90), ("attenuation_step_db", 1), ("max_attenuation_db", 63)])
def test_compensation_rejects_non_firmware_quantization(field, value):
    with pytest.raises(ValidationError):
        CompensationRequest(signal_path="TX", coordinate_id="coords", calibration_files=["cal.hdf5"],
            frequency_indices=[0], output_path="comp.hdf5", **{field: value})


def test_disabled_channels_are_missing_in_compensation_and_zero_in_flash(flash_package):
    with h5py.File(flash_package["path"], "r") as package:
        compensation_path = package["source_files/TX_COMPENSATION"].attrs["path"]
        payload = package["flash_items/TX_COMPENSATION/payload"][...]
        # Four channels per frequency: phase bytes then attenuation bytes.
        assert payload[0] == payload[4] == payload[8] == payload[12] == 0
        assert payload[1] == 48  # -90 degrees wraps to 270 degrees / 5.625.
    with h5py.File(compensation_path, "r") as compensation:
        assert np.all(compensation["compensation/phase_code"][0] == -1)
        assert np.all(np.isnan(compensation["compensation/attenuation_db"][0]))
        assert compensation["compensation"].attrs["rounding"] == "HALF_EVEN"
    assert inspect_data_file(flash_package["path"])["checksum_verified"]


def test_flash_rejects_changed_quantization_metadata(flash_package, small_coordinates):
    with h5py.File(flash_package["path"], "r") as package:
        path = package["source_files/TX_COMPENSATION"].attrs["path"]
    with h5py.File(path, "r+") as compensation:
        compensation["compensation"].attrs["phase_step_deg"] = 90.0
    with pytest.raises(ServiceError, match="量化契约"):
        FlashService()._read_compensation(path, "TX", small_coordinates)


@pytest.mark.parametrize("corruption", ["nonfinite", "finite_changed", "wrong_shape", "wrong_status", "disabled_data"])
def test_history_inspection_rejects_corrupt_measurements(tmp_path, small_coordinates, corruption):
    path = _completed_calibration(tmp_path / "cal.hdf5", small_coordinates)
    assert inspect_data_file(str(path))["integrity"] == "ok"
    with h5py.File(path, "r+") as file:
        if corruption == "nonfinite":
            file["measurements/real"][1, 0] = np.nan
        elif corruption == "finite_changed":
            file["measurements/real"][1, 0] = 99.0
        elif corruption == "wrong_shape":
            del file["measurements/imag"]
            file["measurements"].create_dataset("imag", data=np.zeros((4, 1)))
        elif corruption == "wrong_status":
            file["channels/result_status"][1] = "PENDING"
        else:
            file["measurements/real"][0, 0] = 0.0
    with pytest.raises(ServiceError):
        inspect_data_file(str(path))


def test_missing_digest_is_not_reported_as_checksum_verified(tmp_path, small_coordinates):
    path = _completed_calibration(tmp_path / "cal.hdf5", small_coordinates)
    with h5py.File(path, "r+") as file:
        del file.attrs["content_sha256"]
    result = inspect_data_file(str(path))
    assert result["integrity"] == "content_validated"
    assert result["checksum_verified"] is False


def test_finalize_rejects_false_complete_and_closes_file(tmp_path, small_coordinates):
    path = tmp_path / "incomplete.hdf5"
    store = _store(path, small_coordinates)
    with pytest.raises(ServiceError, match="未完成启用通道"):
        store.finalize("COMPLETED")
    assert not store.file.id.valid
    with h5py.File(path, "r") as file:
        assert file.attrs["status"] == "FAULTED"
        assert file.attrs["file_role"] == "RUN_TERMINAL"


def test_reopen_digest_failure_does_not_leave_completed_label(tmp_path, small_coordinates, monkeypatch):
    path = tmp_path / "reopen-failure.hdf5"
    store = _store(path, small_coordinates)
    for channel in small_coordinates.channels:
        if channel.enabled:
            store.commit_channel(channel, np.array([1j, 1 + 1j]))
    original_digest = hdf5_store.data_content_sha256
    calls = 0

    def different_reopen_digest(file):
        nonlocal calls
        calls += 1
        return original_digest(file) if calls == 1 else "0" * 64

    monkeypatch.setattr(hdf5_store, "data_content_sha256", different_reopen_digest)
    with pytest.raises(ServiceError, match="内容摘要不一致"):
        store.finalize("COMPLETED")
    with h5py.File(path, "r") as file:
        assert file.attrs["status"] == "FAULTED"
        assert file.attrs["file_role"] == "RUN_TERMINAL"


def test_partial_stopped_calibration_remains_available(tmp_path, small_coordinates):
    path = tmp_path / "stopped.hdf5"
    store = _store(path, small_coordinates)
    store.commit_channel(small_coordinates.channels[1], np.array([1j, 1 + 1j]))
    result = store.finalize("STOPPED")
    assert result["status"] == "STOPPED"
    assert inspect_data_file(str(path))["checksum_verified"]
    channels = read_data_view(str(path), 0)["channels"]
    assert channels[1]["magnitude_db"] == 0.0
    assert channels[2]["magnitude_db"] is None


def test_all_disabled_channels_generate_explicit_empty_compensation(tmp_path, small_coordinates):
    coordinates = replace(small_coordinates, channels=[replace(channel, enabled=False) for channel in small_coordinates.channels])
    calibration = _completed_calibration(tmp_path / "disabled-cal.hdf5", coordinates)
    compensation = _compensation(tmp_path / "disabled-comp.hdf5", calibration, coordinates)
    payload, _, _, _ = FlashService()._read_compensation(compensation["path"], "TX", coordinates)
    assert payload == bytes(16)


def test_pattern_complete_requires_all_planned_bundles(tmp_path):
    store = MeasurementStore.create_pattern(tmp_path / "pattern.hdf5", run_id="pattern", metadata={"expected_units": 2},
        frequencies_hz=np.array([8e9]), beams=[{"beam_id": "B", "off_axis_deg": 0, "azimuth_deg": 0, "reference_frequency_hz": 8e9}])
    store.commit_point(bundle_id=0, point_id=0, row_index=0, point_index=0,
        azimuth_deg=0, elevation_deg=0, actual_azimuth_deg=0, actual_elevation_deg=0,
        beam_index=0, beam_id="B", values=np.array([1j]))
    with pytest.raises(ServiceError, match="数量与冻结计划"):
        store.finalize("COMPLETED")


def test_existing_pattern_schema_without_new_evidence_fields_still_checks_contents(tmp_path):
    path = tmp_path / "existing-3.1-pattern.hdf5"
    store = MeasurementStore.create_pattern(path, run_id="pattern", metadata={"expected_units": 1},
        frequencies_hz=np.array([8e9]), beams=[{"beam_id": "B", "off_axis_deg": 0, "azimuth_deg": 0, "reference_frequency_hz": 8e9}])
    store.commit_point(bundle_id=0, point_id=0, row_index=0, point_index=0,
        azimuth_deg=0, elevation_deg=0, actual_azimuth_deg=0, actual_elevation_deg=0,
        beam_index=0, beam_id="B", values=np.array([1j]))
    store.finalize("COMPLETED")
    with h5py.File(path, "r+") as file:
        del file.attrs["expected_units"]
        del file.attrs["content_sha256"]
        del file["metadata"].attrs["expected_units"]
    result = inspect_data_file(str(path))
    assert result["schema"]["schema_version"] == "3.1"
    assert result["integrity"] == "content_validated" and result["checksum_verified"] is False
    assert len(read_data_view(str(path), 0)["points"]) == 1
    with h5py.File(path, "r+") as file:
        file["measurements/imag"][0, 0] = np.nan
    with pytest.raises(ServiceError, match="非有限复数"):
        inspect_data_file(str(path))


class MemoryFlash:
    """A local fake; even the REAL label test never opens any physical transport."""
    def __init__(self, package_path, *, source="SIMULATED", cancel=False, lose_ack=False, bad_read=False):
        self.package_path = package_path
        self.source, self.cancel, self.lose_ack, self.bad_read = source, cancel, lose_ack, bad_read
        self.writes = 0
        self.reads = 0
        self.pages = {}

    async def status(self):
        return {"source": self.source, "identity": "IN_MEMORY_TEST_FAKE", "device_id": "beam_controller"}

    async def flash_write(self, tile_id, address, page):
        with h5py.File(self.package_path, "r") as file:
            assert file["flash_items/TX_COMPENSATION"].attrs["write_status"] == "WRITING"
            assert file["flash_items/TX_COMPENSATION"].attrs["writer_source"] == self.source
        self.writes += 1
        self.pages[(tile_id, address)] = page
        if self.cancel:
            raise asyncio.CancelledError("test cancellation after physical-side-effect boundary")
        if self.lose_ack:
            raise ServiceError("TIMEOUT", "test lost acknowledgement")

    async def flash_read(self, tile_id, address, length):
        self.reads += 1
        return bytes(length) if self.bad_read else self.pages[(tile_id, address)]


@pytest.mark.asyncio
@pytest.mark.parametrize("writer_source,expected", [("SIMULATED", "SIMULATED"), ("REAL", "MIXED")])
async def test_flash_persists_data_and_writer_evidence(flash_package, writer_source, expected):
    path = flash_package["path"]
    assert flash_package["data_evidence"] == "SIMULATED"
    device = MemoryFlash(path, source=writer_source)
    result = await FlashService().write_item(FlashWriteRequest(package_path=path, item_name="TX_COMPENSATION"),
        SimpleNamespace(require=lambda _: device))
    assert result["status"] == "SUCCESS"
    with h5py.File(path, "r") as file:
        assert file.attrs["data_evidence"] == "SIMULATED"
        assert file.attrs["evidence"] == expected
        assert file["source_files/TX_COMPENSATION"].attrs["evidence"] == "SIMULATED"
        assert file["flash_items/TX_COMPENSATION"].attrs["writer_identity"] == "IN_MEMORY_TEST_FAKE"
        assert file["flash_items/TX_COMPENSATION"].attrs["written_pages"] == 1


@pytest.mark.asyncio
async def test_flash_cancel_is_unknown_and_does_not_rewrite(flash_package):
    path = flash_package["path"]
    device = MemoryFlash(path, cancel=True)
    flash, manager = FlashService(), SimpleNamespace(require=lambda _: device)
    request = FlashWriteRequest(package_path=path, item_name="TX_COMPENSATION")
    with pytest.raises(asyncio.CancelledError):
        await flash.write_item(request, manager)
    assert flash.list_items(path)[3]["status"] == "UNKNOWN"
    with pytest.raises(ServiceError, match="必须先人工恢复"):
        await flash.write_item(request, manager)
    assert device.writes == 1


@pytest.mark.asyncio
async def test_flash_lost_ack_uses_same_address_read_without_write_retry(flash_package):
    path = flash_package["path"]
    device = MemoryFlash(path, lose_ack=True)
    result = await FlashService().write_item(FlashWriteRequest(package_path=path, item_name="TX_COMPENSATION"),
        SimpleNamespace(require=lambda _: device))
    assert result["status"] == "SUCCESS" and result["recovered_pages"] == 1
    assert device.writes == 1 and device.reads == 2


@pytest.mark.asyncio
async def test_flash_failed_readback_preserves_failure_and_blocks_new_write(flash_package):
    path = flash_package["path"]
    device = MemoryFlash(path, bad_read=True)
    flash, manager = FlashService(), SimpleNamespace(require=lambda _: device)
    request = FlashWriteRequest(package_path=path, item_name="TX_COMPENSATION")
    with pytest.raises(ServiceError, match="逐字节不一致"):
        await flash.write_item(request, manager)
    assert flash.list_items(path)[3]["status"] == "VERIFY_FAILED"
    with pytest.raises(ServiceError, match="必须先人工恢复"):
        await flash.write_item(request, manager)
    assert device.writes == 1
