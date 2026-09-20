from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from antenna_service.coordinates import CoordinateModel
from antenna_service.errors import ServiceError
from antenna_service.models import CompensationRequest
from antenna_service.storage.hdf5_store import data_content_sha256, file_sha256, validate_data_contents
from antenna_service.workflows.aperture_weights import build_aperture_weights


UTF8 = h5py.string_dtype(encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decode_metadata(file: h5py.File) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in file["metadata"].attrs.items():
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, np.generic):
            value = value.item()
        try:
            result[key] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            result[key] = value
    return result


def _open_calibration(path: str) -> tuple[h5py.File, dict[str, Any]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ServiceError("NOT_FOUND", "标校文件不存在", "compensation_load", str(resolved))
    if not h5py.is_hdf5(resolved):
        raise ServiceError("DATA_INTEGRITY", "标校输入必须是 HDF5 文件", "compensation_load", str(resolved))
    connection = h5py.File(resolved, "r")
    identity = (connection.attrs.get("schema_name"), connection.attrs.get("status"))
    if identity != ("antenna-channel-calibration", "COMPLETED"):
        connection.close()
        raise ServiceError("DATA_INTEGRITY", "输入不是完整标校文件", "compensation_load", str(resolved), {"schema": identity})
    try:
        validate_data_contents(connection, require_complete=True)
        return connection, _decode_metadata(connection)
    except Exception:
        connection.close()
        raise


class CompensationService:
    """Generate a directly inspectable HDF5 TX/RX compensation file."""

    def generate(self, request: CompensationRequest, coordinates: CoordinateModel) -> dict[str, Any]:
        output = Path(request.output_path).resolve()
        if output.suffix.lower() not in {".hdf5", ".h5", ".hdf"}:
            output = output.with_suffix(".hdf5")
        if output.exists():
            raise ServiceError("DATA_INTEGRITY", "补偿输出文件已存在", "compensation_create", str(output))

        sources: list[tuple[Path, h5py.File, dict[str, Any]]] = []
        try:
            for source_path in request.calibration_files:
                connection, metadata = _open_calibration(source_path)
                sources.append((Path(source_path).resolve(), connection, metadata))
            self._validate_sources(request, coordinates, sources)
            output.parent.mkdir(parents=True, exist_ok=True)
            self._write_hdf5(output, request, coordinates, sources)
        finally:
            for _, source, _ in sources:
                source.close()

        # Reopen the completed file to verify data actually persisted to disk.
        if not h5py.is_hdf5(output):
            raise ServiceError("DATA_INTEGRITY", "补偿 HDF5 文件关闭重开失败", "compensation_verify", str(output))
        with h5py.File(output, "r") as verify:
            validate_data_contents(verify, require_complete=True)
            identity = (verify.attrs.get("schema_name"), verify.attrs.get("status"), verify.attrs.get("signal_path"))
            if identity != ("antenna-compensation", "COMPLETED", request.signal_path):
                raise ServiceError("DATA_INTEGRITY", "补偿文件身份复验失败", "compensation_verify", str(output), {"schema": identity})
            counts = {
                "channels": int(verify["channels/element"].shape[0]),
                "compensation": int(np.prod(verify["compensation/phase_code"].shape)),
            }
            frequencies_hz = np.asarray(verify["frequencies_hz"], dtype=np.float64).tolist()
            taper_values = np.asarray(verify["aperture_weighting/taper_attenuation_db"], dtype=np.float64)
            finite_taper = taper_values[np.isfinite(taper_values)]
            weighting = {
                "algorithm": str(verify["aperture_weighting"].attrs["algorithm"]),
                "axes": str(verify["aperture_weighting"].attrs["axes"]),
                "maximum_taper_db": float(np.max(finite_taper)) if finite_taper.size else 0.0,
            }
        return {
            "path": str(output),
            "sha256": file_sha256(output),
            "status": "COMPLETED",
            "frequencies_hz": frequencies_hz,
            "tables": counts,
            "aperture_weighting": weighting,
        }

    def _validate_sources(
        self,
        request: CompensationRequest,
        coordinates: CoordinateModel,
        sources: list[tuple[Path, h5py.File, dict[str, Any]]],
    ) -> None:
        seen_polarizations: set[str] = set()
        frequency_axes: list[list[float]] = []
        for path, connection, metadata in sources:
            if metadata.get("signal_path") != request.signal_path:
                raise ServiceError("DATA_INTEGRITY", "标校文件 TX/RX 与请求不一致", "compensation_validate", str(path))
            polarization = str(metadata.get("polarization"))
            if polarization in seen_polarizations:
                raise ServiceError("DATA_INTEGRITY", "同一极化选择了多个标校文件", "compensation_validate", polarization)
            seen_polarizations.add(polarization)
            if polarization not in coordinates.polarizations:
                raise ServiceError("DATA_INTEGRITY", "标校极化不在坐标表中", "compensation_validate", polarization)
            for key, expected in {
                "coordinate_sha256": coordinates.file_sha256,
                "mapping_sha256": coordinates.mapping_sha256,
                "enabled_sha256": coordinates.enabled_sha256,
                "geometry_sha256": coordinates.geometry_sha256,
            }.items():
                if metadata.get(key) != expected:
                    raise ServiceError("DATA_INTEGRITY", f"标校文件 {key} 与坐标表不一致", "compensation_validate", str(path))
            enabled = np.asarray(connection["channels/enabled"], dtype=bool)
            source_channels = coordinates.for_polarization(polarization)
            for name in ("element", "spi_no", "chip_no", "chip_channel_index", "grid_row", "grid_column", "x", "y", "z", "enabled"):
                expected_values = np.asarray([getattr(channel, name) for channel in source_channels])
                if not np.array_equal(connection[f"channels/{name}"][...], expected_values):
                    raise ServiceError("DATA_INTEGRITY", f"标校文件通道 {name} 与坐标表不一致", "compensation_validate", str(path))
            statuses = connection["channels/result_status"].asstr()[...]
            failed = int(np.count_nonzero(enabled & (statuses != "CALIBRATED")))
            if failed:
                raise ServiceError("DATA_INTEGRITY", "标校文件存在未完成启用通道", "compensation_validate", str(path), {"count": failed})
            frequency_axes.append(np.asarray(connection["frequencies_hz"], dtype=float).tolist())
        if any(axis != frequency_axes[0] for axis in frequency_axes[1:]):
            raise ServiceError("DATA_INTEGRITY", "不同极化标校频率轴不一致", "compensation_validate")
        if any(index >= len(frequency_axes[0]) for index in request.frequency_indices):
            raise ServiceError("INVALID_REQUEST", "所选补偿频点超出标校文件范围", "compensation_validate")

    def _write_hdf5(
        self,
        output: Path,
        request: CompensationRequest,
        coordinates: CoordinateModel,
        sources: list[tuple[Path, h5py.File, dict[str, Any]]],
    ) -> None:
        selected_polarizations = {str(metadata["polarization"]) for _, _, metadata in sources}
        channels = [channel for channel in coordinates.channels if channel.polarization in selected_polarizations]
        source_frequency_axis = np.asarray(sources[0][1]["frequencies_hz"], dtype=np.float64)
        # Selection order is a UI concern. The persisted compensation and later FLASH
        # frequency blocks are always ordered by the actual frequency value ascending.
        source_frequency_indices = sorted(request.frequency_indices, key=lambda index: float(source_frequency_axis[index]))
        frequencies = source_frequency_axis[source_frequency_indices]
        channel_count, frequency_count = len(channels), len(frequencies)
        aperture = build_aperture_weights(channels, request)
        floats = {
            name: np.full((channel_count, frequency_count), np.nan, dtype=np.float64)
            for name in (
                "measured_magnitude_db",
                "measured_phase_deg",
                "phase_compensation_deg",
                "phase_quantization_error_deg",
                "calibration_attenuation_db",
                "attenuation_db",
                "attenuation_quantization_error_db",
            )
        }
        phase_codes = np.full((channel_count, frequency_count), -1, dtype=np.int32)
        attenuation_codes = np.full((channel_count, frequency_count), -1, dtype=np.int32)
        source_by_polarization = {str(metadata["polarization"]): source for _, source, metadata in sources}
        source_row_by_polarization = {
            polarization: {int(element): index for index, element in enumerate(np.asarray(source["channels/element"], dtype=int))}
            for polarization, source in source_by_polarization.items()
        }

        for frequency_index, source_frequency_index in enumerate(source_frequency_indices):
            measured_by_channel: dict[int, complex] = {}
            for channel_index, channel in enumerate(channels):
                if not channel.enabled:
                    continue
                source = source_by_polarization[channel.polarization]
                row = source_row_by_polarization[channel.polarization][channel.element]
                real = float(source["measurements/real"][row, source_frequency_index])
                imag = float(source["measurements/imag"][row, source_frequency_index])
                if not math.isfinite(real) or not math.isfinite(imag):
                    raise ServiceError("DATA_INTEGRITY", "启用通道缺少有效复数测量值", "compensation_compute", f"{channel.polarization}:{channel.element}")
                measured_by_channel[channel_index] = complex(real, imag)
            magnitudes_db = {
                index: 20 * math.log10(abs(value))
                for index, value in measured_by_channel.items()
                if abs(value) > 0
            }
            if len(magnitudes_db) != len(measured_by_channel):
                raise ServiceError("DATA_INTEGRITY", "启用通道存在零幅度，无法生成补偿", "compensation_compute")
            # An entirely disabled polarization has no measurements and retains
            # explicit missing codes; FLASH later emits its disabled bytes as zero.
            weakest_db = min(magnitudes_db.values(), default=0.0)
            for channel_index, value in measured_by_channel.items():
                magnitude_db = magnitudes_db[channel_index]
                phase = math.degrees(math.atan2(value.imag, value.real)) % 360
                ideal_phase = (-phase) % 360
                phase_code = int(round(ideal_phase / request.phase_step_deg)) % max(1, int(round(360 / request.phase_step_deg)))
                quantized_phase = (phase_code * request.phase_step_deg) % 360
                phase_error = ((quantized_phase - ideal_phase + 180) % 360) - 180
                # Channel equalization and aperture taper are deliberately kept
                # separate in HDF5.  The hardware receives their sum as its one
                # attenuation code; phase compensation is not changed by tapering.
                calibration_attenuation = magnitude_db - weakest_db
                taper_attenuation = float(aperture.attenuation_db[channel_index])
                ideal_attenuation = calibration_attenuation + taper_attenuation
                if ideal_attenuation > request.max_attenuation_db + 1e-9:
                    channel = channels[channel_index]
                    raise ServiceError(
                        "NOT_RUNNABLE",
                        "所需通道衰减超过硬件上限",
                        "compensation_compute",
                        f"{channel.polarization}:{channel.element}",
                        {
                            "calibration_db": calibration_attenuation,
                            "taper_db": taper_attenuation,
                            "required_db": ideal_attenuation,
                            "maximum_db": request.max_attenuation_db,
                        },
                    )
                attenuation_code = int(round(ideal_attenuation / request.attenuation_step_db))
                quantized_attenuation = attenuation_code * request.attenuation_step_db
                floats["measured_magnitude_db"][channel_index, frequency_index] = magnitude_db
                floats["measured_phase_deg"][channel_index, frequency_index] = phase
                floats["phase_compensation_deg"][channel_index, frequency_index] = quantized_phase
                floats["phase_quantization_error_deg"][channel_index, frequency_index] = phase_error
                floats["calibration_attenuation_db"][channel_index, frequency_index] = calibration_attenuation
                floats["attenuation_db"][channel_index, frequency_index] = quantized_attenuation
                floats["attenuation_quantization_error_db"][channel_index, frequency_index] = quantized_attenuation - ideal_attenuation
                phase_codes[channel_index, frequency_index] = phase_code
                attenuation_codes[channel_index, frequency_index] = attenuation_code

        with h5py.File(output, "w") as destination:
            destination.attrs.update(
                {
                    "schema_name": "antenna-compensation",
                    "schema_version": "2.1",
                    "file_role": "COMPENSATION_RESULT",
                    "status": "IN_PROGRESS",
                    "signal_path": request.signal_path,
                    "created_at": _utc_now(),
                    "antenna_id": coordinates.antenna_id,
                    "coordinate_sha256": coordinates.file_sha256,
                    "mapping_sha256": coordinates.mapping_sha256,
                    "enabled_sha256": coordinates.enabled_sha256,
                    "geometry_sha256": coordinates.geometry_sha256,
                    "evidence": self._combined_evidence(coordinates.evidence, [str(metadata.get("evidence", "NONE")) for _, _, metadata in sources]),
                }
            )
            destination.create_dataset("frequencies_hz", data=frequencies)
            channel_group = destination.create_group("channels")
            channel_group.create_dataset("polarization", data=np.asarray([channel.polarization for channel in channels], dtype=object), dtype=UTF8)
            for name, dtype in {
                "element": np.int32,
                "spi_no": np.int32,
                "chip_no": np.int32,
                "chip_channel_index": np.int32,
                "grid_row": np.int32,
                "grid_column": np.int32,
                "enabled": np.bool_,
            }.items():
                channel_group.create_dataset(name, data=np.asarray([getattr(channel, name) for channel in channels], dtype=dtype))
            for name in ("x", "y", "z"):
                channel_group.create_dataset(name, data=np.asarray([getattr(channel, name) for channel in channels], dtype=np.float64))
            weighting_group = destination.create_group("aperture_weighting")
            weighting_group.attrs.update(
                {
                    "algorithm": request.sidelobe_algorithm,
                    "axes": request.sidelobe_axes,
                    "taylor_nbar": request.taylor_nbar,
                    "taylor_sll_db": request.taylor_sll_db,
                    "kaiser_beta": request.kaiser_beta,
                }
            )
            weighting_group.create_dataset("linear_weight", data=aperture.linear)
            weighting_group.create_dataset("taper_attenuation_db", data=aperture.attenuation_db)
            compensation_group = destination.create_group("compensation")
            compensation_group.attrs.update({
                "phase_step_deg": request.phase_step_deg,
                "attenuation_step_db": request.attenuation_step_db,
                "max_attenuation_db": request.max_attenuation_db,
                "rounding": "HALF_EVEN",
            })
            for name, values in floats.items():
                compensation_group.create_dataset(name, data=values, compression="gzip", shuffle=True)
            compensation_group.create_dataset("phase_code", data=phase_codes, compression="gzip", shuffle=True)
            compensation_group.create_dataset("attenuation_code", data=attenuation_codes, compression="gzip", shuffle=True)
            provenance = destination.create_group("provenance")
            provenance.create_dataset("polarization", data=np.asarray([metadata["polarization"] for _, _, metadata in sources], dtype=object), dtype=UTF8)
            provenance.create_dataset("source_path", data=np.asarray([str(path) for path, _, _ in sources], dtype=object), dtype=UTF8)
            provenance.create_dataset("source_sha256", data=np.asarray([file_sha256(path) for path, _, _ in sources], dtype=object), dtype=UTF8)
            provenance.create_dataset("evidence", data=np.asarray([metadata.get("evidence", "NONE") for _, _, metadata in sources], dtype=object), dtype=UTF8)
            provenance.create_dataset("source_frequency_index", data=np.asarray(source_frequency_indices, dtype=np.int32))
            destination.attrs["status"] = "COMPLETED"
            destination.attrs["completed_at"] = _utc_now()
            destination.attrs["content_sha256"] = data_content_sha256(destination)
            destination.flush()

    @staticmethod
    def _combined_evidence(coordinate_evidence: str, source_evidence: list[str]) -> str:
        evidence = {coordinate_evidence, *source_evidence} - {"NONE", ""}
        if evidence == {"REAL"}:
            return "REAL"
        if evidence == {"SIMULATED"}:
            return "SIMULATED"
        return "MIXED"
