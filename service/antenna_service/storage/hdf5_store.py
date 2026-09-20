from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from antenna_service.coordinates import Channel, CoordinateModel
from antenna_service.errors import ServiceError
from antenna_service.protocol.flash import LAST_PAGE_ADDRESS, MAX_ADDRESS, PAGE_SIZE


UTF8 = h5py.string_dtype(encoding="utf-8")


def timestamped_path(directory: str, base_filename: str, test_type: str) -> Path:
    label = "标校" if test_type == "CALIBRATION" else "方向图"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = Path(directory) / f"{base_filename}_{label}_{timestamp}.hdf5"
    if path.exists():
        raise ServiceError("DATA_INTEGRITY", "输出文件已存在", "data_create", str(path))
    return path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attribute_value(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _decode_attribute(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str) and value[:1] in {"{", "["}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def data_content_sha256(file: h5py.File) -> str:
    """Hash ordinary datasets in bounded chunks, including names, shapes and dtypes."""
    digest = hashlib.sha256()
    datasets: list[str] = []
    file.visititems(lambda name, item: datasets.append(name) if isinstance(item, h5py.Dataset) else None)
    for name in sorted(datasets):
        dataset = file[name]
        digest.update(json.dumps([name, dataset.shape, dataset.dtype.str]).encode("utf-8"))
        row_size = max(1, int(np.prod(dataset.shape[1:])) * max(8, dataset.dtype.itemsize))
        batch = max(1, 1024 * 1024 // row_size)
        selections = (slice(start, start + batch) for start in range(0, dataset.shape[0], batch)) if dataset.ndim else [()]
        for selection in selections:
            values = np.asarray(dataset[selection])
            if values.dtype.kind in {"O", "S", "U"}:
                for value in values.reshape(-1):
                    encoded = value if isinstance(value, bytes) else str(value).encode("utf-8")
                    digest.update(len(encoded).to_bytes(8, "big"))
                    digest.update(encoded)
            else:
                digest.update(values.tobytes())
    return digest.hexdigest()


def validate_data_contents(file: h5py.File, *, require_complete: bool = False) -> bool:
    """Validate the current schemas and any saved content digest, not just HDF readability.

    Return whether a persisted digest was actually compared. Files without such a
    digest can pass semantic checks, but must not be described as checksum-verified.
    """
    def check(condition: bool, message: str) -> None:
        if not condition:
            raise ServiceError("DATA_INTEGRITY", message, "data_validate", file.filename)

    def vector(path: str, count: int) -> np.ndarray:
        dataset = file[path]
        check(dataset.shape == (count,), f"{path} 数据集维度不一致")
        return dataset[...]

    try:
        schema = str(file.attrs.get("schema_name", ""))
        versions = {
            "antenna-channel-calibration": "3.2", "antenna-pattern-run": "3.1",
            "antenna-compensation": "2.1", "antenna-flash-package": "2.0",
        }
        check(schema in versions and file.attrs.get("schema_version") == versions[schema], "HDF5 Schema 身份或版本不支持")
        frequencies = np.asarray(file["frequencies_hz"], dtype=float)
        check(frequencies.ndim == 1 and frequencies.size > 0 and np.all(np.isfinite(frequencies))
              and np.all(frequencies > 0) and np.all(np.diff(frequencies) > 0), "HDF5 频率轴必须为正、有限且严格递增")
        status = str(file.attrs.get("status", ""))
        complete = require_complete or status == "COMPLETED"
        if schema in {"antenna-channel-calibration", "antenna-pattern-run"}:
            check(status in {"RUNNING", "PAUSED", "STOPPING", "COMPLETED", "STOPPED", "FAULTED", "UNKNOWN"}, "测量文件运行状态无效")
            if status == "COMPLETED":
                check(file.attrs.get("file_role") == "COMPLETE_RESULT", "完成状态与文件角色不一致")
            events = file["run_events"]
            for name in ("timestamp", "level", "stage", "message", "details"):
                vector(f"run_events/{name}", len(events["timestamp"]))
            group_name = "channels" if schema == "antenna-channel-calibration" else "spatial_points"
            identity_name = "element" if group_name == "channels" else "bundle_id"
            count = len(file[f"{group_name}/{identity_name}"])
            for name in file[group_name]:
                vector(f"{group_name}/{name}", count)
            real, imag = file["measurements/real"], file["measurements/imag"]
            check(real.shape == imag.shape == (count, len(frequencies)), "复数测量矩阵与通道/测量束/频率维度不一致")
            statuses = file[f"{group_name}/result_status"].asstr()[...]
            if group_name == "channels":
                check(count > 0, "标校通道集合为空")
                for name in ("element", "spi_no", "chip_no", "chip_channel_index", "grid_row", "grid_column", "polarization", "enabled", "error_code", "x", "y", "z"):
                    vector(f"channels/{name}", count)
                for name in ("element", "spi_no", "chip_no", "chip_channel_index", "grid_row", "grid_column"):
                    values = file[f"channels/{name}"][...]
                    check(values.dtype.kind in {"i", "u"} and np.all(values >= 0), "通道编号和映射必须为非负整数")
                elements = file["channels/element"][...]
                check(np.array_equal(elements, np.arange(count)), "标校通道编号必须从零连续排列")
                enabled_values = file["channels/enabled"][...]
                check(np.all(np.isin(enabled_values, [0, 1])), "通道使能数据无效")
                enabled = enabled_values.astype(bool)
                valid = statuses == "CALIBRATED"
                check(np.all(np.isin(statuses, ["PENDING", "CALIBRATED", "SKIPPED_DISABLED", "FAILED"])), "标校通道结果状态无效")
                check(np.all(statuses[~enabled] == "SKIPPED_DISABLED") and not np.any(statuses[enabled] == "SKIPPED_DISABLED"), "通道使能与结果状态不一致")
                if complete:
                    check(np.all(valid[enabled]), "完整标校文件仍有未完成启用通道")
                    check(count == int(file.attrs.get("expected_units", count)), "标校通道数与冻结计划不一致")
                for name in ("x", "y", "z"):
                    check(np.all(np.isfinite(file[f"channels/{name}"][...])), "通道几何坐标存在非有限数")
                if file["metadata"].attrs.get("topology", "").startswith("RTC_"):
                    group = file["rtc/channel_measurements"]
                    for name in ("wave_address", "accepted_groups", "completed_groups", "valid_triggers", "completed_points", "close_status"):
                        vector(f"rtc/channel_measurements/{name}", count)
                    close_states = group["close_status"].asstr()[...]
                    check(np.all(np.isin(close_states, ["PENDING", "CONFIRMED", "UNKNOWN", "SKIPPED_DISABLED"])),
                          "RTC关闭状态无效")
                    check(np.all(close_states[~enabled] == "SKIPPED_DISABLED"), "禁用通道关闭状态无效")
                    addresses = group["wave_address"][...]
                    check(np.all(addresses[~enabled] == 0) and np.all(addresses[enabled] >= 1)
                          and np.all(addresses[enabled] <= len(file["rtc/wave_frames"])), "RTC标校波位地址无效")
                    for name in ("close_request", "close_response"):
                        check(group[name].shape == (count, 22), "RTC关闭报文证据维度错误")
                    confirmed = close_states == "CONFIRMED"
                    if file["rtc"].attrs.get("close_success_rule") == "FRAME_EQUALS_REQUEST":
                        check(np.array_equal(group["close_request"][...][confirmed], group["close_response"][...][confirmed]),
                              "RTC关闭应答与请求不一致")
                    check(np.all(group["accepted_groups"][...][valid] == 1)
                          and np.all(group["completed_groups"][...][valid] == 1)
                          and np.all(group["valid_triggers"][...][valid] == len(frequencies))
                          and np.all(group["completed_points"][...][valid] == len(frequencies)), "RTC标校计数不一致")
                    if complete:
                        check(np.all(close_states[enabled] == "CONFIRMED"), "RTC标校通道尚未确认关闭")
            else:
                for name in ("bundle_id", "point_id", "row_index", "point_index", "beam_index", "beam_id", "azimuth_deg", "elevation_deg", "actual_azimuth_deg", "actual_elevation_deg", "error_code"):
                    vector(f"spatial_points/{name}", count)
                for name in ("bundle_id", "point_id", "row_index", "point_index", "beam_index"):
                    values = file[f"spatial_points/{name}"][...]
                    check(values.dtype.kind in {"i", "u"} and np.all(values >= 0), "测量束编号必须为非负整数")
                beams = file["beam_definitions/beam_id"].asstr()[...]
                check(len(beams) > 0 and len(set(beams)) == len(beams), "电子波束集合为空或 ID 重复")
                for name in ("off_axis_deg", "azimuth_deg", "reference_frequency_hz"):
                    check(np.all(np.isfinite(vector(f"beam_definitions/{name}", len(beams)))), "波束定义存在非有限数")
                beam_indexes = file["spatial_points/beam_index"][...]
                check(np.all((beam_indexes >= 0) & (beam_indexes < len(beams))), "测量束引用了无效电子波束")
                check(np.array_equal(file["spatial_points/beam_id"].asstr()[...], beams[beam_indexes]), "测量束电子波束 ID 不匹配")
                check(np.array_equal(file["spatial_points/bundle_id"][...], np.arange(count)), "方向图测量束编号缺失或重复")
                rtc_continuous = (file["metadata"].attrs.get("topology") == "RTC_CONTINUOUS")
                for name in ("azimuth_deg", "elevation_deg", "actual_azimuth_deg", "actual_elevation_deg"):
                    values = file[f"spatial_points/{name}"][...]
                    if rtc_continuous and name == "actual_azimuth_deg":
                        # Continuous hardware supplies position-trigger counts, not a
                        # latched encoder value per VNA sample. Unknown readback remains
                        # NaN; never fill target angles into actual-angle evidence.
                        check(file["metadata"].attrs.get("position_source") == "TRIGGER_GRID"
                              and np.all(np.isnan(values)), "连续扫描的逐点实际方位须明确为未回读")
                    else:
                        check(np.all(np.isfinite(values)), "目标或实际机械角存在非有限数")
                if file["metadata"].attrs.get("topology", "").startswith("RTC_"):
                    rtc_group = file["rtc/acquisitions"]
                    size = len(rtc_group["row_index"])
                    for name in rtc_group:
                        vector(f"rtc/acquisitions/{name}", size)
                    points = rtc_group["point_count"][...]
                    check(np.all(points > 0), "RTC采集记录点数无效")
                    check(np.all(rtc_group["pulse_output_disabled"][...] == 1), "RTC行读取前转台脉冲关闭未确认")
                    for name in ("accepted_groups", "completed_groups"):
                        check(np.array_equal(rtc_group[name][...], points), "RTC组计数与记录不一致")
                    expected_sweeps = points * len(beams)
                    check(np.array_equal(rtc_group["buffer_sweeps"][...], expected_sweeps), "RTC缓冲扫频数不一致")
                    for name in ("valid_triggers", "completed_points"):
                        check(np.array_equal(rtc_group[name][...], expected_sweeps * len(frequencies)),
                              "RTC单点计数与缓冲频点数不一致")
                    check(np.all(np.isfinite(rtc_group["start_azimuth_readback_deg"][...]))
                          and np.all(np.isfinite(rtc_group["end_azimuth_readback_deg"][...]))
                          and np.all(np.isfinite(rtc_group["actual_elevation_deg"][...])), "RTC行/组位置回读缺失")
                    if complete:
                        check(int(np.sum(expected_sweeps)) == count, "RTC采集记录与已保存测量束数量不一致")
                check(np.all(np.isin(statuses, ["PENDING", "COMPLETE", "FAILED"])), "方向图测量束状态无效")
                valid = statuses == "COMPLETE"
                enabled = np.ones(count, dtype=bool)
                if complete:
                    check(count > 0 and np.all(valid), "完整方向图文件仍有未完成测量束")
                    if require_complete:
                        check("expected_units" in file.attrs, "方向图缺少冻结计划测量束数量")
                    if "expected_units" in file.attrs:
                        check(count == int(file.attrs["expected_units"]), "方向图测量束数量与冻结计划不一致")
            # Do not load a full large sweep matrix just to validate it.
            for start in range(0, count, 128):
                values_real, values_imag = real[start:start + 128], imag[start:start + 128]
                valid_rows = valid[start:start + 128]
                check(np.all(np.isfinite(values_real[valid_rows])) and np.all(np.isfinite(values_imag[valid_rows])), "已完成测量中存在缺失或非有限复数")
                disabled_rows = ~enabled[start:start + 128]
                check(np.all(np.isnan(values_real[disabled_rows])) and np.all(np.isnan(values_imag[disabled_rows])), "禁用通道不应含有测量数据")
        elif schema == "antenna-compensation":
            check(status == "COMPLETED" and file.attrs.get("signal_path") in {"TX", "RX"}, "补偿文件身份或完成状态无效")
            check(file.attrs.get("evidence") in {"REAL", "SIMULATED", "MIXED"}, "补偿文件缺少有效来源证据")
            count = len(file["channels/element"])
            check(count > 0, "补偿通道集合为空")
            for name in ("polarization", "element", "spi_no", "chip_no", "chip_channel_index", "grid_row", "grid_column", "x", "y", "z", "enabled"):
                vector(f"channels/{name}", count)
            for name in ("element", "spi_no", "chip_no", "chip_channel_index", "grid_row", "grid_column"):
                values = file[f"channels/{name}"][...]
                check(values.dtype.kind in {"i", "u"} and np.all(values >= 0), "补偿通道编号和映射必须为非负整数")
            enabled_values = file["channels/enabled"][...]
            check(np.all(np.isin(enabled_values, [0, 1])), "补偿通道使能无效")
            enabled = enabled_values.astype(bool)
            group = file["compensation"]
            for name, expected in {"phase_step_deg": 5.625, "attenuation_step_db": 0.5, "max_attenuation_db": 31.5, "rounding": "HALF_EVEN"}.items():
                check(group.attrs.get(name) == expected, f"补偿量化契约 {name} 与固件不一致，请重新生成补偿")
            for name in ("phase_code", "attenuation_code"):
                codes = np.asarray(group[name])
                check(codes.shape == (count, len(frequencies)) and codes.dtype.kind in {"i", "u"}, "补偿量化码类型或维度错误")
                check(np.all((codes[enabled] >= 0) & (codes[enabled] <= 63)) and np.all(codes[~enabled] == -1), "补偿启用通道码必须为 0..63，禁用通道必须为缺失码 -1")
            for name in ("measured_magnitude_db", "measured_phase_deg", "phase_compensation_deg", "phase_quantization_error_deg", "calibration_attenuation_db", "attenuation_db", "attenuation_quantization_error_db"):
                values = np.asarray(group[name])
                check(values.shape == (count, len(frequencies)), "补偿数据集维度不一致")
                check(np.all(np.isfinite(values[enabled])) and np.all(np.isnan(values[~enabled])), "补偿有效值或禁用通道空值不正确")
            check(np.array_equal(group["phase_compensation_deg"][...][enabled], group["phase_code"][...][enabled] * 5.625), "相位量化码与角度不一致")
            check(np.array_equal(group["attenuation_db"][...][enabled], group["attenuation_code"][...][enabled] * 0.5), "衰减量化码与 dB 不一致")
            weights = vector("aperture_weighting/linear_weight", count)
            taper = vector("aperture_weighting/taper_attenuation_db", count)
            check(np.all(np.isfinite(weights)) and np.all((weights[enabled] > 0) & (weights[enabled] <= 1))
                  and np.all(weights[~enabled] == 0) and np.all(np.isfinite(taper[enabled])) and np.all(np.isnan(taper[~enabled])), "孔径权值或禁用通道空值无效")
        else:
            check(file.attrs.get("payload_layout") == "FLASH_INTERNAL_OFFICIAL_V1" and status == "PREPARED", "FLASH 载荷布局或包状态无效")
            check(file.attrs.get("evidence") in {"REAL", "SIMULATED", "MIXED"}, "FLASH 工程包缺少来源证据，请重新准备")
            check(0 <= int(file.attrs["tile_id"]) <= 255, "FLASH 阵面 ID 超出范围")
            names = {"ARRAY_ID", "SWITCH_TABLE", "COORDINATE", "TX_COMPENSATION", "RX_COMPENSATION"}
            check(set(file["flash_items"]) == names, "FLASH 五类项目缺失或多余")
            sources = [str(file[f"source_files/{name}"].attrs["evidence"]) for name in ("COORDINATES", "TX_COMPENSATION", "RX_COMPENSATION")]
            check(all(source in {"REAL", "SIMULATED", "MIXED"} for source in sources), "FLASH 输入来源无效")
            data_evidence = sources[0] if len(set(sources)) == 1 else "MIXED"
            check(file.attrs.get("data_evidence") == data_evidence, "FLASH 数据来源与输入文件来源不一致")
            writer_sources = []
            for name in names:
                item = file[f"flash_items/{name}"]
                payload = np.asarray(item["payload"])
                occupied, effective = int(item.attrs["occupied_length"]), int(item.attrs["effective_length"])
                start = int(item.attrs["start_address"])
                check(payload.dtype == np.dtype("uint8") and payload.shape == (occupied,) and occupied > 0 and occupied % PAGE_SIZE == 0, "FLASH 页载荷类型或长度错误")
                check(0 < effective <= occupied and np.all(payload[effective:] == 0), "FLASH 有效长度或补零错误")
                check(0 <= start <= LAST_PAGE_ADDRESS and start % PAGE_SIZE == 0 and start + occupied - 1 <= MAX_ADDRESS, "FLASH 单项地址或末页越界")
                check(int(item.attrs["end_address"]) == start + occupied - 1 and int(item.attrs["page_count"]) * PAGE_SIZE == occupied
                      and int(item.attrs["padding_length"]) == occupied - effective, "FLASH 项目长度元数据不一致")
                check(hashlib.sha256(payload.tobytes()).hexdigest() == item.attrs["payload_sha256"]
                      and hashlib.sha256(payload[:effective].tobytes()).hexdigest() == item.attrs["effective_sha256"], "FLASH 页载荷摘要不一致")
                check(item.attrs.get("write_status") in {"PREPARED", "WRITING", "SUCCESS", "UNKNOWN", "PARTIAL_WRITE", "VERIFY_FAILED"}, "FLASH 项目写入状态无效")
                check(item.attrs.get("data_evidence") == data_evidence, "FLASH 项目数据来源不一致")
                writer_source = str(item.attrs.get("writer_source", ""))
                check(writer_source in {"", "REAL", "SIMULATED"}, "FLASH 写入设备来源无效")
                if writer_source:
                    writer_sources.append(writer_source)
                if item.attrs["write_status"] == "SUCCESS":
                    check(bool(writer_source) and item.attrs.get("readback_sha256") == item.attrs["payload_sha256"]
                          and int(item.attrs["written_pages"]) == int(item.attrs["page_count"]), "FLASH 成功状态缺少完整写入和读回证据")
                if name == "ARRAY_ID":
                    check(effective == 1 and payload[0] == int(file.attrs["tile_id"]), "FLASH 阵面 ID 页与目标不一致")
            combined = {data_evidence, *writer_sources}
            check(file.attrs["evidence"] == (data_evidence if len(combined) == 1 else "MIXED"), "FLASH 工程来源与数据/设备来源不一致")
        expected_digest = file.attrs.get("content_sha256")
        if expected_digest:
            check(expected_digest == data_content_sha256(file), "HDF5 内容摘要不一致")
        plan_count_available = schema != "antenna-pattern-run" or "expected_units" in file.attrs
        return (bool(expected_digest) and plan_count_available) or schema == "antenna-flash-package"
    except ServiceError:
        raise
    except (OSError, KeyError, ValueError, TypeError, IndexError, OverflowError) as exc:
        raise ServiceError("DATA_INTEGRITY", "HDF5 数据结构或内容不完整", "data_validate", file.filename, {"error": str(exc)}) from exc


class MeasurementStore:
    """Inspectable HDF5 measurement store.

    Channel/point axes and real/imaginary samples are ordinary datasets. Metadata
    uses individual attributes, so viewers such as HDFView can inspect the file
    without decoding a root-level document blob.
    """

    def __init__(
        self,
        path: Path,
        *,
        schema_name: str,
        schema_version: str,
        run_id: str,
        metadata: dict[str, Any],
        frequencies_hz: np.ndarray,
    ) -> None:
        self.path = path
        self.file = h5py.File(path, "w", libver="latest")
        self.file.attrs.update(
            {
                "schema_name": schema_name,
                "schema_version": schema_version,
                "file_role": "IN_PROGRESS",
                "status": "RUNNING",
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": "",
            }
        )
        if "expected_units" in metadata:
            self.file.attrs["expected_units"] = int(metadata["expected_units"])
        metadata_group = self.file.create_group("metadata")
        for key, value in metadata.items():
            metadata_group.attrs[key] = _attribute_value(value)
        self.file.create_dataset("frequencies_hz", data=np.asarray(frequencies_hz, dtype=np.float64))
        events = self.file.create_group("run_events")
        for name in ("timestamp", "level", "stage", "message", "details"):
            events.create_dataset(name, shape=(0,), maxshape=(None,), dtype=UTF8)
        self._channel_rows: dict[int, int] = {}
        self.file.flush()

    @classmethod
    def create_calibration(
        cls,
        path: Path,
        *,
        run_id: str,
        metadata: dict[str, Any],
        frequencies_hz: np.ndarray,
        coordinates: CoordinateModel,
        polarization: str,
    ) -> "MeasurementStore":
        store = cls(
            path,
            schema_name="antenna-channel-calibration",
            schema_version="3.2",
            run_id=run_id,
            metadata=metadata,
            frequencies_hz=frequencies_hz,
        )
        channels = coordinates.for_polarization(polarization)
        if "expected_units" not in store.file.attrs:
            store.file.attrs["expected_units"] = len(channels)
        group = store.file.create_group("channels")
        numeric_fields = {
            "element": np.array([item.element for item in channels], dtype=np.int32),
            "spi_no": np.array([item.spi_no for item in channels], dtype=np.int16),
            "chip_no": np.array([item.chip_no for item in channels], dtype=np.int16),
            "chip_channel_index": np.array([item.chip_channel_index for item in channels], dtype=np.int16),
            "grid_row": np.array([item.grid_row for item in channels], dtype=np.int32),
            "grid_column": np.array([item.grid_column for item in channels], dtype=np.int32),
            "x": np.array([item.x for item in channels], dtype=np.float64),
            "y": np.array([item.y for item in channels], dtype=np.float64),
            "z": np.array([item.z for item in channels], dtype=np.float64),
            "enabled": np.array([item.enabled for item in channels], dtype=np.bool_),
        }
        for name, values in numeric_fields.items():
            group.create_dataset(name, data=values)
        group.create_dataset("polarization", data=np.array([item.polarization for item in channels], dtype=UTF8))
        group.create_dataset(
            "result_status",
            data=np.array(["PENDING" if item.enabled else "SKIPPED_DISABLED" for item in channels], dtype=UTF8),
        )
        group.create_dataset("error_code", data=np.array(["" for _ in channels], dtype=UTF8))
        measurements = store.file.create_group("measurements")
        shape = (len(channels), store.frequency_count)
        chunks = (1, max(1, store.frequency_count))
        measurements.create_dataset("real", shape=shape, dtype=np.float64, fillvalue=np.nan, chunks=chunks, compression="gzip")
        measurements.create_dataset("imag", shape=shape, dtype=np.float64, fillvalue=np.nan, chunks=chunks, compression="gzip")
        store._channel_rows = {channel.element: index for index, channel in enumerate(channels)}
        store.file.flush()
        return store

    @classmethod
    def create_pattern(
        cls,
        path: Path,
        *,
        run_id: str,
        metadata: dict[str, Any],
        frequencies_hz: np.ndarray,
        beams: list[dict[str, Any]],
    ) -> "MeasurementStore":
        store = cls(
            path,
            schema_name="antenna-pattern-run",
            schema_version="3.1",
            run_id=run_id,
            metadata=metadata,
            frequencies_hz=frequencies_hz,
        )
        beam_group = store.file.create_group("beam_definitions")
        beam_group.create_dataset("beam_id", data=np.array([item["beam_id"] for item in beams], dtype=UTF8))
        beam_group.create_dataset("off_axis_deg", data=np.array([item["off_axis_deg"] for item in beams], dtype=np.float64))
        beam_group.create_dataset("azimuth_deg", data=np.array([item["azimuth_deg"] for item in beams], dtype=np.float64))
        beam_group.create_dataset(
            "reference_frequency_hz",
            data=np.array([item["reference_frequency_hz"] for item in beams], dtype=np.float64),
        )
        points = store.file.create_group("spatial_points")
        for name, dtype in {
            "bundle_id": np.int32,
            "point_id": np.int32,
            "row_index": np.int32,
            "point_index": np.int32,
            "azimuth_deg": np.float64,
            "elevation_deg": np.float64,
            "actual_azimuth_deg": np.float64,
            "actual_elevation_deg": np.float64,
            "beam_index": np.int16,
        }.items():
            points.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype)
        points.create_dataset("beam_id", shape=(0,), maxshape=(None,), dtype=UTF8)
        points.create_dataset("result_status", shape=(0,), maxshape=(None,), dtype=UTF8)
        points.create_dataset("error_code", shape=(0,), maxshape=(None,), dtype=UTF8)
        measurements = store.file.create_group("measurements")
        chunks = (1, max(1, store.frequency_count))
        for name in ("real", "imag"):
            measurements.create_dataset(
                name,
                shape=(0, store.frequency_count),
                maxshape=(None, store.frequency_count),
                dtype=np.float64,
                fillvalue=np.nan,
                chunks=chunks,
                compression="gzip",
            )
        store.file.flush()
        return store

    @property
    def frequency_count(self) -> int:
        return int(self.file["frequencies_hz"].shape[0])

    def commit_channel(self, channel: Channel, values: np.ndarray) -> None:
        if values.ndim != 1 or values.size != self.frequency_count or not np.all(np.isfinite(values)):
            raise ServiceError("DATA_INTEGRITY", "通道复数数据不完整或非有限", "data_commit_channel", str(channel.element))
        row = self._channel_rows[channel.element]
        self.file["measurements/real"][row, :] = values.real
        self.file["measurements/imag"][row, :] = values.imag
        self.file.flush()
        if not np.array_equal(self.file["measurements/real"][row, :], values.real) or not np.array_equal(
            self.file["measurements/imag"][row, :], values.imag
        ):
            raise ServiceError("DATA_INTEGRITY", "通道数据写盘读回不一致", "data_verify_channel", str(channel.element))
        self.file["channels/result_status"][row] = "CALIBRATED"
        self.file["channels/error_code"][row] = ""
        self.file.flush()

    def fail_channel(self, channel: Channel, error_code: str) -> None:
        row = self._channel_rows[channel.element]
        self.file["channels/result_status"][row] = "FAILED"
        self.file["channels/error_code"][row] = error_code
        self.file.flush()

    def commit_point(
        self,
        *,
        bundle_id: int,
        point_id: int,
        row_index: int,
        point_index: int,
        azimuth_deg: float,
        elevation_deg: float,
        actual_azimuth_deg: float,
        actual_elevation_deg: float,
        beam_index: int,
        beam_id: str,
        values: np.ndarray,
    ) -> None:
        if values.ndim != 1 or values.size != self.frequency_count or not np.all(np.isfinite(values)):
            raise ServiceError("DATA_INTEGRITY", "空间点复数数据不完整或非有限", "data_commit_point", str(point_id))
        row = int(self.file["spatial_points/point_id"].shape[0])
        point_values = {
            "bundle_id": bundle_id,
            "point_id": point_id,
            "row_index": row_index,
            "point_index": point_index,
            "azimuth_deg": azimuth_deg,
            "elevation_deg": elevation_deg,
            "actual_azimuth_deg": actual_azimuth_deg,
            "actual_elevation_deg": actual_elevation_deg,
            "beam_index": beam_index,
            "beam_id": beam_id,
            # PENDING is written before the complex data. A crash can therefore never
            # leave a COMPLETE row backed by HDF5's default zero fill values.
            "result_status": "PENDING",
            "error_code": "",
        }
        for name, value in point_values.items():
            dataset = self.file[f"spatial_points/{name}"]
            dataset.resize((row + 1,))
            dataset[row] = value
        for name, data in (("real", values.real), ("imag", values.imag)):
            dataset = self.file[f"measurements/{name}"]
            dataset.resize((row + 1, self.frequency_count))
            dataset[row, :] = data
        self.file.flush()
        metadata_matches = (
            int(self.file["spatial_points/bundle_id"][row]) == bundle_id
            and int(self.file["spatial_points/point_id"][row]) == point_id
            and int(self.file["spatial_points/beam_index"][row]) == beam_index
        )
        if (
            not metadata_matches
            or not np.array_equal(self.file["measurements/real"][row, :], values.real)
            or not np.array_equal(self.file["measurements/imag"][row, :], values.imag)
        ):
            raise ServiceError("DATA_INTEGRITY", "方向图数据写盘读回不一致", "data_verify_point", str(bundle_id))
        self.file["spatial_points/result_status"][row] = "COMPLETE"
        self.file.flush()

    def initialize_rtc(self, configuration: dict[str, Any], frames: list[bytes]) -> None:
        group = self.file.create_group("rtc")
        group.create_dataset("wave_frames", data=np.asarray([list(frame) for frame in frames], dtype=np.uint8).reshape((-1, 22)))
        group.attrs["clock_hz"] = configuration["clock_hz"]
        for section in ("tr", "timing", "tr_readback"):
            settings = group.create_group(section)
            for key, value in configuration[section].items():
                settings.attrs[key] = value
        if self.file.attrs["schema_name"] == "antenna-channel-calibration":
            channels = self.file["channels"]
            count = len(channels["element"])
            measurements = group.create_group("channel_measurements")
            for name in ("wave_address", "accepted_groups", "completed_groups", "valid_triggers", "completed_points"):
                measurements.create_dataset(name, data=np.zeros(count, dtype=np.int64))
            measurements.create_dataset("close_status", data=np.array(
                ["PENDING" if enabled else "SKIPPED_DISABLED" for enabled in channels["enabled"][...]], dtype=UTF8))
            for name in ("close_request", "close_response"):
                measurements.create_dataset(name, shape=(count, 22), dtype=np.uint8)
            self.file.flush()
            return
        acquisitions = group.create_group("acquisitions")
        for key in ("row_index", "first_point_index", "point_count", "accepted_groups", "completed_groups",
                    "valid_triggers", "completed_points", "buffer_sweeps", "memory_bytes",
                    "pulse_output_disabled", "endpoint_pulse_count_verified"):
            acquisitions.create_dataset(key, shape=(0,), maxshape=(None,), dtype=np.int64)
        for key in ("start_azimuth_readback_deg", "end_azimuth_readback_deg", "actual_elevation_deg"):
            acquisitions.create_dataset(key, shape=(0,), maxshape=(None,), dtype=np.float64)
        self.file.flush()

    def mark_rtc_channel_close(self, channel: Channel, status: str, *, request: bytes | None = None,
                               response: bytes | None = None) -> None:
        group = self.file["rtc/channel_measurements"]
        row = self._channel_rows[channel.element]
        group["close_status"][row] = status
        for name, frame in (("close_request", request), ("close_response", response)):
            if frame is not None:
                if len(frame) != 22:
                    raise ServiceError("DATA_INTEGRITY", "RTC关闭证据长度错误", "rtc_close_store")
                group[name][row] = np.frombuffer(frame, dtype=np.uint8)
        self.file.flush()

    def commit_rtc_channel(self, channel: Channel, evidence: dict[str, Any]) -> None:
        group = self.file["rtc/channel_measurements"]
        row = self._channel_rows[channel.element]
        for key in ("accepted_groups", "completed_groups", "valid_triggers", "completed_points"):
            group[key][row] = evidence["progress"][key]
        settings = self.file["rtc"].require_group("vna_configuration")
        for key, value in evidence["buffer"].items():
            if key not in {"sweep_count", "memory_bytes"}:
                settings.attrs[key] = _attribute_value(value)
        self.file.flush()

    def commit_rtc_group(self, *, row_index: int, first_point_index: int, point_count: int,
                         evidence: dict[str, Any], start_azimuth: float, end_azimuth: float,
                         actual_elevation: float) -> None:
        group = self.file["rtc/acquisitions"]
        index = len(group["row_index"])
        progress = evidence["progress"]
        values = {
            "row_index": row_index, "first_point_index": first_point_index, "point_count": point_count,
            **{key: progress[key] for key in ("accepted_groups", "completed_groups", "valid_triggers", "completed_points")},
            "buffer_sweeps": evidence["buffer"]["sweep_count"], "memory_bytes": evidence["buffer"]["memory_bytes"],
            "pulse_output_disabled": int(evidence.get("motion", {}).get("pulse_output_disabled", True)),
            "endpoint_pulse_count_verified": int(evidence.get("motion", {}).get("endpoint_pulse_count_verified", False)),
            "start_azimuth_readback_deg": start_azimuth, "end_azimuth_readback_deg": end_azimuth,
            "actual_elevation_deg": actual_elevation,
        }
        for key, value in values.items():
            group[key].resize((index + 1,))
            group[key][index] = value
        settings = self.file["rtc"].require_group("vna_configuration")
        for key, value in evidence["buffer"].items():
            if key not in {"sweep_count", "memory_bytes"}:
                settings.attrs[key] = _attribute_value(value)
        self.file.flush()

    def log(self, level: str, stage: str, message: str, details: dict[str, Any] | None = None) -> None:
        group = self.file["run_events"]
        row = int(group["timestamp"].shape[0])
        values = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "stage": stage,
            "message": message,
            "details": json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
        }
        for name, value in values.items():
            dataset = group[name]
            dataset.resize((row + 1,))
            dataset[row] = value
        self.file.flush()

    def finalize(self, terminal_status: str) -> dict[str, Any]:
        role = "COMPLETE_RESULT" if terminal_status == "COMPLETED" else "RUN_TERMINAL"
        try:
            validate_data_contents(self.file, require_complete=terminal_status == "COMPLETED")
            self.file.attrs["file_role"] = role
            self.file.attrs["status"] = terminal_status
            self.file.attrs["completed_at"] = datetime.now(timezone.utc).isoformat()
            self.file.attrs["content_sha256"] = data_content_sha256(self.file)
            self.file.flush()
        except Exception as exc:
            self.file.attrs["file_role"] = "RUN_TERMINAL"
            self.file.attrs["status"] = "FAULTED"
            self.file.attrs["integrity_error"] = str(exc)
            self.file.flush()
            raise
        finally:
            self.file.close()
        try:
            with h5py.File(self.path, "r") as verify:
                if verify.attrs.get("file_role") != role or verify.attrs.get("status") != terminal_status:
                    raise ServiceError("DATA_INTEGRITY", "HDF5 文件关闭重开复验失败", "data_reopen", str(self.path))
                validate_data_contents(verify, require_complete=terminal_status == "COMPLETED")
                counts = _dataset_counts(verify)
        except (OSError, ServiceError) as exc:
            # A failed reopen/content comparison cannot leave a COMPLETE_RESULT label.
            try:
                with h5py.File(self.path, "r+") as failed:
                    failed.attrs["file_role"] = "RUN_TERMINAL"
                    failed.attrs["status"] = "FAULTED"
                    failed.attrs["integrity_error"] = str(exc)
                    failed.flush()
            except OSError:
                pass  # The original read/validation error remains the reported failure.
            if isinstance(exc, ServiceError):
                raise
            raise ServiceError("DATA_INTEGRITY", "HDF5 文件无法重新打开", "data_reopen", str(self.path), {"error": str(exc)}) from exc
        return {"path": str(self.path), "sha256": file_sha256(self.path), "tables": counts, "status": terminal_status}


def _dataset_counts(file: h5py.File) -> dict[str, int]:
    counts: dict[str, int] = {}
    if "frequencies_hz" in file:
        counts["frequencies"] = int(file["frequencies_hz"].shape[0])
    if "channels" in file:
        counts["channels"] = int(file["channels/element"].shape[0])
        if "measurements" in file:
            counts["channel_measurements"] = int(np.isfinite(file["measurements/real"][...]).sum())
    if "compensation" in file:
        counts["compensation"] = int(np.prod(file["compensation/phase_code"].shape))
    if "flash_items" in file:
        counts["flash_items"] = len(file["flash_items"])
        counts["flash_payload_bytes"] = sum(int(file[f"flash_items/{name}"].attrs["occupied_length"]) for name in file["flash_items"])
    if "spatial_points" in file:
        counts["spatial_points"] = int(file["spatial_points/point_id"].shape[0])
        counts["electronic_beams"] = int(file["beam_definitions/beam_id"].shape[0]) if "beam_definitions" in file else 0
        counts["pattern_measurements"] = int(np.isfinite(file["measurements/real"][...]).sum())
    if "run_events" in file:
        counts["run_events"] = int(file["run_events/timestamp"].shape[0])
    return counts


def _analysis_options(file: h5py.File) -> dict[str, Any]:
    schema_name = str(_decode_attribute(file.attrs.get("schema_name", "")))
    frequencies = file["frequencies_hz"][...].astype(float).tolist() if "frequencies_hz" in file else []
    if schema_name == "antenna-pattern-run":
        beams = [
            {
                "beam_index": index,
                "beam_id": beam_id,
                "off_axis_deg": float(file["beam_definitions/off_axis_deg"][index]),
                "azimuth_deg": float(file["beam_definitions/azimuth_deg"][index]),
            }
            for index, beam_id in enumerate(file["beam_definitions/beam_id"].asstr()[...].tolist())
        ]
        return {"kind": "PATTERN", "frequencies_hz": frequencies, "beams": beams}
    if schema_name == "antenna-channel-calibration":
        return {"kind": "CALIBRATION", "frequencies_hz": frequencies, "beams": []}
    return {"kind": "UNSUPPORTED", "frequencies_hz": frequencies, "beams": []}


def _magnitude_phase(real: float, imag: float) -> tuple[float | None, float | None, float | None]:
    if not math.isfinite(real) or not math.isfinite(imag):
        return None, None, None
    amplitude = math.hypot(real, imag)
    if amplitude <= 0:
        return 0.0, None, None
    return amplitude, 20 * math.log10(amplitude), math.degrees(math.atan2(imag, real)) % 360


def read_data_view(path: str, frequency_index: int, beam_index: int = 0) -> dict[str, Any]:
    """Read one frequency slice for interactive history analysis without loading the full HDF."""
    resolved = Path(path).resolve()
    if not resolved.is_file() or not h5py.is_hdf5(resolved):
        raise ServiceError("DATA_INTEGRITY", "不是有效的 HDF5 天线数据文件", "data_view", str(resolved))
    try:
        with h5py.File(resolved, "r") as file:
            options = _analysis_options(file)
            frequencies = options["frequencies_hz"]
            if options["kind"] not in {"CALIBRATION", "PATTERN"}:
                raise ServiceError("INVALID_REQUEST", "该 HDF 类型暂不支持图形分析", "data_view", str(resolved))
            if frequency_index < 0 or frequency_index >= len(frequencies):
                raise ServiceError("INVALID_REQUEST", "频点索引超出数据范围", "data_view", str(frequency_index))
            real_values = file["measurements/real"][:, frequency_index]
            imag_values = file["measurements/imag"][:, frequency_index]
            if options["kind"] == "CALIBRATION":
                statuses = file["channels/result_status"].asstr()[...]
                channels: list[dict[str, Any]] = []
                calibrated_db: list[float] = []
                for index, element in enumerate(file["channels/element"][...]):
                    amplitude, magnitude_db, phase_deg = _magnitude_phase(float(real_values[index]), float(imag_values[index]))
                    if statuses[index] != "CALIBRATED":
                        amplitude, magnitude_db, phase_deg = None, None, None
                    if statuses[index] == "CALIBRATED" and magnitude_db is not None:
                        calibrated_db.append(magnitude_db)
                    channels.append(
                        {
                            "element": int(element),
                            "grid_row": int(file["channels/grid_row"][index]),
                            "grid_column": int(file["channels/grid_column"][index]),
                            "status": str(statuses[index]),
                            "amplitude_linear": amplitude,
                            "magnitude_db": magnitude_db,
                            "phase_deg": phase_deg,
                        }
                    )
                return {
                    "kind": "CALIBRATION",
                    "frequency_index": frequency_index,
                    "frequency_hz": frequencies[frequency_index],
                    "max_amplitude_difference_db": max(calibrated_db) - min(calibrated_db) if calibrated_db else None,
                    "channels": channels,
                }
            if options["kind"] != "PATTERN":
                raise ServiceError("INVALID_REQUEST", "该 HDF 类型暂不支持图形分析", "data_view", str(resolved))
            available_beams = options["beams"]
            if beam_index < -1 or beam_index >= len(available_beams):
                raise ServiceError("INVALID_REQUEST", "电子波束索引超出数据范围", "data_view", str(beam_index))
            stored_beam_indexes = file["spatial_points/beam_index"][...]
            rows = np.arange(real_values.size) if beam_index == -1 else np.flatnonzero(stored_beam_indexes == beam_index)
            statuses = file["spatial_points/result_status"].asstr()[...]
            rows = rows[statuses[rows] == "COMPLETE"]
            points: list[dict[str, Any]] = []
            for row in rows:
                point_beam_index = int(stored_beam_indexes[row])
                _, magnitude_db, phase_deg = _magnitude_phase(float(real_values[row]), float(imag_values[row]))
                points.append(
                    {
                        "row_index": int(file["spatial_points/row_index"][row]),
                        "point_index": int(file["spatial_points/point_index"][row]),
                        "azimuth_deg": float(file["spatial_points/azimuth_deg"][row]),
                        "elevation_deg": float(file["spatial_points/elevation_deg"][row]),
                        "beam_index": point_beam_index,
                        "beam_id": available_beams[point_beam_index]["beam_id"],
                        "magnitude_db": magnitude_db,
                        "phase_deg": phase_deg,
                    }
                )
            return {
                "kind": "PATTERN",
                "frequency_index": frequency_index,
                "frequency_hz": frequencies[frequency_index],
                "beam": available_beams[beam_index] if beam_index >= 0 else None,
                "beams": available_beams,
                "points": points,
            }
    except ServiceError:
        raise
    except (OSError, KeyError, ValueError) as exc:
        raise ServiceError("DATA_INTEGRITY", "HDF5 分析数据结构不完整", "data_view", str(resolved), {"error": str(exc)}) from exc


def inspect_data_file(path: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ServiceError("NOT_FOUND", "数据文件不存在", "data_inspect", str(resolved))
    if not h5py.is_hdf5(resolved):
        raise ServiceError("DATA_INTEGRITY", "不是有效的 HDF5 天线数据文件", "data_inspect", str(resolved))
    try:
        with h5py.File(resolved, "r") as file:
            checksum_verified = validate_data_contents(file)
            schema = {key: _decode_attribute(value) for key, value in file.attrs.items()}
            metadata = {key: _decode_attribute(value) for key, value in file["metadata"].attrs.items()} if "metadata" in file else {}
            counts = _dataset_counts(file)
            analysis = _analysis_options(file)
    except (OSError, KeyError) as exc:
        raise ServiceError("DATA_INTEGRITY", "HDF5 天线数据结构不完整", "data_inspect", str(resolved), {"error": str(exc)}) from exc
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "integrity": "ok" if checksum_verified else "content_validated",
        "checksum_verified": checksum_verified,
        "schema": schema,
        "metadata": metadata,
        "tables": counts,
        "analysis": analysis,
    }
