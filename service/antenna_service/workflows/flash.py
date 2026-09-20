from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from antenna_service.coordinates import Channel, CoordinateModel
from antenna_service.devices.manager import DeviceManager
from antenna_service.errors import ServiceError
from antenna_service.models import FlashPrepareRequest, FlashWriteRequest
from antenna_service.protocol.flash import LAST_PAGE_ADDRESS, MAX_ADDRESS, PAGE_SIZE
from antenna_service.storage.hdf5_store import file_sha256, validate_data_contents


ITEM_ORDER = ["ARRAY_ID", "SWITCH_TABLE", "COORDINATE", "TX_COMPENSATION", "RX_COMPENSATION"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FlashService:
    """Encode the five firmware payloads, persist them in HDF5 and write by page."""

    def prepare(self, request: FlashPrepareRequest, coordinates: CoordinateModel) -> dict[str, Any]:
        tx, tx_frequencies, tx_channels, tx_evidence = self._read_compensation(request.tx_compensation_file, "TX", coordinates)
        rx, rx_frequencies, rx_channels, rx_evidence = self._read_compensation(request.rx_compensation_file, "RX", coordinates)
        data_evidence = self._combined_evidence(coordinates.evidence, tx_evidence, rx_evidence)
        if not np.array_equal(tx_frequencies, rx_frequencies):
            raise ServiceError("DATA_INTEGRITY", "TX 与 RX 补偿频点必须完全一致", "flash_prepare", "frequencies_hz")
        if tx_channels != rx_channels:
            raise ServiceError("DATA_INTEGRITY", "TX 与 RX 补偿通道集合或顺序不一致", "flash_prepare", "channels")

        effective_payloads = {
            "ARRAY_ID": bytes((request.array_id,)),
            "SWITCH_TABLE": self._switch_payload(coordinates),
            "COORDINATE": self._coordinate_payload(coordinates),
            "TX_COMPENSATION": tx,
            "RX_COMPENSATION": rx,
        }
        payloads = {name: self._pad_to_pages(payload) for name, payload in effective_payloads.items()}
        self._validate_address_layout(request.start_addresses, payloads)

        output = Path(request.output_path).resolve()
        if output.suffix.lower() not in {".hdf5", ".h5", ".hdf"}:
            output = output.with_suffix(".hdf5")
        if output.exists():
            raise ServiceError("DATA_INTEGRITY", "FLASH HDF5 输出文件已存在", "flash_prepare", str(output))
        output.parent.mkdir(parents=True, exist_ok=True)

        summaries: list[dict[str, Any]] = []
        with h5py.File(output, "w") as package:
            package.attrs.update(
                {
                    "schema_name": "antenna-flash-package",
                    "schema_version": "2.0",
                    "file_role": "FLASH_PACKAGE",
                    "payload_layout": "FLASH_INTERNAL_OFFICIAL_V1",
                    "status": "PREPARED",
                    "created_at": _utc_now(),
                    "antenna_id": coordinates.antenna_id,
                    "tile_id": request.array_id,
                    "page_size": PAGE_SIZE,
                    "data_evidence": data_evidence,
                    "evidence": data_evidence,
                }
            )
            package.create_dataset("frequencies_hz", data=tx_frequencies)
            sources = package.create_group("source_files")
            for role, source_path, sha256, evidence in (
                ("COORDINATES", coordinates.path, coordinates.file_sha256, coordinates.evidence),
                ("TX_COMPENSATION", str(Path(request.tx_compensation_file).resolve()), file_sha256(request.tx_compensation_file), tx_evidence),
                ("RX_COMPENSATION", str(Path(request.rx_compensation_file).resolve()), file_sha256(request.rx_compensation_file), rx_evidence),
            ):
                source = sources.create_group(role)
                source.attrs["path"] = source_path
                source.attrs["sha256"] = sha256
                source.attrs["evidence"] = evidence

            items = package.create_group("flash_items", track_order=True)
            for name in ITEM_ORDER:
                effective = effective_payloads[name]
                payload = payloads[name]
                start = int(request.start_addresses[name])
                occupied_length = len(payload)
                item = items.create_group(name)
                item.attrs.update(
                    {
                        "start_address": start,
                        "end_address": start + occupied_length - 1,
                        "effective_length": len(effective),
                        "occupied_length": occupied_length,
                        "padding_length": occupied_length - len(effective),
                        "page_size": PAGE_SIZE,
                        "page_count": occupied_length // PAGE_SIZE,
                        "effective_sha256": hashlib.sha256(effective).hexdigest(),
                        "payload_sha256": hashlib.sha256(payload).hexdigest(),
                        "write_status": "PREPARED",
                        "readback_sha256": "",
                        "data_evidence": data_evidence,
                        "evidence": data_evidence,
                        "writer_source": "",
                        "writer_identity": "",
                        "written_pages": 0,
                    }
                )
                # This dataset is the exact page-aligned stream sent to the device.
                # Addresses and CRC stay in the serial protocol layer, not in page data.
                item.create_dataset("payload", data=np.frombuffer(payload, dtype=np.uint8), compression="gzip")
                summaries.append(self._item_summary(name, item))
            package.flush()

        if not h5py.is_hdf5(output):
            raise ServiceError("DATA_INTEGRITY", "FLASH HDF5 文件关闭重开失败", "flash_prepare", str(output))
        with h5py.File(output, "r") as verify:
            self._validate_package_identity(verify, output)
            if set(verify["flash_items"].keys()) != set(ITEM_ORDER):
                raise ServiceError("DATA_INTEGRITY", "FLASH HDF5 缺少数据项目", "flash_prepare", str(output))
        return {"path": str(output), "sha256": file_sha256(output), "tile_id": request.array_id,
                "data_evidence": data_evidence, "evidence": data_evidence, "items": summaries}

    async def write_item(self, request: FlashWriteRequest, devices: DeviceManager) -> dict[str, Any]:
        path = Path(request.package_path).resolve()
        if not path.is_file() or not h5py.is_hdf5(path):
            raise ServiceError("NOT_FOUND", "FLASH HDF5 包不存在或格式无效", "flash_write", str(path))

        # Close HDF5 before hardware awaits so the file is not locked for the whole operation.
        with h5py.File(path, "r") as package:
            self._validate_package_identity(package, path)
            unsafe_items = [name for name, item in package["flash_items"].items()
                            if item.attrs.get("write_status") in {"WRITING", "UNKNOWN", "PARTIAL_WRITE", "VERIFY_FAILED"}]
            if unsafe_items:
                raise ServiceError(
                    "NOT_RUNNABLE", "FLASH 工程包含未确认或失败写入，必须先人工恢复", "flash_write", str(path),
                    {"items": unsafe_items}, side_effect_possible=True,
                    next_action="核对目标区域并完成现场恢复后，重新准备工程包；不要直接重写历史项目",
                )
            item_path = f"flash_items/{request.item_name}"
            if item_path not in package:
                raise ServiceError("NOT_FOUND", "FLASH 项目不存在", "flash_write", request.item_name)
            item = package[item_path]
            tile_id = int(package.attrs["tile_id"])
            start = int(item.attrs["start_address"])
            occupied_length = int(item.attrs["occupied_length"])
            effective_length = int(item.attrs["effective_length"])
            expected_sha = str(item.attrs["payload_sha256"])
            payload = np.asarray(item["payload"], dtype=np.uint8).tobytes()
        if len(payload) != occupied_length or occupied_length % PAGE_SIZE or hashlib.sha256(payload).hexdigest() != expected_sha:
            raise ServiceError("DATA_INTEGRITY", "FLASH HDF5 页载荷长度或摘要不一致", "flash_write", request.item_name)

        device = devices.require(request.device_id)
        writer_status = await device.status()
        if writer_status.get("source") not in {"REAL", "SIMULATED"}:
            raise ServiceError("DATA_INTEGRITY", "FLASH 写入设备缺少来源证据", "flash_write", request.device_id)
        # Persist intent before the first physical write. An interrupted process leaves
        # WRITING on disk, which is treated as an unconfirmed write on the next request.
        self._set_item_status(path, request.item_name, "WRITING", writer_source=writer_status["source"],
                              writer_identity=str(writer_status.get("identity") or ""), writer_device_id=request.device_id,
                              write_started_at=_utc_now(), write_completed_at="", written_pages=0, last_error="")
        try:
            return await self._write_pages(request, device, path, tile_id, start, payload, effective_length, expected_sha, writer_status)
        except (Exception, asyncio.CancelledError) as exc:
            try:
                with h5py.File(path, "r") as package:
                    still_writing = package[f"flash_items/{request.item_name}"].attrs.get("write_status") == "WRITING"
                if still_writing:
                    self._set_item_status(path, request.item_name, "UNKNOWN", last_error=str(exc) or type(exc).__name__)
            except Exception as persistence_error:
                # WRITING was durably stored before sending; even if this later update
                # fails, another request must not interpret the package as untouched.
                if isinstance(exc, asyncio.CancelledError):
                    raise exc
                raise ServiceError("DATA_INTEGRITY", "FLASH 写入中断且状态记录失败，目标状态未知", "flash_write", request.item_name,
                                   {"error": str(exc), "persistence_error": str(persistence_error)}, side_effect_possible=True) from exc
            if isinstance(exc, (ServiceError, asyncio.CancelledError)):
                raise
            raise ServiceError("DEVICE_FAULT", "FLASH 写入中断，目标状态未知", "flash_write", request.item_name,
                               {"error": str(exc)}, side_effect_possible=True) from exc

    async def _write_pages(
        self, request: FlashWriteRequest, device: Any, path: Path, tile_id: int, start: int,
        payload: bytes, effective_length: int, expected_sha: str, writer_status: dict[str, Any],
    ) -> dict[str, Any]:
        occupied_length = len(payload)
        recovered_pages = 0
        for offset in range(0, occupied_length, PAGE_SIZE):
            page = payload[offset : offset + PAGE_SIZE]
            address = start + offset
            self._set_item_status(path, request.item_name, "WRITING", current_address=address, written_pages=offset // PAGE_SIZE)
            try:
                # One page-write request is sent at most once. If its acknowledgement is
                # missing, only a read of this same tile/address may resolve the uncertainty.
                await getattr(device, "flash_write")(tile_id, address, page)
            except Exception as write_error:
                try:
                    recovered = await self._read_page(device, tile_id, address)
                except Exception as read_error:
                    self._set_item_status(path, request.item_name, "UNKNOWN", last_error=f"write: {write_error}; read: {read_error}")
                    raise ServiceError(
                        "DEVICE_FAULT",
                        "FLASH 页写结果未知，同址读回也失败",
                        "flash_write_recovery",
                        request.item_name,
                        {"tile_id": tile_id, "address": address, "write_error": str(write_error), "read_error": str(read_error)},
                        side_effect_possible=True,
                        next_action="停止新写入并人工检查目标页；不要自动重写",
                    ) from read_error
                if recovered != page:
                    self._set_item_status(path, request.item_name, "PARTIAL_WRITE", last_error=str(write_error))
                    raise ServiceError(
                        "DATA_INTEGRITY",
                        "FLASH 页写应答不确定且同址读回不匹配",
                        "flash_write_recovery",
                        request.item_name,
                        {"tile_id": tile_id, "address": address, "write_error": str(write_error)},
                        side_effect_possible=True,
                        next_action="停止新写入并人工恢复目标区域",
                    ) from write_error
                recovered_pages += 1
            self._set_item_status(path, request.item_name, "WRITING", written_pages=offset // PAGE_SIZE + 1)

        # After all page writes, read the complete padded item from beginning to end.
        readback = bytearray()
        for offset in range(0, occupied_length, PAGE_SIZE):
            address = start + offset
            try:
                page = await self._read_page(device, tile_id, address)
            except Exception as exc:
                self._set_item_status(path, request.item_name, "UNKNOWN", current_address=address, last_error=str(exc))
                raise ServiceError(
                    "DEVICE_FAULT",
                    "FLASH 整项读回失败，物理状态未知",
                    "flash_verify",
                    request.item_name,
                    {"tile_id": tile_id, "address": address, "error": str(exc)},
                    side_effect_possible=True,
                ) from exc
            expected_page = payload[offset : offset + PAGE_SIZE]
            if page != expected_page:
                self._set_item_status(path, request.item_name, "VERIFY_FAILED", current_address=address, last_error="读回与页载荷不一致")
                raise ServiceError(
                    "DATA_INTEGRITY",
                    "FLASH 整项读回逐字节不一致",
                    "flash_verify",
                    request.item_name,
                    {"tile_id": tile_id, "address": address},
                    side_effect_possible=True,
                )
            readback.extend(page)

        readback_sha = hashlib.sha256(readback).hexdigest()
        if readback_sha != expected_sha:
            self._set_item_status(path, request.item_name, "VERIFY_FAILED")
            raise ServiceError("DATA_INTEGRITY", "FLASH 整项读回摘要不一致", "flash_verify", request.item_name)
        self._set_item_status(path, request.item_name, "SUCCESS", readback_sha, write_completed_at=_utc_now())
        return {
            "item_name": request.item_name,
            "status": "SUCCESS",
            "tile_id": tile_id,
            "start_address": start,
            "effective_length": effective_length,
            "occupied_length": occupied_length,
            "recovered_pages": recovered_pages,
            "readback_sha256": readback_sha,
            "device": writer_status,
            "writer_source": writer_status["source"],
        }

    @staticmethod
    async def _read_page(device: Any, tile_id: int, address: int) -> bytes:
        """Read one page with at most one retry; page writes are never retried."""

        for attempt in range(2):
            try:
                data = await getattr(device, "flash_read")(tile_id, address, PAGE_SIZE)
                if len(data) != PAGE_SIZE:
                    raise ServiceError("DATA_INTEGRITY", "FLASH 页读回长度不是 256 字节", "flash_read", hex(address))
                return data
            except ServiceError as exc:
                # Only a timeout proves that no valid read response arrived and permits
                # one more read request. CRC, identity, reserved-byte and length errors
                # are data-integrity failures and are not retried automatically.
                if exc.code != "TIMEOUT" or attempt == 1:
                    raise
            except Exception:
                raise
        raise AssertionError("unreachable")

    def list_items(self, package_path: str) -> list[dict[str, Any]]:
        path = Path(package_path).resolve()
        if not path.is_file() or not h5py.is_hdf5(path):
            raise ServiceError("DATA_INTEGRITY", "不是有效的 FLASH HDF5 包", "flash_items", str(path))
        with h5py.File(path, "r") as package:
            self._validate_package_identity(package, path)
            return [self._item_summary(name, package[f"flash_items/{name}"]) for name in ITEM_ORDER]

    @staticmethod
    def _combined_evidence(*sources: str) -> str:
        return sources[0] if len(set(sources)) == 1 else "MIXED"

    def _set_item_status(self, path: Path, item_name: str, status: str, readback_sha256: str = "", **details: Any) -> None:
        with h5py.File(path, "r+") as package:
            item = package[f"flash_items/{item_name}"]
            item.attrs["write_status"] = status
            item.attrs["readback_sha256"] = readback_sha256
            item.attrs.update(details)
            if item.attrs.get("writer_source"):
                item.attrs["evidence"] = self._combined_evidence(str(item.attrs["data_evidence"]), str(item.attrs["writer_source"]))
            writer_sources = [str(group.attrs["writer_source"]) for group in package["flash_items"].values() if group.attrs.get("writer_source")]
            package.attrs["evidence"] = self._combined_evidence(str(package.attrs["data_evidence"]), *writer_sources)
            package.flush()

    @staticmethod
    def _validate_package_identity(package: h5py.File, path: Path) -> None:
        identity = (package.attrs.get("schema_name"), package.attrs.get("schema_version"), package.attrs.get("payload_layout"))
        if identity != ("antenna-flash-package", "2.0", "FLASH_INTERNAL_OFFICIAL_V1"):
            raise ServiceError("DATA_INTEGRITY", "FLASH HDF5 包身份或载荷布局不正确", "flash_package", str(path), {"schema": identity})
        validate_data_contents(package)

    @staticmethod
    def _item_summary(name: str, item: h5py.Group) -> dict[str, Any]:
        return {
            "item_name": name,
            "start_address": int(item.attrs["start_address"]),
            "end_address": int(item.attrs["end_address"]),
            "effective_length": int(item.attrs["effective_length"]),
            "occupied_length": int(item.attrs["occupied_length"]),
            "padding_length": int(item.attrs["padding_length"]),
            "page_count": int(item.attrs["page_count"]),
            "sha256": str(item.attrs["payload_sha256"]),
            "status": str(item.attrs["write_status"]),
            "readback_sha256": str(item.attrs.get("readback_sha256", "")) or None,
            "data_evidence": str(item.attrs["data_evidence"]),
            "evidence": str(item.attrs["evidence"]),
            "writer_source": str(item.attrs.get("writer_source", "")) or None,
            "writer_identity": str(item.attrs.get("writer_identity", "")) or None,
            "written_pages": int(item.attrs.get("written_pages", 0)),
        }

    def _read_compensation(
        self, path: str, expected_path: str, coordinates: CoordinateModel
    ) -> tuple[bytes, np.ndarray, tuple[tuple[str, int, int, int], ...], str]:
        resolved = Path(path).resolve()
        if not resolved.is_file() or not h5py.is_hdf5(resolved):
            raise ServiceError("DATA_INTEGRITY", f"{expected_path} 补偿输入必须是 HDF5 文件", "flash_prepare", str(resolved))
        with h5py.File(resolved, "r") as connection:
            validate_data_contents(connection, require_complete=True)
            evidence = str(connection.attrs["evidence"])
            schema = (connection.attrs.get("schema_name"), connection.attrs.get("status"), connection.attrs.get("signal_path"))
            if schema != ("antenna-compensation", "COMPLETED", expected_path):
                raise ServiceError("DATA_INTEGRITY", f"{expected_path} 补偿文件身份不正确", "flash_prepare", str(resolved), {"schema": schema})
            provenance = (
                connection.attrs.get("coordinate_sha256"),
                connection.attrs.get("mapping_sha256"),
                connection.attrs.get("enabled_sha256"),
                connection.attrs.get("geometry_sha256"),
            )
            expected = (coordinates.file_sha256, coordinates.mapping_sha256, coordinates.enabled_sha256, coordinates.geometry_sha256)
            if provenance != expected:
                raise ServiceError("DATA_INTEGRITY", f"{expected_path} 补偿与坐标表不一致", "flash_prepare", str(resolved))
            frequencies = np.asarray(connection["frequencies_hz"], dtype=np.float64)
            polarizations = connection["channels/polarization"].asstr()[...]
            spi_numbers = np.asarray(connection["channels/spi_no"], dtype=np.int64)
            chip_numbers = np.asarray(connection["channels/chip_no"], dtype=np.int64)
            chip_channels = np.asarray(connection["channels/chip_channel_index"], dtype=np.int64)
            enabled = np.asarray(connection["channels/enabled"], dtype=bool)
            phase_codes = np.asarray(connection["compensation/phase_code"], dtype=np.int64)
            attenuation_codes = np.asarray(connection["compensation/attenuation_code"], dtype=np.int64)

        channel_count = len(polarizations)
        if frequencies.ndim != 1 or not len(frequencies) or not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
            raise ServiceError("DATA_INTEGRITY", f"{expected_path} 补偿频点无效", "flash_prepare", str(resolved))
        if phase_codes.shape != attenuation_codes.shape or phase_codes.shape != (channel_count, len(frequencies)):
            raise ServiceError("DATA_INTEGRITY", f"{expected_path} 补偿数据集维度不一致", "flash_prepare", str(resolved))
        if any(len(values) != channel_count for values in (spi_numbers, chip_numbers, chip_channels, enabled)):
            raise ServiceError("DATA_INTEGRITY", f"{expected_path} 补偿通道字段长度不一致", "flash_prepare", str(resolved))
        if np.any(phase_codes[enabled] < 0) or np.any(phase_codes[enabled] > 63) or np.any(attenuation_codes[enabled] < 0) or np.any(attenuation_codes[enabled] > 63):
            raise ServiceError("DATA_INTEGRITY", f"{expected_path} 相位码或衰减码超出 0..63", "flash_prepare", str(resolved))

        signatures = [
            (str(polarizations[index]).upper(), int(chip_numbers[index]), int(chip_channels[index]), int(spi_numbers[index]))
            for index in range(channel_count)
        ]
        expected_signatures = {
            (channel.polarization, channel.chip_no, channel.chip_channel_index, channel.spi_no)
            for channel in coordinates.channels
        }
        if len(set(signatures)) != channel_count or set(signatures) != expected_signatures:
            raise ServiceError("DATA_INTEGRITY", f"{expected_path} 补偿通道与坐标表不一致", "flash_prepare", str(resolved))
        enabled_by_mapping = {(channel.polarization, channel.chip_no, channel.chip_channel_index, channel.spi_no): channel.enabled
                              for channel in coordinates.channels}
        if any(bool(enabled[index]) != enabled_by_mapping[signature] for index, signature in enumerate(signatures)):
            raise ServiceError("DATA_INTEGRITY", f"{expected_path} 补偿通道使能与坐标表不一致", "flash_prepare", str(resolved))

        frequency_order = np.argsort(frequencies, kind="stable")
        channel_order: list[int] = []
        for polarization in ("H", "V"):
            channel_order.extend(
                sorted(
                    (index for index, signature in enumerate(signatures) if signature[0] == polarization),
                    key=lambda index: (chip_numbers[index], chip_channels[index], spi_numbers[index]),
                )
            )
        payload = bytearray()
        for frequency_index in frequency_order:
            for codes in (phase_codes, attenuation_codes):
                for channel_index in channel_order:
                    payload.append(int(codes[channel_index, frequency_index]) if enabled[channel_index] else 0)
        ordered_signatures = tuple(signatures[index] for index in channel_order)
        return bytes(payload), frequencies[frequency_order], ordered_signatures, evidence

    @staticmethod
    def _pad_to_pages(payload: bytes) -> bytes:
        if not payload:
            raise ServiceError("DATA_INTEGRITY", "FLASH 有效载荷不能为空", "flash_prepare")
        occupied_length = ((len(payload) + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
        return payload.ljust(occupied_length, b"\x00")

    @staticmethod
    def _validate_address_layout(addresses: dict[str, int], payloads: dict[str, bytes]) -> None:
        """Check per-item encodability only; the operator owns partition overlap safety."""

        for name in ITEM_ORDER:
            start = int(addresses[name])
            if start < 0 or start > LAST_PAGE_ADDRESS or start % PAGE_SIZE:
                raise ServiceError(
                    "INVALID_REQUEST",
                    "FLASH 起始地址必须在 0x000000..0xFFFF00 内并按 0x100 对齐",
                    "flash_prepare",
                    name,
                    {"start_address": start},
                )
            end = start + len(payloads[name]) - 1
            if end > MAX_ADDRESS:
                raise ServiceError("INVALID_REQUEST", "FLASH 项目末页超出 24 位地址范围", "flash_prepare", name, {"end_address": end})

    @staticmethod
    def _switch_payload(coordinates: CoordinateModel) -> bytes:
        payload = bytearray()
        for polarization in ("H", "V"):
            channels = sorted(
                (channel for channel in coordinates.channels if channel.polarization == polarization),
                key=lambda channel: (channel.chip_no, channel.spi_no, channel.chip_channel_index),
            )
            for offset in range(0, len(channels), 8):
                packed = 0
                for bit, channel in enumerate(channels[offset : offset + 8]):
                    packed |= int(channel.enabled) << bit
                payload.append(packed)
        return bytes(payload)

    @staticmethod
    def _coordinate_payload(coordinates: CoordinateModel) -> bytes:
        ordered: list[Channel] = []
        for polarization in ("H", "V"):
            ordered.extend(
                sorted(
                    (channel for channel in coordinates.channels if channel.polarization == polarization),
                    key=lambda channel: (channel.chip_no, channel.chip_channel_index, channel.spi_no),
                )
            )
        if any(channel.grid_row > 255 or channel.grid_column > 255 for channel in ordered):
            raise ServiceError("DATA_INTEGRITY", "FLASH 行号和列号必须在 0..255 范围内", "flash_prepare", "COORDINATE")
        return bytes(channel.grid_row for channel in ordered) + bytes(channel.grid_column for channel in ordered)
