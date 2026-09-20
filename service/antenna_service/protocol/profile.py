from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from antenna_service.errors import ServiceError, invalid
from antenna_service.protocol.crc import append_crc_be, verify_crc_be


EXPECTED_SHEETS = [
    "00_使用说明",
    "01_基本信息",
    "02_指令",
    "03_发送字段",
    "04_接收解析",
    "05_枚举位域",
    "06_测试向量",
]

RESPONSE_RULES_MARKER = "A. 应答组帧规则"
RESPONSE_FIELDS_MARKER = "B. 应答字段"
ENUMS_MARKER = "A. 枚举映射"
BITS_MARKER = "B. 位域布局（可选）"
BASIC_HEADERS = ("配置项", "值", "必填", "可填写内容、含义与来源", "对组帧/运行的影响")
COMMAND_HEADERS = ("启用发送", "CommandId", "显示名称", "自动业务角色", "OpcodeHEX", "应答模式", "应答Opcode", "超时", "风险", "成功判定", "成功字段", "成功值HEX", "返回即稳定", "用途与副作用说明")
SEND_HEADERS = ("启用", "CommandId", "FieldKey", "显示名称", "起始字节", "字节数", "数据类型", "Scale", "单位", "最小值", "最大值", "值来源", "默认/常量/引用", "AutoKey", "EnumId", "用途、可选值及组帧影响")
ENUM_HEADERS = ("启用", "MappingId", "线路值HEX", "逻辑值", "显示文本", "用途、来源及组帧影响")
BIT_HEADERS = ("启用", "LayoutId", "MemberKey", "首位", "位数", "值来源", "值或映射键", "EnumId", "位序及用途说明")
VECTOR_HEADERS = ("启用", "VectorId", "方向", "CommandId", "用途", "阵面ID", "输入参数", "期望帧HEX", "期望结果", "来源说明")
RESPONSE_RULE_HEADERS = (
    "启用",
    "RuleId",
    "请求CommandId",
    "匹配Opcode",
    "组帧模式",
    "预计帧数",
    "总帧数字节",
    "当前帧数字节",
    "数据起始字节",
    "最大帧数",
    "要求顺序",
    "超时",
    "规则说明",
)
RESPONSE_FIELD_HEADERS = (
    "启用",
    "RuleId",
    "帧选择",
    "FieldKey",
    "显示名称",
    "起始字节",
    "字节数",
    "数据类型",
    "Scale",
    "小数位",
    "单位",
    "EnumId",
    "AutoKey",
)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _enabled(value: Any) -> bool:
    return _text(value).upper() == "Y"


def _header_text(value: Any) -> str:
    return _text(value).split("\n", 1)[0].strip()


def _header_row(sheet: Any, headers: tuple[str, ...], start: int = 1, end: int | None = None) -> int:
    rows = [row for row in range(start, (end or sheet.max_row + 1)) if _header_text(sheet.cell(row, 1).value) == headers[0]]
    if len(rows) != 1:
        raise invalid("表头缺失或重复", stage="profile_load", target=f"{sheet.title}!A:A", expected=headers[0])
    row = rows[0]
    for column, expected in enumerate(headers, 1):
        if _header_text(sheet.cell(row, column).value) != expected:
            raise invalid(f"表头错误，应为 {expected}", stage="profile_load", target=f"{sheet.title}!{sheet.cell(row, column).coordinate}")
    return row


def _marker_row(sheet: Any, marker: str) -> int:
    rows = [row for row in range(1, sheet.max_row + 1) if _text(sheet.cell(row, 1).value) == marker]
    if len(rows) != 1:
        raise invalid(f"章节标记缺失或重复：{marker}", stage="profile_load", target=f"{sheet.title}!A:A")
    return rows[0]


def _active_rows(sheet: Any, start: int, end: int | None = None):
    for row in range(start, end or sheet.max_row + 1):
        flag = _text(sheet.cell(row, 1).value).upper()
        if not flag:
            if any(sheet.cell(row, col).value not in (None, "") for col in range(2, sheet.max_column + 1)):
                raise invalid("数据行必须填写启用 Y/N", stage="profile_load", target=f"{sheet.title}!A{row}")
            continue
        if flag not in {"Y", "N"}:
            raise invalid("启用只能填写 Y 或 N", stage="profile_load", target=f"{sheet.title}!A{row}")
        if flag == "Y":
            yield row


def _integer(value: Any, location: str, minimum: int = 0) -> int:
    try:
        number = float(value)
        if isinstance(value, bool) or not math.isfinite(number) or not number.is_integer() or number < minimum:
            raise ValueError()
        return int(number)
    except (TypeError, ValueError, OverflowError) as exc:
        raise invalid(f"必须是大于等于 {minimum} 的整数", stage="profile_load", target=location) from exc


def _field_type(data_type: str, length: int, location: str, *, sending: bool = False) -> None:
    kind = data_type.lower()
    match = re.fullmatch(r"u?int(8|16|24|32|64)|bytes([1-9]\d*)", kind)
    if match:
        width = int(match.group(1)) // 8 if match.group(1) else int(match.group(2))
        if length != width:
            raise invalid("数据类型与字节数不一致", stage="profile_load", target=location, data_type=kind, length=length)
    elif kind not in {"bytes", "ascii", "enum"} | ({"constant_hex", "raw_hex", "bitfield"} if sending else set()):
        raise invalid("不支持的数据类型", stage="profile_load", target=location, data_type=kind)


def _hex_bytes(value: Any, *, location: str, expected: int | None = None) -> bytes:
    if isinstance(value, bytes):
        result = value
    elif isinstance(value, (bytearray, list, tuple)):
        result = bytes(value)
    elif isinstance(value, int):
        if value < 0:
            raise invalid("HEX 值不能为负数", stage="profile", target=location)
        width = expected or max(1, math.ceil(value.bit_length() / 8))
        result = value.to_bytes(width, "big")
    else:
        clean = re.sub(r"0x|[\s,]", "", _text(value), flags=re.IGNORECASE)
        if not clean or len(clean) % 2 or not re.fullmatch(r"[0-9A-Fa-f]+", clean):
            raise invalid("HEX 必须由完整字节组成", stage="profile", target=location, value=value)
        result = bytes.fromhex(clean)
    if expected is not None and len(result) != expected:
        raise invalid(
            f"HEX 长度应为 {expected} 字节，实际为 {len(result)} 字节",
            stage="profile",
            target=location,
        )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Command:
    command_id: str
    display_name: str
    auto_role: str
    opcode: int
    response_mode: str
    response_opcode: int | None
    timeout_ms: int
    risk: str
    success_rule: str
    success_field: str | None
    success_value: bytes | None
    stable_on_response: bool
    description: str


@dataclass(frozen=True, slots=True)
class SendField:
    command_id: str
    key: str
    display_name: str
    start: int
    length: int
    data_type: str
    scale: float
    unit: str | None
    minimum: float | None
    maximum: float | None
    source: str
    default: Any
    auto_key: str | None
    enum_id: str | None
    description: str
    layout_id: str | None = None


@dataclass(frozen=True, slots=True)
class ResponseRule:
    rule_id: str
    request_command_id: str
    opcode: int
    mode: str
    expected_frames: int
    total_frame_byte: int
    frame_index_byte: int
    data_start: int
    max_frames: int
    ordered: bool
    timeout_ms: int


@dataclass(frozen=True, slots=True)
class ResponseField:
    rule_id: str
    frame_index: int | None
    key: str
    display_name: str
    start: int
    length: int
    data_type: str
    scale: float
    decimals: int
    unit: str | None
    enum_id: str | None
    auto_key: str | None


class ResponseAssembly:
    """One V1.0 response group; byte positions are 1-based within the 14-byte payload."""

    def __init__(self, rule: ResponseRule) -> None:
        self.rule = rule
        self.started_at = time.monotonic()
        self._frames: dict[int, bytes] = {}
        self._total: int | None = None
        self._array_id: int | None = None

    @property
    def frames(self) -> list[bytes]:
        return [self._frames[index] for index in sorted(self._frames)]

    def add(self, frame: bytes) -> bool:
        rule = self.rule
        if time.monotonic() - self.started_at > rule.timeout_ms / 1000:
            raise ServiceError("TIMEOUT", "多帧应答未在规定时间内收齐", "response_assembly", rule.rule_id)
        if len(frame) != 22 or frame[:4] != b"\xAA\x55\x00\x16" or not verify_crc_be(frame):
            raise ServiceError("DATA_INTEGRITY", "应答组包含非法帧", "response_assembly", rule.rule_id)
        if frame[4] != rule.opcode or (self._array_id is not None and frame[5] != self._array_id):
            raise ServiceError("DATA_INTEGRITY", "应答组的指令或阵面 ID 不一致", "response_assembly", rule.rule_id)
        self._array_id = frame[5]
        if rule.mode == "SINGLE":
            index, total = 1, 1
        else:
            total = frame[5 + rule.total_frame_byte] if rule.total_frame_byte else rule.expected_frames
            index = frame[5 + rule.frame_index_byte]
        if not 1 <= total <= rule.max_frames or not 1 <= index <= total:
            raise ServiceError("DATA_INTEGRITY", "多帧总数或序号超出范围（序号从 1 开始）", "response_assembly", rule.rule_id)
        if rule.expected_frames and total != rule.expected_frames:
            raise ServiceError("DATA_INTEGRITY", "应答帧数与配置不一致", "response_assembly", rule.rule_id)
        if self._total is not None and self._total != total:
            raise ServiceError("DATA_INTEGRITY", "应答组的总帧数发生变化", "response_assembly", rule.rule_id)
        if index in self._frames or (rule.ordered and index != len(self._frames) + 1):
            raise ServiceError("DATA_INTEGRITY", "多帧应答重复或顺序错误", "response_assembly", rule.rule_id)
        self._total = total
        self._frames[index] = bytes(frame)
        return len(self._frames) == total


@dataclass(frozen=True, slots=True)
class EnumValue:
    mapping_id: str
    wire: bytes
    logical: str
    display: str


@dataclass(frozen=True, slots=True)
class BitMember:
    layout_id: str
    key: str
    first_bit: int
    bit_count: int
    source: str
    value_or_key: Any
    enum_id: str | None


@dataclass(slots=True)
class ProtocolProfile:
    asset_id: str
    path: str
    file_sha256: str
    profile_id: str
    profile_name: str
    protocol_version: str
    supported_polarizations: str
    description: str
    commands: dict[str, Command]
    all_command_ids: set[str]
    send_fields: dict[str, list[SendField]]
    response_rules: dict[int, ResponseRule]
    response_fields: dict[str, list[ResponseField]]
    enums: dict[str, list[EnumValue]]
    bit_layouts: dict[str, list[BitMember]]
    vector_results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def capabilities(self) -> dict[str, bool]:
        roles = {command.auto_role for command in self.commands.values()}
        beam_commands = [command for command in self.commands.values() if command.auto_role == "BEAM_SET"]
        beam_signal_path = bool(
            beam_commands
            and any(
                field.auto_key == "signal_path"
                for field in self.send_fields.get(beam_commands[0].command_id, [])
            )
        )
        return {
            "manual_debug": bool(self.commands or self.response_rules),
            "pattern": "BEAM_SET" in roles,
            "beam_signal_path": beam_signal_path,
            "calibration": "CALIBRATION_WRITE" in roles,
            "initialization": "INITIALIZE_QUERY" in roles,
            "flash_protocol": any(role.startswith("FLASH_") for role in roles),
        }

    def command_for_role(self, role: str) -> Command:
        matches = [command for command in self.commands.values() if command.auto_role == role]
        if len(matches) != 1:
            raise ServiceError(
                "NOT_RUNNABLE",
                f"配置包没有唯一的自动业务角色 {role}",
                "profile_compile",
                self.profile_id,
            )
        return matches[0]

    def _enum_wire(self, mapping_id: str, value: Any, *, length: int) -> bytes:
        entries = self.enums.get(mapping_id, [])
        requested = _text(value).upper()
        for entry in entries:
            if requested in {entry.logical.upper(), entry.display.upper(), entry.wire.hex().upper()}:
                if len(entry.wire) != length:
                    raise invalid("枚举线路值字节数与字段不一致", stage="profile_encode", target=mapping_id)
                return entry.wire
        if isinstance(value, (int, bytes, bytearray, list, tuple)) or re.fullmatch(
            r"(?:0x)?[0-9A-Fa-f]{2,}", _text(value)
        ):
            return _hex_bytes(value, location=f"enum:{mapping_id}", expected=length)
        raise invalid(
            f"枚举 {mapping_id} 中不存在值 {value}", stage="profile_encode", target=mapping_id
        )

    def compile_bit_layout(self, layout_id: str, values: dict[str, Any]) -> bytes:
        members = self.bit_layouts.get(layout_id)
        if not members:
            raise ServiceError("NOT_RUNNABLE", f"缺少位域 {layout_id}", "profile_compile", self.profile_id)
        total_bits = max(member.first_bit + member.bit_count for member in members)
        if total_bits % 8:
            raise invalid("位域总长度不是完整字节", stage="profile_compile", target=layout_id)
        word = 0
        occupied = 0
        for member in members:
            mask = ((1 << member.bit_count) - 1) << (total_bits - member.first_bit - member.bit_count)
            if occupied & mask:
                raise invalid("位域成员重叠", stage="profile_compile", target=f"{layout_id}.{member.key}")
            occupied |= mask
            if member.source == "CONSTANT":
                raw = int(member.value_or_key)
            elif member.source == "MAPPING":
                if member.value_or_key not in values:
                    raise invalid(
                        "位域映射值缺失",
                        stage="profile_compile",
                        target=f"{layout_id}.{member.value_or_key}",
                    )
                raw = _integer(values[member.value_or_key], f"{layout_id}.{member.key}")
            elif member.source == "ENUM":
                key = _text(member.value_or_key)
                if key not in values:
                    raise invalid("位域枚举值缺失", stage="profile_compile", target=f"{layout_id}.{key}")
                entries = self.enums.get(member.enum_id or "", [])
                width = len(entries[0].wire) if entries else math.ceil(member.bit_count / 8)
                wire = self._enum_wire(member.enum_id or "", values[key], length=width)
                raw = int.from_bytes(wire, "big")
            else:
                raise invalid("不支持的位域值来源", stage="profile_compile", target=member.source)
            if not 0 <= raw < (1 << member.bit_count):
                raise invalid("位域值越界", stage="profile_compile", target=f"{layout_id}.{member.key}")
            word |= raw << (total_bits - member.first_bit - member.bit_count)
        return word.to_bytes(total_bits // 8, "big")

    def build_calibration_frame(
        self,
        *,
        array_id: int,
        spi_no: int,
        chip_no: int,
        chip_channel_index: int,
        signal_path: str,
        enabled: bool = True,
    ) -> bytes:
        # The coordinate workbook is explicitly zero-based, while the x_radar enum is
        # CHANNEL_1..CHANNEL_8. This is the single, audited conversion boundary.
        physical_channel = chip_channel_index + 1
        selector = f"CHANNEL_{physical_channel}" if enabled else "ALL_OFF"
        control_word = self.compile_bit_layout(
            "CHIP_DIRECT_WRITE_WORD",
            {"chip_no": chip_no, "chip_channel_no": selector},
        )
        command = self.command_for_role("CALIBRATION_WRITE")
        return self.encode(
            command.command_id,
            array_id,
            {
                "spi_no": spi_no,
                "chip_control_word": control_word,
                "chip_trx": signal_path.upper(),
            },
        )

    def _encode_field(self, item: SendField, value: Any) -> bytes:
        data_type = item.data_type.lower()
        if data_type in {"constant_hex", "raw_hex", "bytes"} or re.fullmatch(r"bytes[1-9]\d*", data_type):
            return _hex_bytes(value, location=f"03_发送字段:{item.command_id}.{item.key}", expected=item.length)
        if data_type == "ascii":
            result = _text(value).encode("ascii", errors="strict")
            return result[: item.length].ljust(item.length, b"\x00")
        if data_type == "enum":
            if not item.enum_id:
                raise invalid("枚举字段缺少 EnumId", stage="profile_encode", target=item.key)
            return self._enum_wire(item.enum_id, value, length=item.length)
        if data_type == "bitfield":
            if not isinstance(value, dict) or not item.layout_id:
                raise invalid("位域字段需要布局及成员参数", stage="profile_encode", target=item.key)
            return self.compile_bit_layout(item.layout_id, value)
        match = re.fullmatch(r"(u?int)(8|16|24|32|64)", data_type)
        if not match:
            raise invalid("不支持的数据类型", stage="profile_encode", target=data_type)
        numeric = float(value)
        if not math.isfinite(numeric):
            raise invalid("工程值必须为有限数", stage="profile_encode", target=item.key)
        if item.minimum is not None and numeric < item.minimum:
            raise invalid("工程值低于最小值", stage="profile_encode", target=item.key, value=numeric)
        if item.maximum is not None and numeric > item.maximum:
            raise invalid("工程值高于最大值", stage="profile_encode", target=item.key, value=numeric)
        raw = round(numeric * item.scale)
        bits = int(match.group(2))
        signed = not data_type.startswith("u")
        low = -(1 << (bits - 1)) if signed else 0
        high = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
        if not low <= raw <= high:
            raise invalid("缩放后的线路整数越界", stage="profile_encode", target=item.key, raw=raw)
        return int(raw).to_bytes(item.length, "big", signed=signed)

    def encode(self, command_id: str, array_id: int, inputs: dict[str, Any] | None = None) -> bytes:
        inputs = dict(inputs or {})
        command = self.commands.get(command_id)
        if command is None:
            raise ServiceError("NOT_FOUND", f"未启用指令 {command_id}", "profile_encode", self.profile_id)
        if not 0 <= array_id <= 255:
            raise invalid("阵面 ID 必须为 0..255", stage="profile_encode", target="array_id")
        payload = bytearray(14)
        resolved: dict[str, Any] = {}
        for item in self.send_fields.get(command_id, []):
            source = item.source.upper()
            if item.data_type.lower() == "bitfield":
                value = dict(inputs.get(item.key, {})) if isinstance(inputs.get(item.key), dict) else {}
                for member in self.bit_layouts[item.layout_id or ""]:
                    if member.source == "CONSTANT":
                        continue
                    name = _text(member.value_or_key)
                    if f"{item.key}.{name}" in inputs:
                        value[name] = inputs[f"{item.key}.{name}"]
                    elif name in inputs:
                        value[name] = inputs[name]
            elif source == "CONSTANT":
                value = item.default
            elif source == "MIRROR_FIELD":
                reference = _text(item.default)
                value = resolved.get(reference, inputs.get(reference))
                if value is None:
                    raise invalid("镜像字段来源缺失", stage="profile_encode", target=item.key)
            else:
                candidates = [item.auto_key, item.key]
                value = next((inputs[key] for key in candidates if key and key in inputs), item.default)
                if value is None:
                    raise invalid("发送字段值缺失", stage="profile_encode", target=item.key)
            encoded = self._encode_field(item, value)
            if len(encoded) != item.length:
                raise invalid("字段编码长度与声明字节数不一致", stage="profile_encode", target=item.key)
            start = item.start - 1
            payload[start : start + item.length] = encoded
            resolved[item.key] = value
        prefix = bytes([0xAA, 0x55, 0x00, 0x16, command.opcode, array_id]) + bytes(payload)
        return append_crc_be(prefix)

    def decode(self, frame: bytes) -> dict[str, Any]:
        if len(frame) != 22 or frame[:2] != b"\xAA\x55" or int.from_bytes(frame[2:4], "big") != 22:
            raise invalid("不是合法的 22 字节天线帧", stage="profile_decode", target=self.profile_id)
        if not verify_crc_be(frame):
            raise ServiceError("DATA_INTEGRITY", "天线帧 CRC 错误", "profile_decode", self.profile_id)
        opcode = frame[4]
        rule = self.response_rules.get(opcode)
        payload = frame[6:20]
        result: dict[str, Any] = {
            "opcode": opcode,
            "array_id": frame[5],
            "raw_hex": frame.hex(" ").upper(),
            "fields": {},
        }
        if rule is None:
            return result
        frame_index = payload[rule.frame_index_byte - 1] if rule.mode == "MULTI" else 1
        result["frame_index"] = frame_index
        data = payload[rule.data_start - 1 :]
        for item in self.response_fields.get(rule.rule_id, []):
            if item.frame_index is not None and item.frame_index != frame_index:
                continue
            raw_bytes = data[item.start - 1 : item.start - 1 + item.length]
            data_type = item.data_type.lower()
            if data_type.startswith("uint"):
                value: Any = int.from_bytes(raw_bytes, "big") / item.scale
            elif data_type.startswith("int"):
                value = int.from_bytes(raw_bytes, "big", signed=True) / item.scale
            elif data_type == "ascii":
                value = raw_bytes.rstrip(b"\x00").decode("ascii", errors="replace")
            else:
                value = raw_bytes.hex(" ").upper()
            if item.enum_id:
                matched = next((e for e in self.enums.get(item.enum_id, []) if e.wire == raw_bytes), None)
                if matched:
                    value = {"wire": raw_bytes.hex(" ").upper(), "logical": matched.logical, "display": matched.display}
            result["fields"][item.key] = {"value": value, "wire": raw_bytes.hex(" ").upper(), "unit": item.unit, "display": item.display_name}
        return result

    def decode_response(self, frames: list[bytes]) -> dict[str, Any]:
        if not frames:
            raise ServiceError("DATA_INTEGRITY", "应答为空", "profile_decode", self.profile_id)
        first = self.decode(frames[0])
        rule = self.response_rules.get(first["opcode"])
        if rule is None:
            if len(frames) != 1:
                raise ServiceError("DATA_INTEGRITY", "多帧应答缺少组帧规则", "profile_decode", self.profile_id)
            return first
        assembly = ResponseAssembly(rule)
        complete = False
        for frame in frames:
            complete = assembly.add(frame)
        if not complete:
            raise ServiceError("DATA_INTEGRITY", "多帧应答不完整", "profile_decode", rule.rule_id)
        decoded = [self.decode(frame) for frame in assembly.frames]
        if len(decoded) == 1:
            return decoded[0]
        return {
            "opcode": first["opcode"], "array_id": first["array_id"], "complete": True,
            "frames": decoded,
            "fields": {f"{item['frame_index']}:{key}": value for item in decoded for key, value in item["fields"].items()},
        }

    def validate_response(self, command: Command, request: bytes, frames: list[bytes]) -> dict[str, Any]:
        decoded = self.decode_response(frames)
        expected_opcode = command.response_opcode if command.response_opcode is not None else command.opcode
        if decoded["opcode"] != expected_opcode or decoded["array_id"] != request[5]:
            raise ServiceError("DATA_INTEGRITY", "应答与请求的指令/阵面不一致", "antenna_response", command.command_id)
        if command.success_rule == "FRAME_EQUALS_REQUEST" and frames != [request]:
            raise ServiceError("DATA_INTEGRITY", "天线回显与请求不一致", "antenna_response", command.command_id)
        if command.success_rule == "FIELD_EQUALS":
            matches = [item["fields"][command.success_field] for item in decoded.get("frames", [decoded])
                       if command.success_field in item["fields"]]
            expected = command.success_value
            if not matches or expected is None or any(bytes.fromhex(item["wire"]) != expected for item in matches):
                raise ServiceError("NOT_RUNNABLE", "应答字段未满足配置的成功条件", "antenna_response", command.command_id,
                                   {"field": command.success_field, "expected": expected.hex() if expected is not None else None, "actual": matches})
        return decoded

    def summary(self) -> dict[str, Any]:
        command_summaries: list[dict[str, Any]] = []
        for command in self.commands.values():
            fields: list[dict[str, Any]] = []
            for item in self.send_fields.get(command.command_id, []):
                field = asdict(item)
                if isinstance(field["default"], (bytes, bytearray)):
                    field["default"] = bytes(field["default"]).hex(" ").upper()
                enum_id = item.enum_id if item.data_type.lower() == "enum" else None
                field["enum_options"] = [
                    {"logical": entry.logical, "display": entry.display, "wire": entry.wire.hex(" ").upper()}
                    for entry in self.enums.get(enum_id or "", [])
                ]
                if item.data_type.lower() == "bitfield":
                    for member in self.bit_layouts[item.layout_id or ""]:
                        if member.source == "CONSTANT":
                            continue
                        fields.append(field | {
                            "key": f"{item.key}.{member.value_or_key}", "display_name": f"{item.display_name} / {member.key}",
                            "data_type": "enum" if member.source == "ENUM" else "uint64",
                            "source": "USER", "default": None, "minimum": 0, "maximum": (1 << member.bit_count) - 1,
                            "enum_options": [{"logical": entry.logical, "display": entry.display, "wire": entry.wire.hex(" ").upper()}
                                             for entry in self.enums.get(member.enum_id or "", [])],
                        })
                    continue
                fields.append(field)
            command_summaries.append(
                asdict(command)
                | {
                    "opcode_hex": f"{command.opcode:02X}",
                    "success_value": command.success_value.hex(" ").upper() if command.success_value else None,
                    "fields": fields,
                }
            )
        return {
            "asset_id": self.asset_id,
            "profile_id": self.profile_id,
            "profile_name": self.profile_name,
            "protocol_version": self.protocol_version,
            "supported_polarizations": self.supported_polarizations,
            "file_sha256": self.file_sha256,
            "command_count": len(self.commands),
            "commands": command_summaries,
            "capabilities": self.capabilities,
            "vectors": self.vector_results,
        }


class ProfileLoader:
    """Strict parser for the V1.0 workbook; formulas are never trusted as validation."""

    def load(self, file_path: str) -> ProtocolProfile:
        path = Path(file_path).resolve()
        if path.suffix.lower() != ".xlsx":
            raise invalid("配置包必须是 .xlsx", stage="profile_load", target=str(path))
        try:
            workbook = load_workbook(path, read_only=False, data_only=False)
        except Exception as exc:
            raise ServiceError("DATA_INTEGRITY", "无法读取配置包", "profile_load", str(path), {"error": str(exc)}) from exc
        if workbook.sheetnames != EXPECTED_SHEETS:
            raise invalid(
                "配置包工作表名称或顺序不符合 V1.0",
                stage="profile_load",
                target=str(path),
                expected=EXPECTED_SHEETS,
                actual=workbook.sheetnames,
            )

        basic_sheet = workbook["01_基本信息"]
        basic_header = _header_row(basic_sheet, BASIC_HEADERS)
        basic: dict[str, Any] = {}
        basic_locations: dict[str, str] = {}
        for row in range(basic_header + 1, basic_sheet.max_row + 1):
            key = _text(basic_sheet.cell(row, 1).value)
            if not key:
                continue
            if key in basic:
                raise invalid("基本信息键重复", stage="profile_load", target=f"01_基本信息!A{row}")
            basic[key] = basic_sheet.cell(row, 2).value
            basic_locations[key] = f"01_基本信息!B{row}"
        if _text(basic.get("schemaVersion")).upper() != "V1.0":
            raise invalid("只支持 schemaVersion=V1.0", stage="profile_load", target=basic_locations.get("schemaVersion", "01_基本信息!A:A"))
        for key in ("profileId", "profileName", "protocolVersion", "supportedPolarizations", "arrayIdPolicy"):
            if not _text(basic.get(key)):
                raise invalid(f"基本信息缺少 {key}", stage="profile_load", target=basic_locations.get(key,"01_基本信息!A:A"))
        if _text(basic.get("frameFormat")) != "AA55_22_CCITT_FALSE_BE" or int(basic.get("payloadBytes", 0)) != 14:
            raise invalid("当前版本只支持固定 22 字节 AA55 帧", stage="profile_load", target="01_基本信息")

        commands, all_command_ids = self._commands(workbook["02_指令"])
        fields = self._send_fields(workbook["03_发送字段"], all_command_ids)
        response_rules, response_fields = self._responses(workbook["04_接收解析"], all_command_ids)
        enums, bit_layouts = self._enums_and_bits(workbook["05_枚举位域"])
        self._validate_field_mappings(fields, enums, commands)
        file_sha = _sha256(path)
        profile = ProtocolProfile(
            asset_id=f"{_text(basic['profileId'])}:{file_sha[:12]}",
            path=str(path),
            file_sha256=file_sha,
            profile_id=_text(basic["profileId"]),
            profile_name=_text(basic["profileName"]),
            protocol_version=_text(basic["protocolVersion"]),
            supported_polarizations=_text(basic["supportedPolarizations"]).upper(),
            description=_text(basic.get("description")),
            commands=commands,
            all_command_ids=all_command_ids,
            send_fields=fields,
            response_rules=response_rules,
            response_fields=response_fields,
            enums=enums,
            bit_layouts=bit_layouts,
        )
        self._validate_profile(profile, workbook)
        profile.vector_results = self._verify_vectors(workbook["06_测试向量"], profile)
        workbook.close()
        return profile

    def _commands(self, sheet: Any) -> tuple[dict[str, Command], set[str]]:
        commands: dict[str, Command] = {}
        all_ids: set[str] = set()
        roles: set[str] = set()
        header = _header_row(sheet, COMMAND_HEADERS)
        active = set(_active_rows(sheet, header + 1))
        for row in range(header + 1, sheet.max_row + 1):
            command_id = _text(sheet.cell(row, 2).value)
            if command_id:
                if command_id in all_ids:
                    raise invalid("CommandId 重复", stage="profile_load", target=f"02_指令!B{row}")
                all_ids.add(command_id)
            if row not in active:
                continue
            if not command_id:
                raise invalid("启用的指令缺少 CommandId", stage="profile_load", target=f"02_指令!B{row}")
            role = _text(sheet.cell(row, 4).value).upper() or "MANUAL_ONLY"
            if role != "MANUAL_ONLY" and role in roles:
                raise invalid("自动业务角色重复", stage="profile_load", target=f"02_指令!D{row}")
            roles.add(role)
            opcode = int.from_bytes(_hex_bytes(sheet.cell(row, 5).value, location=f"02_指令!E{row}", expected=1), "big")
            response_hex = sheet.cell(row, 7).value
            response_opcode = int.from_bytes(_hex_bytes(response_hex, location=f"02_指令!G{row}", expected=1), "big") if response_hex not in (None, "") else None
            success_value = _hex_bytes(sheet.cell(row, 12).value, location=f"02_指令!L{row}") if sheet.cell(row, 12).value not in (None, "") else None
            commands[command_id] = Command(
                command_id,
                _text(sheet.cell(row, 3).value),
                role,
                opcode,
                _text(sheet.cell(row, 6).value).upper(),
                response_opcode,
                _integer(sheet.cell(row, 8).value, f"{sheet.title}!H{row}", 1),
                _text(sheet.cell(row, 9).value).upper(),
                _text(sheet.cell(row, 10).value).upper(),
                _text(sheet.cell(row, 11).value) or None,
                success_value,
                _enabled(sheet.cell(row, 13).value),
                _text(sheet.cell(row, 14).value),
            )
        return commands, all_ids

    def _send_fields(self, sheet: Any, all_ids: set[str]) -> dict[str, list[SendField]]:
        result: dict[str, list[SendField]] = {}
        identities: set[tuple[str, str]] = set()
        occupied: dict[str, set[int]] = {}
        header = _header_row(sheet, SEND_HEADERS)
        has_layout = _header_text(sheet.cell(header, 17).value) == "LayoutId"
        if _text(sheet.cell(header, 17).value) and not has_layout:
            raise invalid("第17列表头应为 LayoutId", stage="profile_load", target=f"{sheet.title}!Q{header}")
        for row in _active_rows(sheet, header + 1):
            command_id = _text(sheet.cell(row, 2).value)
            key = _text(sheet.cell(row, 3).value)
            if command_id not in all_ids:
                raise invalid("发送字段引用未知 CommandId", stage="profile_load", target=f"03_发送字段!B{row}")
            identity = (command_id, key)
            if identity in identities:
                raise invalid("CommandId+FieldKey 重复", stage="profile_load", target=f"03_发送字段!C{row}")
            identities.add(identity)
            if not key:
                raise invalid("FieldKey 不能为空", stage="profile_load", target=f"{sheet.title}!C{row}")
            start = _integer(sheet.cell(row, 5).value, f"{sheet.title}!E{row}", 1)
            length = _integer(sheet.cell(row, 6).value, f"{sheet.title}!F{row}", 1)
            _field_type(_text(sheet.cell(row, 7).value), length, f"{sheet.title}!G{row}", sending=True)
            positions = set(range(start, start + length))
            if start < 1 or start + length - 1 > 14:
                raise invalid("发送字段超出 14 字节负载", stage="profile_load", target=f"03_发送字段!E{row}")
            if positions & occupied.setdefault(command_id, set()):
                raise invalid("同一指令发送字段重叠", stage="profile_load", target=f"03_发送字段!E{row}")
            occupied[command_id] |= positions
            scale = float(sheet.cell(row, 8).value)
            if not math.isfinite(scale) or scale <= 0:
                raise invalid("Scale 必须为正数", stage="profile_load", target=f"03_发送字段!H{row}")
            result.setdefault(command_id, []).append(
                SendField(
                    command_id,
                    key,
                    _text(sheet.cell(row, 4).value),
                    start,
                    length,
                    _text(sheet.cell(row, 7).value),
                    scale,
                    _text(sheet.cell(row, 9).value) or None,
                    float(sheet.cell(row, 10).value) if sheet.cell(row, 10).value is not None else None,
                    float(sheet.cell(row, 11).value) if sheet.cell(row, 11).value is not None else None,
                    _text(sheet.cell(row, 12).value).upper(),
                    sheet.cell(row, 13).value,
                    _text(sheet.cell(row, 14).value) or None,
                    _text(sheet.cell(row, 15).value) or None,
                    _text(sheet.cell(row, 16).value),
                    _text(sheet.cell(row, 17).value) or None if has_layout else None,
                )
            )
        for items in result.values():
            items.sort(key=lambda item: item.start)
        return result

    def _responses(
        self, sheet: Any, all_ids: set[str]
    ) -> tuple[dict[int, ResponseRule], dict[str, list[ResponseField]]]:
        marker_rows: dict[str, int] = {}
        for row in range(1, sheet.max_row + 1):
            marker = _text(sheet.cell(row, 1).value)
            if marker not in {RESPONSE_RULES_MARKER, RESPONSE_FIELDS_MARKER}:
                continue
            if marker in marker_rows:
                raise invalid(
                    f"接收解析章节标记重复：{marker}",
                    stage="profile_load",
                    target=f"04_接收解析!A{row}",
                )
            marker_rows[marker] = row
        for marker in (RESPONSE_RULES_MARKER, RESPONSE_FIELDS_MARKER):
            if marker not in marker_rows:
                raise invalid(
                    f"缺少接收解析章节标记：{marker}",
                    stage="profile_load",
                    target="04_接收解析!A:A",
                )
        rules_marker_row = marker_rows[RESPONSE_RULES_MARKER]
        fields_marker_row = marker_rows[RESPONSE_FIELDS_MARKER]
        if fields_marker_row <= rules_marker_row + 1:
            raise invalid(
                "接收解析章节顺序错误，B. 应答字段必须位于 A. 应答组帧规则之后",
                stage="profile_load",
                target=f"04_接收解析!A{fields_marker_row}",
            )
        rules_header_row = _header_row(sheet, RESPONSE_RULE_HEADERS, rules_marker_row + 1, fields_marker_row)
        fields_header_row = _header_row(sheet, RESPONSE_FIELD_HEADERS, fields_marker_row + 1)

        rules_by_opcode: dict[int, ResponseRule] = {}
        rule_ids: set[str] = set()
        for row in _active_rows(sheet, rules_header_row + 1, fields_marker_row):
            rule_id = _text(sheet.cell(row, 2).value)
            request_id = _text(sheet.cell(row, 3).value)
            if rule_id in rule_ids or request_id not in all_ids:
                raise invalid("接收规则 ID 重复或引用未知指令", stage="profile_load", target=f"04_接收解析!B{row}")
            rule_ids.add(rule_id)
            opcode = int.from_bytes(_hex_bytes(sheet.cell(row, 4).value, location=f"04_接收解析!D{row}", expected=1), "big")
            if opcode in rules_by_opcode:
                raise invalid("同一 Opcode 存在多条接收规则", stage="profile_load", target=f"04_接收解析!D{row}")
            rules_by_opcode[opcode] = ResponseRule(
                rule_id,
                request_id,
                opcode,
                _text(sheet.cell(row, 5).value).upper(),
                _integer(sheet.cell(row, 6).value or 0, f"{sheet.title}!F{row}"),
                _integer(sheet.cell(row, 7).value or 0, f"{sheet.title}!G{row}"),
                _integer(sheet.cell(row, 8).value or 0, f"{sheet.title}!H{row}"),
                _integer(sheet.cell(row, 9).value or 1, f"{sheet.title}!I{row}", 1),
                _integer(sheet.cell(row, 10).value or 1, f"{sheet.title}!J{row}", 1),
                _enabled(sheet.cell(row, 11).value),
                _integer(sheet.cell(row, 12).value, f"{sheet.title}!L{row}", 1),
            )
            rule = rules_by_opcode[opcode]
            if rule.mode not in {"SINGLE", "MULTI"} or not 1 <= rule.data_start <= 14 or rule.timeout_ms <= 0:
                raise invalid("应答模式、数据起点或超时不合法", stage="profile_load", target=f"04_接收解析!E{row}")
            if not 1 <= rule.max_frames <= 255 or not 0 <= rule.expected_frames <= rule.max_frames:
                raise invalid("应答帧数范围不合法", stage="profile_load", target=f"04_接收解析!F{row}")
            if rule.mode == "MULTI" and (not 1 <= rule.frame_index_byte <= 14 or not 0 <= rule.total_frame_byte <= 14 or (not rule.total_frame_byte and not rule.expected_frames)):
                raise invalid("多帧应答必须声明有效的序号及总帧数来源", stage="profile_load", target=f"04_接收解析!G{row}")
            if rule.mode == "SINGLE" and (rule.expected_frames != 1 or rule.max_frames != 1):
                raise invalid("单帧应答的帧数必须为 1", stage="profile_load", target=f"04_接收解析!F{row}")
        fields: dict[str, list[ResponseField]] = {}
        for row in _active_rows(sheet, fields_header_row + 1):
            rule_id = _text(sheet.cell(row, 2).value)
            if rule_id not in rule_ids:
                raise invalid("应答字段引用未知 RuleId", stage="profile_load", target=f"04_接收解析!B{row}")
            selector = _text(sheet.cell(row, 3).value)
            match = re.fullmatch(r"frame_index=(\d+)", selector) if selector else None
            if selector and (match is None or int(match.group(1)) < 1):
                raise invalid("帧选择必须留空或为 frame_index=正整数", stage="profile_load", target=f"04_接收解析!C{row}")
            fields.setdefault(rule_id, []).append(
                ResponseField(
                    rule_id,
                    int(match.group(1)) if match else None,
                    _text(sheet.cell(row, 4).value),
                    _text(sheet.cell(row, 5).value),
                    _integer(sheet.cell(row, 6).value, f"{sheet.title}!F{row}", 1),
                    _integer(sheet.cell(row, 7).value, f"{sheet.title}!G{row}", 1),
                    _text(sheet.cell(row, 8).value),
                    float(sheet.cell(row, 9).value),
                    _integer(sheet.cell(row, 10).value or 0, f"{sheet.title}!J{row}"),
                    _text(sheet.cell(row, 11).value) or None,
                    _text(sheet.cell(row, 12).value) or None,
                    _text(sheet.cell(row, 13).value) or None,
                )
            )
            item = fields[rule_id][-1]
            rule = next(value for value in rules_by_opcode.values() if value.rule_id == rule_id)
            _field_type(item.data_type, item.length, f"{sheet.title}!H{row}")
            if not item.key or any(other.key == item.key and (other.frame_index is None or item.frame_index is None or other.frame_index == item.frame_index) for other in fields[rule_id][:-1]):
                raise invalid("同一应答帧的 FieldKey 为空或重复", stage="profile_load", target=f"{sheet.title}!D{row}")
            if item.start < 1 or item.length < 1 or rule.data_start + item.start + item.length - 2 > 14 or not math.isfinite(item.scale) or item.scale <= 0:
                raise invalid("应答字段越过 Payload 或 Scale 无效", stage="profile_load", target=f"04_接收解析!F{row}")
            if item.frame_index is not None and item.frame_index > rule.max_frames:
                raise invalid("帧选择超出最大帧数", stage="profile_load", target=f"04_接收解析!C{row}")
        return rules_by_opcode, fields

    def _enums_and_bits(self, sheet: Any) -> tuple[dict[str, list[EnumValue]], dict[str, list[BitMember]]]:
        enums: dict[str, list[EnumValue]] = {}
        seen: set[tuple[str, bytes]] = set()
        enum_marker, bits_marker = _marker_row(sheet, ENUMS_MARKER), _marker_row(sheet, BITS_MARKER)
        if enum_marker >= bits_marker:
            raise invalid("枚举和位域章节顺序错误", stage="profile_load", target=f"{sheet.title}!A{bits_marker}")
        enum_header = _header_row(sheet, ENUM_HEADERS, enum_marker + 1, bits_marker)
        bit_header = _header_row(sheet, BIT_HEADERS, bits_marker + 1)
        aliases: dict[tuple[str, str], bytes] = {}
        for row in _active_rows(sheet, enum_header + 1, bits_marker):
            mapping_id = _text(sheet.cell(row, 2).value)
            wire = _hex_bytes(sheet.cell(row, 3).value, location=f"05_枚举位域!C{row}")
            logical, display = _text(sheet.cell(row, 4).value), _text(sheet.cell(row, 5).value)
            if not mapping_id or not logical or not display:
                raise invalid("枚举 ID、逻辑值及显示文本不能为空", stage="profile_load", target=f"{sheet.title}!B{row}")
            if (mapping_id, wire) in seen:
                raise invalid("枚举线路值重复", stage="profile_load", target=f"05_枚举位域!C{row}")
            seen.add((mapping_id, wire))
            for alias in {logical.upper(), display.upper(), wire.hex().upper()}:
                if (mapping_id, alias) in aliases and aliases[(mapping_id, alias)] != wire:
                    raise invalid("枚举逻辑值或显示文本存在歧义", stage="profile_load", target=f"{sheet.title}!D{row}")
                aliases[(mapping_id, alias)] = wire
            if enums.get(mapping_id) and len(enums[mapping_id][0].wire) != len(wire):
                raise invalid("同一枚举的线路值必须等宽", stage="profile_load", target=f"{sheet.title}!C{row}")
            enums.setdefault(mapping_id, []).append(
                EnumValue(mapping_id, wire, _text(sheet.cell(row, 4).value), _text(sheet.cell(row, 5).value))
            )
        layouts: dict[str, list[BitMember]] = {}
        for row in _active_rows(sheet, bit_header + 1):
            member = BitMember(
                _text(sheet.cell(row, 2).value),
                _text(sheet.cell(row, 3).value),
                _integer(sheet.cell(row, 4).value, f"{sheet.title}!D{row}"),
                _integer(sheet.cell(row, 5).value, f"{sheet.title}!E{row}", 1),
                _text(sheet.cell(row, 6).value).upper(),
                sheet.cell(row, 7).value,
                _text(sheet.cell(row, 8).value) or None,
            )
            existing = layouts.get(member.layout_id, [])
            if not member.layout_id or not member.key or any(other.key == member.key for other in existing):
                raise invalid("位域ID/成员键为空或重复", stage="profile_load", target=f"{sheet.title}!C{row}")
            if member.first_bit + member.bit_count > 112 or any(member.first_bit < other.first_bit + other.bit_count and other.first_bit < member.first_bit + member.bit_count for other in existing):
                raise invalid("位域成员越界或重叠", stage="profile_load", target=f"{sheet.title}!D{row}")
            if member.source not in {"CONSTANT", "MAPPING", "ENUM"}:
                raise invalid("位域值来源不支持", stage="profile_load", target=f"{sheet.title}!F{row}")
            if member.source == "ENUM":
                if member.enum_id not in enums or any(int.from_bytes(entry.wire, 'big') >= 1 << member.bit_count for entry in enums[member.enum_id]):
                    raise invalid("位域枚举引用不存在或值超出位数", stage="profile_load", target=f"{sheet.title}!H{row}")
            elif member.enum_id:
                raise invalid("非枚举位域成员不能填写 EnumId", stage="profile_load", target=f"{sheet.title}!H{row}")
            if member.source == "CONSTANT" and _integer(member.value_or_key, f"{sheet.title}!G{row}") >= 1 << member.bit_count:
                raise invalid("位域常量超出位数", stage="profile_load", target=f"{sheet.title}!G{row}")
            if member.source != "CONSTANT" and not _text(member.value_or_key):
                raise invalid("位域参数键不能为空", stage="profile_load", target=f"{sheet.title}!G{row}")
            layouts.setdefault(member.layout_id, []).append(member)
        for layout_id, members in layouts.items():
            if max(member.first_bit + member.bit_count for member in members) % 8:
                raise invalid("位域总长度必须为完整字节", stage="profile_load", target=f"{sheet.title}:{layout_id}")
        return enums, layouts

    def _validate_field_mappings(
        self,
        fields: dict[str, list[SendField]],
        enums: dict[str, list[EnumValue]],
        commands: dict[str, Command],
    ) -> None:
        allowed_auto_keys = {
            "off_axis_deg",
            "azimuth_deg",
            "frequency_ghz",
            "signal_path",
            "spi_no",
            "chip_control_word",
            "chip_trx",
        }
        for command_id, items in fields.items():
            keys = {item.key for item in items}
            for item in items:
                location = f"03_发送字段:{command_id}.{item.key}"
                if item.auto_key and item.auto_key not in allowed_auto_keys:
                    raise invalid("未知 AutoKey", stage="profile_load", target=location, value=item.auto_key)
                if item.data_type.lower() == "enum":
                    if not item.enum_id:
                        raise invalid("枚举字段必须填写 EnumId", stage="profile_load", target=location)
                    if item.enum_id not in enums:
                        raise invalid("枚举字段引用未知 EnumId", stage="profile_load", target=location, value=item.enum_id)
                elif item.enum_id:
                    raise invalid("非枚举字段不能填写 EnumId", stage="profile_load", target=location)
                if item.source.upper() == "MIRROR_FIELD" and _text(item.default) not in keys:
                    raise invalid("镜像字段引用未知 FieldKey", stage="profile_load", target=location, value=item.default)

        beam_commands = [item for item in commands.values() if item.auto_role == "BEAM_SET"]
        if not beam_commands:
            return
        beam = beam_commands[0]
        beam_fields = fields.get(beam.command_id, [])
        by_auto_key = {item.auto_key: item for item in beam_fields if item.auto_key}
        required = {"off_axis_deg", "azimuth_deg", "frequency_ghz"}
        missing = sorted(required - set(by_auto_key))
        if missing:
            raise invalid(
                "BEAM_SET 缺少自动字段：" + ", ".join(missing),
                stage="profile_load",
                target=f"03_发送字段:{beam.command_id}",
            )
        if any(item.auto_key == "polarization" for item in beam_fields):
            raise invalid("BEAM_SET 不允许极化协议字段", stage="profile_load", target=f"03_发送字段:{beam.command_id}")
        signal_field = by_auto_key.get("signal_path")
        if signal_field is not None:
            if signal_field.length != 1 or signal_field.data_type.lower() != "enum" or not signal_field.enum_id:
                raise invalid("收发模式必须是1字节枚举字段", stage="profile_load", target=f"03_发送字段:{beam.command_id}.{signal_field.key}")
            mapping = {entry.logical.upper(): entry.wire for entry in enums[signal_field.enum_id]}
            if mapping != {"RX": b"\x00", "TX": b"\xff"}:
                raise invalid(
                    "收发模式线路值必须严格为 RX=00、TX=FF",
                    stage="profile_load",
                    target=f"05_枚举位域:{signal_field.enum_id}",
                )

    def _validate_profile(self, profile: ProtocolProfile, workbook: Any) -> None:
        send_sheet = workbook["03_发送字段"]
        send_rows = {(_text(send_sheet.cell(row, 2).value), _text(send_sheet.cell(row, 3).value)): row
                     for row in range(1, send_sheet.max_row + 1) if _enabled(send_sheet.cell(row, 1).value)}
        for command_id, items in profile.send_fields.items():
            for item in items:
                row = send_rows[(command_id, item.key)]
                if item.source not in {"CONSTANT", "USER", "AUTO_OR_USER", "MIRROR_FIELD"}:
                    raise invalid("发送字段值来源无效", stage="profile_load", target=f"03_发送字段!L{row}")
                if item.data_type.lower() == "enum" and any(len(entry.wire) != item.length for entry in profile.enums[item.enum_id]):
                    raise invalid("枚举线路值字节数与字段不一致", stage="profile_load", target=f"03_发送字段!O{row}")
                if item.data_type.lower() == "bitfield":
                    members = profile.bit_layouts.get(item.layout_id or "")
                    if not members or max(member.first_bit + member.bit_count for member in members) != item.length * 8:
                        raise invalid("LayoutId 不存在或布局长度与字段不一致", stage="profile_load", target=f"03_发送字段!Q{row}")
                    if item.source not in {"USER", "AUTO_OR_USER"} or item.default is not None:
                        raise invalid("位域通过成员参数输入，来源为 USER/AUTO_OR_USER 且默认值留空", stage="profile_load", target=f"03_发送字段!L{row}")
                elif item.layout_id:
                    raise invalid("只有 bitfield 字段可填写 LayoutId", stage="profile_load", target=f"03_发送字段!Q{row}")
                if (item.minimum is not None and not math.isfinite(item.minimum)) or (item.maximum is not None and not math.isfinite(item.maximum)) or (item.minimum is not None and item.maximum is not None and item.minimum > item.maximum):
                    raise invalid("字段最小/最大值无效", stage="profile_load", target=f"03_发送字段!J{row}")
                if item.source == "MIRROR_FIELD":
                    source = next(field for field in items if field.key == _text(item.default))
                    if source.start >= item.start or source.length != item.length or source.data_type.lower() != item.data_type.lower():
                        raise invalid("镜像来源必须位于当前字段之前且类型/长度一致", stage="profile_load", target=f"03_发送字段!M{row}")
                elif item.default is not None and item.data_type.lower() != "bitfield":
                    try:
                        if len(profile._encode_field(item, item.default)) != item.length:
                            raise ValueError("编码长度错误")
                    except (ServiceError, ValueError, OverflowError, TypeError) as exc:
                        raise invalid(f"默认/常量值不能编码：{exc}", stage="profile_load", target=f"03_发送字段!M{row}") from exc
                elif item.source == "CONSTANT":
                    raise invalid("常量字段必须填写值", stage="profile_load", target=f"03_发送字段!M{row}")
        response_sheet = workbook["04_接收解析"]
        for rule_id, fields in profile.response_fields.items():
            for item in fields:
                if item.data_type.lower() == "enum":
                    entries = profile.enums.get(item.enum_id or "")
                    if not entries or any(len(entry.wire) != item.length for entry in entries):
                        row = next(row for row in range(1, response_sheet.max_row + 1) if _enabled(response_sheet.cell(row, 1).value) and _text(response_sheet.cell(row, 2).value) == rule_id and _text(response_sheet.cell(row, 4).value) == item.key)
                        raise invalid("接收 EnumId 不存在或线路值宽度不符", stage="profile_load", target=f"04_接收解析!L{row}")
                elif item.enum_id:
                    raise invalid("非枚举接收字段不能填写 EnumId", stage="profile_load", target=f"04_接收解析:{rule_id}.{item.key}")
        for command in profile.commands.values():
            if command.response_mode not in {"NONE", "SINGLE", "MULTI"} or command.success_rule not in {"NONE", "VALID_RESPONSE", "MULTI_COMPLETE", "FIELD_EQUALS", "FRAME_EQUALS_REQUEST"}:
                raise invalid("指令应答模式或成功判定无效", stage="profile_load", target=f"02_指令:{command.command_id}")
            if command.success_rule == "FIELD_EQUALS":
                rule = profile.response_rules.get(command.response_opcode if command.response_opcode is not None else command.opcode)
                matched = [field for field in profile.response_fields.get(rule.rule_id if rule else "", []) if field.key == command.success_field]
                if not matched or command.success_value is None or any(len(command.success_value) != field.length for field in matched):
                    raise invalid("成功字段不存在或成功值宽度不符", stage="profile_load", target=f"02_指令:{command.command_id}")

    def _parse_vector_inputs(self, value: Any, previous: dict[str, Any]) -> dict[str, Any]:
        text = _text(value)
        result = dict(previous) if text.startswith("同上") else {}
        for token in text.split(";"):
            if "=" not in token:
                continue
            key, raw = [part.strip() for part in token.split("=", 1)]
            if key == "trx":
                key = "signal_path"
            lower = raw.lower()
            if lower in {"true", "false"}:
                result[key] = lower == "true"
            else:
                try:
                    result[key] = float(raw) if "." in raw else int(raw)
                except ValueError:
                    result[key] = raw
        return result

    def _verify_vectors(self, sheet: Any, profile: ProtocolProfile) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        previous_by_command: dict[str, dict[str, Any]] = {}
        header = _header_row(sheet, VECTOR_HEADERS)
        vector_ids: set[str] = set()
        for row in _active_rows(sheet, header + 1):
            vector_id = _text(sheet.cell(row, 2).value)
            if not vector_id or vector_id in vector_ids:
                raise invalid("VectorId 不能为空或重复", stage="profile_load", target=f"06_测试向量!B{row}")
            vector_ids.add(vector_id)
            direction = _text(sheet.cell(row, 3).value).upper()
            if direction not in {"TX", "RX"}:
                raise invalid("测试向量方向必须为 TX/RX", stage="profile_load", target=f"06_测试向量!C{row}")
            command_id = _text(sheet.cell(row, 4).value)
            if command_id not in profile.commands:
                raise invalid("向量引用未启用或不存在的指令", stage="profile_load", target=f"06_测试向量!D{row}")
            array_id = int(sheet.cell(row, 6).value)
            if direction == "RX":
                frame = _hex_bytes(sheet.cell(row, 8).value, location=f"06_测试向量!H{row}", expected=22)
                decoded = profile.decode(frame)
                command = profile.commands[command_id]
                if decoded['array_id'] != array_id or decoded['opcode'] != (command.response_opcode if command.response_opcode is not None else command.opcode):
                    raise invalid("RX 向量指令或阵面 ID 不匹配", stage="profile_vectors", target=f"06_测试向量!H{row}")
                expectations = self._parse_vector_inputs(sheet.cell(row, 7).value, {})
                if not expectations:
                    raise invalid("RX 向量需要填写 field=value 期望解析值", stage="profile_vectors", target=f"06_测试向量!G{row}")
                for key, expected_value in expectations.items():
                    value = decoded['fields'].get(key, {}).get('value')
                    if isinstance(value, dict):
                        value = value['logical']
                    if value != expected_value:
                        raise invalid("RX 向量解析结果不匹配", stage="profile_vectors", target=f"06_测试向量!G{row}", field=key, expected=expected_value, actual=value)
                results.append({"vector_id": vector_id, "passed": True, "direction": "RX"})
                continue
            inputs = self._parse_vector_inputs(sheet.cell(row, 7).value, previous_by_command.get(command_id, {}))
            previous_by_command[command_id] = inputs
            if profile.commands[command_id].auto_role == "CALIBRATION_WRITE":
                frame = profile.build_calibration_frame(
                    array_id=array_id,
                    spi_no=int(inputs["spi_no"]),
                    chip_no=int(inputs["chip_no"]),
                    chip_channel_index=int(inputs.get("chip_channel_no", 1)) - 1,
                    signal_path=_text(inputs.get("signal_path", "TX")),
                    enabled=bool(inputs.get("channel_enabled", True)),
                )
            else:
                frame = profile.encode(command_id, array_id, inputs)
            expected = _hex_bytes(sheet.cell(row, 8).value, location=f"06_测试向量!H{row}", expected=22)
            passed = frame == expected
            results.append({"vector_id": vector_id, "passed": passed})
            if not passed:
                raise ServiceError(
                    "DATA_INTEGRITY",
                    f"固定测试向量失败：{vector_id}",
                    "profile_vectors",
                    f"06_测试向量!H{row}",
                    {"expected": expected.hex(" ").upper(), "actual": frame.hex(" ").upper()},
                )
        return results


class AssetRegistry:
    def __init__(self) -> None:
        self.profiles: dict[str, ProtocolProfile] = {}
        self.coordinates: dict[str, Any] = {}
        self._response_assemblies: dict[tuple[str, int, int], ResponseAssembly] = {}

    def add_profile(self, profile: ProtocolProfile) -> ProtocolProfile:
        self.profiles[profile.asset_id] = profile
        return profile

    def profile(self, asset_id: str) -> ProtocolProfile:
        try:
            return self.profiles[asset_id]
        except KeyError as exc:
            raise ServiceError("NOT_FOUND", "配置包尚未加载", "assets", asset_id) from exc

    def add_coordinates(self, coordinates: Any) -> Any:
        self.coordinates[coordinates.asset_id] = coordinates
        return coordinates

    def coordinate(self, asset_id: str) -> Any:
        try:
            return self.coordinates[asset_id]
        except KeyError as exc:
            raise ServiceError("NOT_FOUND", "坐标表尚未加载", "assets", asset_id) from exc

    def decode_matching_frame(self, frame: bytes) -> dict[str, Any] | None:
        """Decode a frame only when an already-loaded profile owns its opcode.

        Raw serial bytes are logged independently by the transport.  This method deliberately
        returns ``None`` for unknown opcodes, invalid framing, or a bad CRC so arbitrary device
        diagnostics never become misleading protocol results.
        """
        if len(frame) != 22 or frame[:2] != b"\xAA\x55":
            return None
        opcode = frame[4]
        for profile in reversed(tuple(self.profiles.values())):
            command = next(
                (
                    item
                    for item in profile.commands.values()
                    if opcode in {item.opcode, item.response_opcode}
                ),
                None,
            )
            rule = profile.response_rules.get(opcode)
            if command is None and rule is None:
                continue
            try:
                decoded = profile.decode(frame)
                if rule and rule.mode == "MULTI":
                    key = (profile.asset_id, frame[5], opcode)
                    assembly = self._response_assemblies.get(key)
                    if assembly is None:
                        assembly = ResponseAssembly(rule)
                        self._response_assemblies[key] = assembly
                    if assembly.add(frame):
                        decoded = profile.decode_response(assembly.frames)
                        del self._response_assemblies[key]
                    else:
                        decoded = {"fields": {"assembly": {"display": "多帧应答", "value": f"已收 {len(assembly.frames)} 帧，等待收齐"}}}
            except ServiceError as exc:
                if rule and rule.mode == "MULTI":
                    self._response_assemblies.pop((profile.asset_id, frame[5], opcode), None)
                    return {"profile_id": profile.asset_id, "profile_name": profile.profile_name,
                            "command_id": rule.request_command_id,
                            "fields": [{"key": "assembly_error", "label": "组帧失败", "value": exc.message, "unit": None}]}
                return None
            if command is None and rule is not None:
                command = profile.commands.get(rule.request_command_id)
            fields = []
            for key, item in decoded.get("fields", {}).items():
                value = item.get("value")
                if isinstance(value, dict):
                    value = value.get("display") or value.get("logical") or value.get("wire")
                fields.append(
                    {
                        "key": key,
                        "label": item.get("display") or key,
                        "value": value,
                        "unit": item.get("unit"),
                    }
                )
            return {
                "profile_id": profile.asset_id,
                "profile_name": profile.profile_name,
                "command_id": command.command_id if command else rule.request_command_id,
                "fields": fields,
            }
        return None
