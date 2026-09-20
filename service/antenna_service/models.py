from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Evidence(str, Enum):
    REAL = "REAL"
    MIXED = "MIXED"
    SIMULATED = "SIMULATED"


class DeviceSource(str, Enum):
    REAL = "REAL"
    SIMULATED = "SIMULATED"


class DeviceState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    READY = "READY"
    BUSY = "BUSY"
    FAULT = "FAULT"
    UNKNOWN = "UNKNOWN"


class RunState(str, Enum):
    PREPARED = "PREPARED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"
    FAULTED = "FAULTED"
    UNKNOWN = "UNKNOWN"


class LoadPathRequest(BaseModel):
    path: str

    @field_validator("path")
    @classmethod
    def path_must_exist(cls, value: str) -> str:
        if not Path(value).is_file():
            raise ValueError("文件不存在")
        return str(Path(value).resolve())


class DataViewRequest(LoadPathRequest):
    frequency_index: int = Field(default=0, ge=0)
    beam_index: int = Field(default=0, ge=-1)


class DeviceConnectRequest(BaseModel):
    source: DeviceSource = DeviceSource.SIMULATED
    parameters: dict[str, Any] = Field(default_factory=dict)


class DeviceCommandRequest(BaseModel):
    action: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class BeamDefinition(BaseModel):
    """One electronic beam, independent from the turntable's mechanical position."""

    beam_id: str = Field(default="BEAM-1", min_length=1, max_length=64)
    off_axis_deg: float = Field(default=0, ge=0, le=360)
    azimuth_deg: float = Field(default=0, ge=0, le=360)
    reference_frequency_hz: float | None = Field(default=None, gt=0)


class RtcWaveRequest(BaseModel):
    """Existing antenna/beam inputs only; preloading never needs a run or output path."""
    profile_id: str
    coordinate_id: str | None = None
    test_type: Literal["PATTERN", "CALIBRATION"] = "PATTERN"
    array_id: int = Field(default=0, ge=0, le=255)
    signal_path: Literal["TX", "RX"] = "TX"
    polarization: Literal["H", "V"] = "H"
    reference_frequency_hz: float = Field(default=8e9, gt=0, allow_inf_nan=False)
    beams: list[BeamDefinition] = Field(default_factory=lambda: [BeamDefinition()], min_length=1, max_length=512)


class RunPlan(BaseModel):
    name: str = "天线测试"
    test_type: Literal["CALIBRATION", "PATTERN"]
    topology: Literal["SOFTWARE_VNA_SWEEP", "RTC_STOP_AND_GO", "RTC_CONTINUOUS"] = "SOFTWARE_VNA_SWEEP"
    beam_control_mode: Literal["SOFTWARE_DIRECT", "EXTERNAL_FIXED"] = "SOFTWARE_DIRECT"
    signal_path: Literal["TX", "RX"] = "TX"
    polarization: Literal["H", "V"] = "H"
    s_parameter: Literal["S11", "S21", "S12", "S22"] = "S21"
    profile_id: str
    coordinate_id: str
    array_id: int = Field(default=0, ge=0, le=255)
    output_directory: str
    base_filename: str = "test"
    frequency_start_hz: float = Field(default=8.0e9, gt=0)
    frequency_stop_hz: float = Field(default=8.0e9, gt=0)
    frequency_points: int = Field(default=1, ge=1, le=10001)
    if_bandwidth_hz: float = Field(default=1000, gt=0)
    source_power_dbm: float = Field(default=-10.0, ge=-120.0, le=30.0)
    averaging_enabled: bool = False
    averaging_count: int = Field(default=1, ge=1, le=65536)
    settle_ms: int = Field(default=10, ge=0, le=60000)
    azimuth_start_deg: float = -10
    azimuth_stop_deg: float = 10
    azimuth_step_deg: float = Field(default=1, gt=0)
    elevation_start_deg: float = 0
    elevation_stop_deg: float = 0
    elevation_step_deg: float = Field(default=1, gt=0)
    move_speed_deg_s: float = Field(default=1, gt=0)
    beams: list[BeamDefinition] = Field(default_factory=lambda: [BeamDefinition()], min_length=1, max_length=512)

    @field_validator("move_speed_deg_s", mode="before")
    @classmethod
    def normalize_turntable_speed(cls, value: Any) -> float:
        """Keep the device-facing scan speed positive and at four-decimal precision."""
        try:
            speed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("转台移动速度必须是数字") from exc
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("转台移动速度必须大于 0，不能为负数")
        speed = round(speed, 4)
        if speed <= 0:
            raise ValueError("转台移动速度最小为 0.0001")
        return speed

    @field_validator("base_filename")
    @classmethod
    def safe_base_filename(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ch in value for ch in '<>:"/\\|?*'):
            raise ValueError("文件名为空或包含 Windows 非法字符")
        return value

    @field_validator("output_directory")
    @classmethod
    def output_directory_exists(cls, value: str) -> str:
        path = Path(value)
        if not path.is_dir():
            raise ValueError("输出目录不存在")
        return str(path.resolve())

    @model_validator(mode="after")
    def ranges_are_ordered(self) -> "RunPlan":
        if self.topology != "SOFTWARE_VNA_SWEEP":
            if self.test_type == "CALIBRATION" and self.topology != "RTC_STOP_AND_GO":
                raise ValueError("RTC标校仅支持软件逐通道触发，不支持连续扫描")
            for value in (self.azimuth_start_deg, self.azimuth_stop_deg, self.azimuth_step_deg,
                          self.elevation_start_deg, self.elevation_stop_deg, self.elevation_step_deg):
                if not math.isfinite(value):
                    raise ValueError("RTC扫描角度与步进必须为有限数")
            if self.settle_ms > 65:
                raise ValueError("RTC波控稳定等待使用uint16微秒，当前整毫秒输入最大65 ms")
        if self.topology == "RTC_CONTINUOUS":
            interval_count = (self.azimuth_stop_deg - self.azimuth_start_deg) / self.azimuth_step_deg
            if interval_count < 1 or not math.isclose(interval_count, round(interval_count), abs_tol=1e-7, rel_tol=0):
                raise ValueError("RTC连续扫描要求终点大于起点，且方位范围能被步进整除")
            if self.azimuth_step_deg < 0.0001:
                raise ValueError("RTC连续扫描方位步进最小0.0001°")
        if self.beam_control_mode == "EXTERNAL_FIXED":
            if self.test_type != "PATTERN":
                raise ValueError("仅采集模式只用于方向图扫描")
            if len(self.beams) != 1:
                raise ValueError("仅采集模式每次运行只能记录一个外部固定波位")
        if self.frequency_stop_hz < self.frequency_start_hz:
            raise ValueError("终止频率不能小于起始频率")
        if self.frequency_points == 1 and not math.isclose(
            self.frequency_stop_hz,
            self.frequency_start_hz,
            rel_tol=0,
            abs_tol=1e-6,
        ):
            raise ValueError("单频 1 点测试要求起始频率等于终止频率")
        if self.frequency_points > 1 and math.isclose(
            self.frequency_stop_hz,
            self.frequency_start_hz,
            rel_tol=0,
            abs_tol=1e-6,
        ):
            raise ValueError("多点扫频要求终止频率大于起始频率")
        if self.averaging_enabled and self.averaging_count < 2:
            raise ValueError("启用矢网内部平均时，平均次数必须至少为 2")
        if not self.averaging_enabled and self.averaging_count != 1:
            raise ValueError("未启用矢网内部平均时，平均次数必须为 1")
        if self.azimuth_stop_deg < self.azimuth_start_deg:
            raise ValueError("方位终点不能小于起点；首版只支持单向扫描")
        if self.elevation_stop_deg < self.elevation_start_deg:
            raise ValueError("俯仰终点不能小于起点")
        return self


class CompensationRequest(BaseModel):
    signal_path: Literal["TX", "RX"]
    coordinate_id: str
    calibration_files: list[str] = Field(min_length=1, max_length=2)
    frequency_indices: list[int] = Field(min_length=1, max_length=10001)
    output_path: str
    # The physical FLASH layout has fixed six-bit phase/attenuation codes.
    phase_step_deg: Literal[5.625] = 5.625
    attenuation_step_db: Literal[0.5] = 0.5
    max_attenuation_db: Literal[31.5] = 31.5
    sidelobe_algorithm: Literal["NONE", "TAYLOR", "HAMMING", "KAISER"] = "NONE"
    sidelobe_axes: Literal["ROW", "COLUMN", "BOTH"] = "BOTH"
    taylor_nbar: int = Field(default=4, ge=2, le=20)
    taylor_sll_db: float = Field(default=30.0, ge=10.0, le=80.0)
    kaiser_beta: float = Field(default=6.0, ge=0.0, le=20.0)

    @field_validator("frequency_indices")
    @classmethod
    def frequency_selection_is_unique(cls, value: list[int]) -> list[int]:
        if any(index < 0 for index in value):
            raise ValueError("补偿频点索引不能为负数")
        if len(set(value)) != len(value):
            raise ValueError("补偿频点不能重复选择")
        return value


FlashItemName = Literal[
    "ARRAY_ID", "SWITCH_TABLE", "COORDINATE", "TX_COMPENSATION", "RX_COMPENSATION"
]


class FlashPrepareRequest(BaseModel):
    coordinate_id: str
    tx_compensation_file: str
    rx_compensation_file: str
    output_path: str
    array_id: int = Field(default=0, ge=0, le=255)
    start_addresses: dict[FlashItemName, int] = Field(
        default_factory=lambda: {
            "ARRAY_ID": 0x0000,
            "SWITCH_TABLE": 0x1000,
            "COORDINATE": 0x2000,
            "TX_COMPENSATION": 0x4000,
            "RX_COMPENSATION": 0x6000,
        }
    )

    @model_validator(mode="after")
    def flash_addresses_are_complete(self) -> "FlashPrepareRequest":
        required = {"ARRAY_ID", "SWITCH_TABLE", "COORDINATE", "TX_COMPENSATION", "RX_COMPENSATION"}
        if set(self.start_addresses) != required:
            raise ValueError("必须为五类 FLASH 数据分别提供起始地址")
        if any(address < 0 or address > 0xFFFF00 for address in self.start_addresses.values()):
            raise ValueError("FLASH 起始地址必须在 0x000000..0xFFFF00 范围内")
        if any(address % 0x100 for address in self.start_addresses.values()):
            raise ValueError("FLASH 起始地址必须按 0x100 页对齐")
        return self


class FlashWriteRequest(BaseModel):
    package_path: str
    item_name: FlashItemName
    device_id: str = "beam_controller"
