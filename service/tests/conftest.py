from __future__ import annotations

from pathlib import Path

import pytest

from antenna_service.coordinates import CoordinateLoader
from antenna_service.protocol.profile import AssetRegistry, ProfileLoader


REPOSITORY = Path(__file__).resolve().parents[2]
SAMPLES = REPOSITORY / "功能总结文档和必要协议" / "原协议配置包与坐标表模板与示例" / "天线通道坐标表"
PROFILE_PATH = REPOSITORY / "功能总结文档和必要协议" / "当前模板与示例" / "x_radar_天线协议配置包_V1.0.xlsx"
COORDINATE_PATH = SAMPLES / "天线通道坐标表_256通道_H极化_8SPIx8芯片x4通道_v2.1_SIMULATED.xlsx"


@pytest.fixture()
def loaded_assets():
    registry = AssetRegistry()
    profile = registry.add_profile(ProfileLoader().load(str(PROFILE_PATH)))
    coordinates = registry.add_coordinates(CoordinateLoader().load(str(COORDINATE_PATH)))
    return registry, profile, coordinates
