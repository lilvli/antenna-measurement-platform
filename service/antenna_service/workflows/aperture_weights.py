from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from antenna_service.coordinates import Channel
from antenna_service.errors import ServiceError
from antenna_service.models import CompensationRequest


@dataclass(frozen=True, slots=True)
class ApertureWeights:
    """Per-channel aperture weights before hardware attenuation quantization."""

    linear: np.ndarray
    attenuation_db: np.ndarray


def _taylor(length: int, nbar: int, sidelobe_level_db: float) -> np.ndarray:
    """Return a symmetric Taylor window normalized to a peak value of one.

    This is the standard n-bar Taylor formulation.  It is implemented locally
    because the service already depends on NumPy and adding SciPy solely for one
    window would materially increase the Windows deployment size.
    """
    if length <= 1:
        return np.ones(length, dtype=np.float64)
    if nbar > length:
        raise ServiceError(
            "INVALID_REQUEST",
            "Taylor nbar 不能大于所作用轴的点数",
            "compensation_weighting",
            details={"nbar": nbar, "axis_points": length},
        )

    a = np.arccosh(10 ** (sidelobe_level_db / 20.0)) / np.pi
    s2 = nbar**2 / (a**2 + (nbar - 0.5) ** 2)
    orders = np.arange(1, nbar, dtype=np.float64)
    order_squared = orders**2
    coefficients = np.empty(nbar - 1, dtype=np.float64)

    for index in range(nbar - 1):
        sign = 1.0 if index % 2 == 0 else -1.0
        numerator = sign * np.prod(
            1.0 - order_squared[index] / (s2 * (a**2 + (orders - 0.5) ** 2))
        )
        denominator = 2.0
        if index:
            denominator *= np.prod(1.0 - order_squared[index] / order_squared[:index])
        if index + 1 < len(orders):
            denominator *= np.prod(1.0 - order_squared[index] / order_squared[index + 1 :])
        coefficients[index] = numerator / denominator

    positions = np.arange(length, dtype=np.float64)
    cosines = np.cos(
        2.0
        * np.pi
        * orders[:, np.newaxis]
        * (positions - length / 2.0 + 0.5)
        / length
    )
    window = 1.0 + 2.0 * np.dot(coefficients, cosines)
    peak = float(np.max(window))
    if not math.isfinite(peak) or peak <= 0 or np.any(window <= 0):
        raise ServiceError(
            "INVALID_REQUEST",
            "Taylor 参数未生成有效的正孔径权值",
            "compensation_weighting",
            details={"nbar": nbar, "sll_db": sidelobe_level_db, "axis_points": length},
        )
    return window / peak


def _one_dimensional_window(length: int, request: CompensationRequest) -> np.ndarray:
    if length <= 1 or request.sidelobe_algorithm == "NONE":
        return np.ones(length, dtype=np.float64)
    if request.sidelobe_algorithm == "TAYLOR":
        return _taylor(length, request.taylor_nbar, request.taylor_sll_db)
    if request.sidelobe_algorithm == "HAMMING":
        window = np.hamming(length)
    elif request.sidelobe_algorithm == "KAISER":
        window = np.kaiser(length, request.kaiser_beta)
    else:  # Pydantic rejects this, but keep the computation boundary explicit.
        raise ServiceError(
            "INVALID_REQUEST",
            "不支持的副瓣压低算法",
            "compensation_weighting",
            request.sidelobe_algorithm,
        )
    peak = float(np.max(window))
    if not math.isfinite(peak) or peak <= 0 or np.any(window <= 0):
        raise ServiceError(
            "INVALID_REQUEST",
            "副瓣算法产生了硬件无法表示的零或负权值",
            "compensation_weighting",
            request.sidelobe_algorithm,
        )
    return np.asarray(window / peak, dtype=np.float64)


def build_aperture_weights(channels: list[Channel], request: CompensationRequest) -> ApertureWeights:
    """Map row/column windows to channels, independently for each polarization.

    A two-dimensional taper is the outer product of its row and column windows.
    Disabled channels retain a zero linear weight and NaN attenuation because no
    phase/attenuation command is generated for them.
    """
    linear = np.zeros(len(channels), dtype=np.float64)
    attenuation_db = np.full(len(channels), np.nan, dtype=np.float64)

    for polarization in sorted({channel.polarization for channel in channels}):
        indexed = [(index, channel) for index, channel in enumerate(channels) if channel.polarization == polarization]
        rows = sorted({channel.grid_row for _, channel in indexed})
        columns = sorted({channel.grid_column for _, channel in indexed})
        row_index = {value: index for index, value in enumerate(rows)}
        column_index = {value: index for index, value in enumerate(columns)}
        row_window = (
            _one_dimensional_window(len(rows), request)
            if request.sidelobe_axes in {"ROW", "BOTH"}
            else np.ones(len(rows), dtype=np.float64)
        )
        column_window = (
            _one_dimensional_window(len(columns), request)
            if request.sidelobe_axes in {"COLUMN", "BOTH"}
            else np.ones(len(columns), dtype=np.float64)
        )

        for index, channel in indexed:
            if not channel.enabled:
                continue
            weight = float(row_window[row_index[channel.grid_row]] * column_window[column_index[channel.grid_column]])
            linear[index] = weight
            attenuation_db[index] = -20.0 * math.log10(weight)

    return ApertureWeights(linear=linear, attenuation_db=attenuation_db)
