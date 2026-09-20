from __future__ import annotations

import numpy as np
import pytest

from antenna_service.models import CompensationRequest
from antenna_service.workflows.aperture_weights import build_aperture_weights


def _request(**overrides) -> CompensationRequest:
    values = {
        "signal_path": "TX",
        "coordinate_id": "coordinates",
        "calibration_files": ["calibration.hdf5"],
        "frequency_indices": [0],
        "output_path": "compensation.hdf5",
    }
    values.update(overrides)
    return CompensationRequest(**values)


def test_uniform_weighting_preserves_existing_compensation(loaded_assets):
    _, _, coordinates = loaded_assets
    channels = coordinates.for_polarization("H")
    result = build_aperture_weights(channels, _request())
    assert np.allclose(result.linear, 1.0)
    assert np.allclose(result.attenuation_db, 0.0)


@pytest.mark.parametrize("algorithm", ["TAYLOR", "HAMMING", "KAISER"])
def test_supported_windows_are_symmetric_and_peak_normalized(loaded_assets, algorithm):
    _, _, coordinates = loaded_assets
    channels = coordinates.for_polarization("H")
    result = build_aperture_weights(
        channels,
        _request(sidelobe_algorithm=algorithm, sidelobe_axes="BOTH"),
    )
    by_position = {(channel.grid_row, channel.grid_column): result.linear[index] for index, channel in enumerate(channels)}
    rows = sorted({channel.grid_row for channel in channels})
    columns = sorted({channel.grid_column for channel in channels})
    matrix = np.asarray([[by_position[(row, column)] for column in columns] for row in rows])
    assert np.max(matrix) == pytest.approx(1.0)
    assert np.allclose(matrix, np.flip(matrix, axis=0))
    assert np.allclose(matrix, np.flip(matrix, axis=1))
    assert np.all(matrix > 0)


def test_hamming_two_dimensional_taper_exposes_hardware_limit(loaded_assets):
    _, _, coordinates = loaded_assets
    channels = coordinates.for_polarization("H")
    result = build_aperture_weights(
        channels,
        _request(sidelobe_algorithm="HAMMING", sidelobe_axes="BOTH"),
    )
    assert np.max(result.attenuation_db) > 31.5
