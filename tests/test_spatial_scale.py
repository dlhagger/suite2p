import logging

import numpy as np
import pytest

from suite2p.detection.sparsedetect import (
    EstimateMode,
    estimate_spatial_scale,
    find_best_scale,
)


def _peak_coordinates(n_peaks, shape=(128, 128)):
    """Return peaks spaced beyond the estimator's 11-pixel max filter."""
    coords = [(y, x) for y in range(6, shape[0] - 6, 12)
              for x in range(6, shape[1] - 6, 12)]
    return coords[:n_peaks]


def test_scale_zero_peaks_are_downweighted_not_treated_as_failure(caplog):
    maps = np.zeros((5, 128, 128), dtype=np.float32)
    for i, (y, x) in enumerate(_peak_coordinates(30)):
        amplitude = 10 - i * 0.1
        maps[0, y, x] = amplitude
        maps[1, y, x] = amplitude * 0.8

    caplog.set_level(logging.INFO)
    scale, mode = find_best_scale(maps, spatial_scale=0)

    assert scale == 1
    assert mode is EstimateMode.Estimated
    assert "scale0_peaks=30" in caplog.text


def test_clear_cell_scale_is_estimated():
    maps = np.zeros((5, 128, 128), dtype=np.float32)
    for i, (y, x) in enumerate(_peak_coordinates(30)):
        amplitude = 10 - i * 0.1
        maps[1, y, x] = amplitude * 0.2
        maps[3, y, x] = amplitude

    assert estimate_spatial_scale(maps) == 3


def test_pixel_scale_noise_without_cell_evidence_is_inconclusive():
    maps = np.zeros((5, 128, 128), dtype=np.float32)
    for i, (y, x) in enumerate(_peak_coordinates(30)):
        amplitude = 10 - i * 0.1
        maps[0, y, x] = amplitude
        maps[1, y, x] = amplitude * 0.01

    assert estimate_spatial_scale(maps) is None


def test_ambiguous_cell_scales_are_inconclusive():
    maps = np.zeros((5, 128, 128), dtype=np.float32)
    for i, (y, x) in enumerate(_peak_coordinates(40)):
        amplitude = 10 - i * 0.1
        scale = 1 if i % 2 == 0 else 2
        maps[scale, y, x] = amplitude

    assert estimate_spatial_scale(maps) is None


def test_no_positive_peaks_uses_safe_fallback():
    maps = np.zeros((5, 32, 32), dtype=np.float32)

    with pytest.warns(UserWarning, match="inconclusive"):
        scale, mode = find_best_scale(maps, spatial_scale=0)

    assert scale == 1
    assert mode is EstimateMode.Forced


def test_forced_scale_is_still_clamped():
    maps = np.zeros((5, 32, 32), dtype=np.float32)

    scale, mode = find_best_scale(maps, spatial_scale=99)

    assert scale == 4
    assert mode is EstimateMode.Forced
