# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Max-Planck-Institut für Kernphysik
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.

"""Bit-exactness tests for the C++ extract_around_peak / adaptive_centroid kernels."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ctapipe")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from ctapipe.image.extractor import (
    adaptive_centroid as ct_adaptive_centroid,
)
from ctapipe.image.extractor import (
    extract_around_peak as ct_extract_around_peak,
)

from phepex import adaptive_centroid, extract_around_peak

N_UP = 88


def _random(seed, n_pix=3000):
    rng = np.random.default_rng(seed)
    wf = rng.normal(0.0, 50.0, (1, n_pix, N_UP)).astype(np.float32)
    peak = rng.integers(0, N_UP, (1, n_pix)).astype(np.int64)
    return wf, peak


@pytest.mark.parametrize("rate", [1.0, 0.25, 4.0])
@pytest.mark.parametrize("width,shift", [(7, 3), (5, 0), (10, 8)])
def test_extract_around_peak_bit_exact(rate, width, shift):
    wf, peak = _random(1)
    c_ref, t_ref = ct_extract_around_peak(wf, peak, width, shift, rate)
    c_got, t_got = extract_around_peak(wf, peak, width, shift, rate)
    assert np.array_equal(c_ref, c_got)
    assert np.array_equal(t_ref, t_got)


@pytest.mark.parametrize("limit", [0.05, 0.2, 0.5])
def test_adaptive_centroid_bit_exact(limit):
    wf, peak = _random(2)
    ref = ct_adaptive_centroid(wf, peak, limit)
    got = adaptive_centroid(wf, peak, limit)
    assert np.array_equal(ref, got)


def test_extract_window_clamped_at_edges():
    """Windows overrunning the trace edges (partial sums) still match ctapipe."""
    wf = np.random.default_rng(3).normal(0, 50, (1, 500, N_UP)).astype(np.float32)
    peak = np.zeros((1, 500), np.int64)
    peak[0, :250] = 0  # window clamped at the left edge
    peak[0, 250:] = N_UP - 1  # window clamped at the right edge
    c_ref, t_ref = ct_extract_around_peak(wf, peak, 7, 3, 0.25)
    c_got, t_got = extract_around_peak(wf, peak, 7, 3, 0.25)
    assert np.array_equal(c_ref, c_got)
    assert np.array_equal(t_ref, t_got)


def test_adaptive_centroid_negative_peak_falls_back():
    """A negative peak amplitude leaves the centroid at peak_index (both impls)."""
    wf = np.full((1, 10, N_UP), -1.0, np.float32)
    peak = np.full((1, 10), 40, np.int64)
    ref = ct_adaptive_centroid(wf, peak, 0.05)
    got = adaptive_centroid(wf, peak, 0.05)
    assert np.array_equal(ref, got)
    assert np.all(got == 40)


if __name__ == "__main__":
    for r in (1.0, 0.25, 4.0):
        for w, s in ((7, 3), (5, 0), (10, 8)):
            test_extract_around_peak_bit_exact(r, w, s)
    print("extract_around_peak bit-exact: OK")
    for lim in (0.05, 0.2, 0.5):
        test_adaptive_centroid_bit_exact(lim)
    print("adaptive_centroid bit-exact: OK")
    test_extract_window_clamped_at_edges()
    print("edge clamp: OK")
    test_adaptive_centroid_negative_peak_falls_back()
    print("negative-peak fallback: OK")
    print("all extract/centroid checks passed")
