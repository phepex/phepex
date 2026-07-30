# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Max-Planck-Institut für Kernphysik
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.

"""Guard tests for the NumPy-facing kernel wrappers / bindings.

These cover input handling that does not need ctapipe or scipy: leading-axis promotion
of sub-3D inputs, and the bounds/shape validation that stops an out-of-range sample
window or a mismatched neighbour matrix from reading past the raw buffers.
"""

import numpy as np
import pytest

from phepex import deconvolve, neighbor_peak_indices, pos_soft_clip


class _CSR:
    """Minimal duck-typed stand-in for a scipy CSR matrix (indptr/indices only)."""

    def __init__(self, indptr, indices):
        self.indptr = np.asarray(indptr, dtype=np.int32)
        self.indices = np.asarray(indices, dtype=np.int32)


def _self_neighbour_csr(n_pix):
    """A valid (n_pix, n_pix) CSR where each pixel is its own sole neighbour."""
    return _CSR(np.arange(n_pix + 1, dtype=np.int32), np.arange(n_pix, dtype=np.int32))


def test_deconvolve_2d_input_promotes_leading_axis():
    """A 2D (n_pix, n_samples) input becomes (1, n_pix, n_samples), == explicit 3D."""
    wf2d = np.random.default_rng(0).normal(0, 10, (6, 20)).astype(np.float32)
    up = 4
    out2d = deconvolve(wf2d, 0.0, up, 0.5)
    out3d = deconvolve(wf2d[None], 0.0, up, 0.5)
    assert out2d.shape == (1, 6, 20 * up)
    assert np.array_equal(out2d, out3d)


def test_deconvolve_upsampling_one_pole_zero_zero_preserves_first_sample():
    """up==1, pole_zero==0 is just wf-baseline; the first sample is NOT zeroed.

    Sample 0 must be preserved, consistent with deconvolve_valid_range(1, n, 0) == (0, n):
    every sample is valid when there is no deconvolution.
    """
    wf = np.random.default_rng(2).normal(50, 10, (1, 4, 12)).astype(np.float32)
    baseline = 5.0
    out = deconvolve(wf, baseline, 1, 0.0)
    expected = (wf - np.float32(baseline)).astype(np.float32)
    assert np.array_equal(out, expected)  # includes sample 0
    assert out[0, 0, 0] != 0.0


@pytest.mark.parametrize("bad", [0, -1])
def test_deconvolve_rejects_upsampling_below_one(bad):
    """upsampling < 1 is out of contract and rejected (no silent zero-width output)."""
    wf = np.ones((1, 4, 20), np.float32)
    with pytest.raises(ValueError):
        deconvolve(wf, 0.0, bad, 0.5)


def test_pos_soft_clip_2d_input_promotes_leading_axis():
    """sub-3D inputs get the new axis PREPENDED (channel), not appended"""
    wf2d = np.random.default_rng(1).normal(0, 10, (6, 20)).astype(np.float32)
    out2d = pos_soft_clip(wf2d, 2.0)
    out3d = pos_soft_clip(wf2d[None], 2.0)
    assert out2d.shape == (1, 6, 20)
    assert np.array_equal(out2d, out3d)


@pytest.mark.parametrize("lo,hi", [(-1, 5), (0, 999), (5, 3)])
def test_pos_soft_clip_rejects_bad_window(lo, hi):
    """pos_soft_clip rejects an out-of-range sample window"""
    wf = np.ones((1, 4, 20), np.float32)
    with pytest.raises(ValueError):
        pos_soft_clip(wf, 1.0, lo, hi)


def test_pos_soft_clip_accepts_full_and_valid_windows():
    wf = np.ones((1, 4, 20), np.float32)
    assert pos_soft_clip(wf, 1.0).shape == (1, 4, 20)  # (0, 0) == full trace
    assert pos_soft_clip(wf, 1.0, 2, 18).shape == (1, 4, 20)  # valid sub-window


def test_neighbor_rejects_mismatched_matrix():
    """neighbor_peak_indices rejects a mismatched matrix / broken_pixels"""
    n_pix = 5
    wf = np.ones((1, n_pix, 20), np.float32)
    bp = np.zeros((1, n_pix), bool)
    wrong = _self_neighbour_csr(n_pix + 2)  # matrix built for a different pixel count
    with pytest.raises(ValueError):
        neighbor_peak_indices(wf, wrong, 0, bp)


def test_neighbor_rejects_mismatched_broken_pixels():
    n_pix = 5
    wf = np.ones((1, n_pix, 20), np.float32)
    good = _self_neighbour_csr(n_pix)
    bp_wrong = np.zeros((1, n_pix + 3), bool)
    with pytest.raises(ValueError):
        neighbor_peak_indices(wf, good, 0, bp_wrong)


def test_neighbor_accepts_matching_shapes():
    n_pix = 5
    # rising ramp so the per-pixel argmax is deterministic (last sample)
    wf = np.tile(np.arange(20, dtype=np.float32), (1, n_pix, 1))
    good = _self_neighbour_csr(n_pix)
    bp = np.zeros((1, n_pix), bool)
    out = neighbor_peak_indices(wf, good, 0, bp)
    assert out.shape == (1, n_pix)
    assert np.all(out == 19)
