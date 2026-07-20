# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Max-Planck-Institut für Kernphysik
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.

"""Equivalence tests for the C++ neighbour clip + peak-search kernels vs ctapipe."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ctapipe")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from ctapipe.image.extractor import FlashCamExtractor, neighbor_average_maximum
from generate_events import build_flashcam_mst_subarray, generate_events

from phepex import (
    deconvolve,
    deconvolve_valid_range,
    neighbor_peak_indices,
    pos_soft_clip,
)


@pytest.fixture(scope="module")
def setup():
    sub = build_flashcam_mst_subarray(22)
    wf, q, t = generate_events(sub, n_events=40, seed=9)
    ex = FlashCamExtractor(subarray=sub)
    up = ex.upsampling.tel[1]
    pz = ex._get_deconvolution_parameters(1)[0][0]
    nsb_clip = float(ex.neighbour_sum_clipping.tel[1])
    neighbors = sub.tel[1].camera.geometry.neighbor_matrix_sparse
    # deconvolved + clipped waveforms (realistic peak-search input) for each event
    t_wf = [deconvolve(wf[e], 0.0, up, pz) for e in range(wf.shape[0])]
    return {
        "sub": sub,
        "wf": wf,
        "q": q,
        "up": up,
        "pz": pz,
        "nsb_clip": nsb_clip,
        "neighbors": neighbors,
        "t_wf": t_wf,
    }


@pytest.mark.parametrize("local_weight", [0, 1, 4])
@pytest.mark.parametrize("with_broken", [False, True])
def test_neighbor_peak_indices_full_range_exact(setup, local_weight, with_broken):
    """Full-range C++ neighbor_peak_indices == ctapipe neighbor_average_maximum
    (bit-exact)."""
    neighbors = setup["neighbors"]
    n_pix = setup["sub"].tel[1].camera.geometry.n_pixels
    rng = np.random.default_rng(0)
    bp = np.zeros((1, n_pix), bool)
    if with_broken:
        bp[0, rng.choice(n_pix, 40, replace=False)] = True
    for t in setup["t_wf"][:10]:
        nn = np.ascontiguousarray(t, np.float32)
        ref = neighbor_average_maximum(
            nn,
            neighbors_indices=neighbors.indices,
            neighbors_indptr=neighbors.indptr,
            local_weight=local_weight,
            broken_pixels=bp,
        )
        got = neighbor_peak_indices(nn, neighbors, local_weight, bp)  # full range (0,0)
        assert np.array_equal(ref, got)


def test_pos_soft_clip_exact(setup):
    """C++ pos_soft_clip(t, scale) == FlashCamExtractor.clip(t/scale)."""
    scale = setup["nsb_clip"]
    for t in setup["t_wf"][:10]:
        ref = FlashCamExtractor.clip(t / scale).astype(np.float32)
        got = pos_soft_clip(t, scale)  # full range
        assert np.abs(ref - got).max() < 1e-6


def test_valid_range_returns_interior_index(setup):
    """Restricted peak indices always lie inside the valid range."""
    up, pz, scale = setup["up"], setup["pz"], setup["nsb_clip"]
    neighbors = setup["neighbors"]
    lo, hi = deconvolve_valid_range(up, 22, pz)
    bp = np.zeros((1, setup["sub"].tel[1].camera.geometry.n_pixels), bool)
    nn = pos_soft_clip(setup["t_wf"][0], scale, lo, hi)
    peaks = neighbor_peak_indices(nn, neighbors, 0, bp, lo, hi)
    assert peaks.min() >= lo and peaks.max() < hi


def test_empty_input_guard(setup):
    neighbors = setup["neighbors"]
    empty = np.zeros((0, 0, 0), np.float32)
    r = neighbor_peak_indices(empty, neighbors, 0, np.zeros((0, 0), bool))
    assert r.shape == (0, 0)


if __name__ == "__main__":
    sub = build_flashcam_mst_subarray(22)
    wf, q, t = generate_events(sub, n_events=40, seed=9)
    ex = FlashCamExtractor(subarray=sub)
    s = {
        "sub": sub,
        "wf": wf,
        "q": q,
        "up": ex.upsampling.tel[1],
        "pz": ex._get_deconvolution_parameters(1)[0][0],
        "nsb_clip": float(ex.neighbour_sum_clipping.tel[1]),
        "neighbors": sub.tel[1].camera.geometry.neighbor_matrix_sparse,
        "t_wf": [
            deconvolve(
                wf[e],
                0.0,
                ex.upsampling.tel[1],
                ex._get_deconvolution_parameters(1)[0][0],
            )
            for e in range(40)
        ],
    }
    for lw in (0, 1, 4):
        for br in (False, True):
            test_neighbor_peak_indices_full_range_exact(s, lw, br)
    print("neighbor_peak_indices full-range exact: OK")
    test_pos_soft_clip_exact(s)
    print("pos_soft_clip exact: OK")
    test_valid_range_returns_interior_index(s)
    print("valid-range index: OK")
    test_empty_input_guard(s)
    print("empty-input guard: OK")
    print("all neighbor kernel checks passed")
