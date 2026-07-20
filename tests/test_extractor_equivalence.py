# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Max-Planck-Institut für Kernphysik
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.

"""Verify FastFlashCamExtractor (C++ deconvolution) matches ctapipe FlashCamExtractor."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ctapipe")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from ctapipe.image.extractor import FlashCamExtractor, deconvolve
from generate_events import build_flashcam_mst_subarray, generate_events

from phepex import deconvolve as fast_deconvolve
from phepex.extractor import FastFlashCamExtractor


def test_deconvolve_interior_matches(subarray_and_events):
    """C++ deconvolution == ctapipe deconvolve in the valid (non-edge) region."""
    sub, wf, _q, _t = subarray_and_events
    ex = FlashCamExtractor(subarray=sub)
    up = ex.upsampling.tel[1]
    pz = ex._get_deconvolution_parameters(1)[0][0]
    for pole in (pz, 1.0):
        ref = deconvolve(wf[0], 0.0, up, pole)
        got = fast_deconvolve(wf[0], 0.0, up, pole)
        n = ref.shape[-1]
        lo, hi = 3 * up - 2, n - 2 * (up - 1)  # documented invalid edges
        assert np.abs(ref[..., lo:hi] - got[..., lo:hi]).max() < 1e-4


def _extract_all(extractor, wf, sgc, bp):
    n = wf.shape[0]
    n_pix = wf.shape[2]
    q = np.empty((n, n_pix), np.float32)
    t = np.empty_like(q)
    for e in range(n):
        dl1 = extractor(wf[e], 1, sgc, bp)
        q[e], t[e] = dl1.image, dl1.peak_time
    return q, t


def test_extractor_bit_exact_without_nsb():
    """With no NSB, the fast extractor reproduces ctapipe on *every* pixel."""
    sub = build_flashcam_mst_subarray(22)
    wf, q, _t = generate_events(sub, n_events=60, nsb_rate_ghz=0.0, seed=11)
    n_pix = sub.tel[1].camera.geometry.n_pixels
    sgc = np.zeros(n_pix, np.int64)
    bp = np.zeros((1, n_pix), bool)
    q_ref, t_ref = _extract_all(FlashCamExtractor(subarray=sub), wf, sgc, bp)
    q_fast, t_fast = _extract_all(FastFlashCamExtractor(subarray=sub), wf, sgc, bp)
    assert np.abs(q_fast - q_ref).max() < 1e-2  # float32 accumulation only
    assert np.corrcoef(q_fast.ravel(), q_ref.ravel())[0, 1] > 0.99999


def test_signal_pixels_match_with_nsb():
    """With 200 MHz NSB, genuine signal pixels still match to ~1e-6.

    (Pure-NSB pixels legitimately diverge: argmax on noise is ill-conditioned and
    both methods just report noise there; those pixels carry no physics.)"""
    sub = build_flashcam_mst_subarray(22)
    wf, q, _t = generate_events(sub, n_events=60, nsb_rate_ghz=0.2, seed=11)
    n_pix = sub.tel[1].camera.geometry.n_pixels
    sgc = np.zeros(n_pix, np.int64)
    bp = np.zeros((1, n_pix), bool)
    q_ref, t_ref = _extract_all(FlashCamExtractor(subarray=sub), wf, sgc, bp)
    q_fast, t_fast = _extract_all(FastFlashCamExtractor(subarray=sub), wf, sgc, bp)
    lit = q > 50
    rel = np.abs(q_fast[lit] - q_ref[lit]) / np.abs(q_ref[lit])
    assert np.median(rel) < 1e-4
    assert np.percentile(rel, 99) < 1e-2
    assert np.corrcoef(q_fast[lit], q_ref[lit])[0, 1] > 0.999
    dt = np.abs(t_fast[lit] - t_ref[lit])
    assert np.percentile(dt, 99) < 0.5  # ns


@pytest.fixture(scope="module")
def subarray_and_events():
    sub = build_flashcam_mst_subarray(22)
    wf, q, t = generate_events(sub, n_events=60, seed=11)
    return sub, wf, q, t


if __name__ == "__main__":
    sub = build_flashcam_mst_subarray(22)
    wf, q, t = generate_events(sub, n_events=60, seed=11)
    test_deconvolve_interior_matches((sub, wf, q, t))
    print("deconvolve interior match: OK")
    test_extractor_bit_exact_without_nsb()
    print("extractor bit-exact (no NSB): OK")
    test_signal_pixels_match_with_nsb()
    print("signal pixels match (200 MHz NSB): OK")
    print("all equivalence checks passed")
