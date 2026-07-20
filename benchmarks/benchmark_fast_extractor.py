# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Max-Planck-Institut für Kernphysik
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.

"""Estimate the time saved by replacing FlashCamExtractor's pole-zero deconvolution
with the C++ (`phepex`) implementation.

Reports three numbers:
  1. isolated `deconvolve` step: ctapipe (scipy) vs C++
  2. full FlashCamExtractor vs FastFlashCamExtractor (leading-edge timing ON, default)
  3. same with leading-edge timing OFF
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from ctapipe.image.extractor import FlashCamExtractor
from ctapipe.image.extractor import deconvolve as ct_deconvolve
from generate_events import build_flashcam_mst_subarray, generate_events

from phepex import deconvolve as cpp_deconvolve
from phepex.extractor import FastFlashCamExtractor


def _time_loop(fn, waveforms, tel_id, sgc, bp, warmup=3):
    for _ in range(warmup):
        fn(waveforms[0], tel_id, sgc, bp)
    t0 = time.perf_counter()
    for e in range(waveforms.shape[0]):
        fn(waveforms[e], tel_id, sgc, bp)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tel_id = 1
    sub = build_flashcam_mst_subarray(22)
    n_pix = sub.tel[tel_id].camera.geometry.n_pixels
    wf, q, _t = generate_events(sub, n_events=args.events, seed=args.seed)
    sgc = np.zeros(n_pix, np.int64)
    bp = np.zeros((1, n_pix), bool)

    ex = FlashCamExtractor(subarray=sub)
    up = ex.upsampling.tel[tel_id]
    pz = ex._get_deconvolution_parameters(tel_id)[0][0]
    print(f"# events={args.events}  upsampling={up}  pole_zero={pz:.4f}")

    # 1. isolated deconvolution step (called twice per event with leading-edge)
    for _ in range(3):
        ct_deconvolve(wf[0], 0.0, up, pz)
        cpp_deconvolve(wf[0], 0.0, up, pz)
    t0 = time.perf_counter()
    for e in range(args.events):
        ct_deconvolve(wf[e], 0.0, up, pz)
        ct_deconvolve(wf[e], 0.0, up, 1)
    ct_dt = time.perf_counter() - t0
    t0 = time.perf_counter()
    for e in range(args.events):
        cpp_deconvolve(wf[e], 0.0, up, pz)
        cpp_deconvolve(wf[e], 0.0, up, 1)
    cpp_dt = time.perf_counter() - t0
    print("\n=== 1. deconvolution step (2 calls/event) ===")
    print(f"  ctapipe (scipy): {ct_dt / args.events * 1e6:8.1f} us/event")
    print(
        f"  C++            : {cpp_dt / args.events * 1e6:8.1f} us/event   "
        f"({ct_dt / cpp_dt:.1f}x faster)"
    )

    # 2 & 3. full extractor, both timing modes
    for le in (True, False):
        ref = FlashCamExtractor(subarray=sub, leading_edge_timing=le)
        fast = FastFlashCamExtractor(subarray=sub, leading_edge_timing=le)
        t_ref = _time_loop(ref, wf, tel_id, sgc, bp)
        t_fast = _time_loop(fast, wf, tel_id, sgc, bp)
        tag = "leading_edge=ON (default)" if le else "leading_edge=OFF"
        print(f"\n=== full extractor, {tag} ===")
        print(
            f"  ctapipe FlashCamExtractor : {t_ref / args.events * 1e6:8.1f} us/event  "
            f"({args.events / t_ref:6.0f} ev/s)"
        )
        print(
            f"  FastFlashCamExtractor     : {t_fast / args.events * 1e6:8.1f} us/event  "
            f"({args.events / t_fast:6.0f} ev/s)"
        )
        print(
            f"  speedup: {t_ref / t_fast:.2f}x   "
            f"(saved {(t_ref - t_fast) / args.events * 1e6:.0f} us/event, "
            f"{(1 - t_fast / t_ref) * 100:.0f}% of runtime)"
        )


if __name__ == "__main__":
    main()
