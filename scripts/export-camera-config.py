# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Max-Planck-Institut für Kernphysik
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.

"""Export a camera configuration from ctapipe into the flat text format consumed by the
C++ microbenchmark (benchmarks/cpp/microbench.cpp).

The exported file holds more than geometry: pixel count and neighbour adjacency (CSR)
alongside readout scalars (sampling rate, reference pulse shape, sample widths, default
number of samples). The microbenchmark links only libphepex and has no ctapipe (or any
Python) dependency at run time, so this configuration is frozen once into a text file that
ships with the repository. Re-run this script to regenerate that file after a ctapipe
upgrade or to target a different camera:

    python3 scripts/export-camera-config.py --camera FlashCam \\
        --out benchmarks/flashcam-config.txt

Only the fields the kernels consume are written: pixel count, neighbour adjacency (CSR),
the reference pulse shape (for generate_waveforms), and the sampling/readout scalars.
Pixel coordinates are omitted because no kernel reads them; the neighbour matrix already
encodes the connectivity derived from those coordinates.

The output format is line-oriented and whitespace-tokenised (see the parser in
microbench.cpp). Lines beginning with '#' are comments. Scalars are `key value`; arrays
are introduced by a `key` line on its own followed by the declared number of tokens (which
may span multiple lines). Array lengths are fixed by a preceding count field:
`indptr` has num_pixels+1 entries, `indices` has neighbor_nnz entries, and
`reference_pulse` has num_reference_pulse entries.
"""

from __future__ import annotations

import argparse
import warnings

import astropy.units as u
import numpy as np


def export(camera: str, n_samples: int, out_path: str) -> None:
    from ctapipe.instrument import CameraGeometry, CameraReadout

    # Bundled reference data is exactly what a synthetic benchmark needs, so silence
    # ctapipe's advisory that .from_name may differ from the deployed instrument.
    try:
        from ctapipe.instrument import FromNameWarning

        warnings.simplefilter("ignore", FromNameWarning)
    except ImportError:
        pass

    geom = CameraGeometry.from_name(camera)
    readout = CameraReadout.from_name(camera)

    n_pixels = int(geom.n_pixels)
    # scipy CSR of the boolean neighbour matrix: indptr[p]..indptr[p+1] index the columns
    # (neighbour pixel ids) of row p. int32 matches the kernel's CSR pointer type.
    nm = geom.neighbor_matrix_sparse
    indptr = np.ascontiguousarray(nm.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(nm.indices, dtype=np.int64)
    assert indptr.shape == (n_pixels + 1,)
    nnz = int(indptr[-1])
    assert indices.shape == (nnz,)

    sampling_rate_ghz = float(readout.sampling_rate.to_value(u.GHz))
    ref_sample_width_ns = float(readout.reference_pulse_sample_width.to_value(u.ns))
    # Single gain channel for FlashCam; take channel 0 of the reference pulse shape.
    ref_pulse = np.ascontiguousarray(readout.reference_pulse_shape[0], dtype=np.float64)

    import ctapipe

    degree = np.diff(indptr)
    lines: list[str] = []
    lines.append(f"# Camera configuration for phepex C++ microbenchmarks: {camera}")
    lines.append(f"# Exported from ctapipe {ctapipe.__version__}")
    lines.append("# scripts/export-camera-config.py. Do not edit by hand.")
    lines.append("#")
    lines.append(
        f"# Neighbour adjacency: {nnz} directed edges, "
        f"degree min/mean/max = {degree.min()}/{degree.mean():.2f}/{degree.max()}."
    )
    lines.append("")
    lines.append(f"name {camera}")
    lines.append(f"num_pixels {n_pixels}")
    lines.append(f"num_samples {n_samples}")
    lines.append(f"sampling_rate_ghz {sampling_rate_ghz!r}")
    lines.append(f"ref_sample_width_ns {ref_sample_width_ns!r}")
    lines.append(f"num_reference_pulse {ref_pulse.size}")
    lines.append(f"neighbor_nnz {nnz}")
    lines.append("")

    def _emit_array(key: str, values, fmt) -> None:
        # 16 tokens per line: keeps lines under ~120 cols and the file diffable.
        lines.append(key)
        row: list[str] = []
        for v in values:
            row.append(fmt(v))
            if len(row) == 16:
                lines.append(" ".join(row))
                row = []
        if row:
            lines.append(" ".join(row))
        lines.append("")

    _emit_array("reference_pulse", ref_pulse, lambda v: repr(float(v)))
    _emit_array("indptr", indptr, lambda v: str(int(v)))
    _emit_array("indices", indices, lambda v: str(int(v)))

    with open(out_path, "w") as f:
        f.write("\n".join(lines).rstrip("\n") + "\n")

    print(
        f"wrote {out_path}: {camera}, {n_pixels} pixels, {n_samples} samples, "
        f"{nnz} neighbour edges, {ref_pulse.size}-sample reference pulse"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", default="FlashCam", help="ctapipe camera name")
    p.add_argument(
        "--n-samples",
        type=int,
        default=22,
        help="readout window length in samples (metadata; the FlashCam default is 22)",
    )
    p.add_argument(
        "--out",
        default="benchmarks/flashcam-config.txt",
        help="output text file path",
    )
    args = p.parse_args()
    export(args.camera, args.n_samples, args.out)


if __name__ == "__main__":
    main()
