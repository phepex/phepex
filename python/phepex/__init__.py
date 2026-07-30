# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Max-Planck-Institut für Kernphysik
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.

"""phepex -- photo-electron pulse extraction.

C++ kernels for extracting charge and timing from digitised PMT/SiPM waveforms (e.g. from
Cherenkov telescopes or Water Cherenkov Detectors): pole-zero deconvolution + upsampling,
neighbour-sum peak finding, soft clipping, window integration and leading-edge timing,
plus a waveform generator for tests and benchmarks.

``import phepex`` depends only on numpy and the compiled extension; the ctapipe-based
``phepex.extractor`` is a separate submodule imported on demand.
"""

from importlib.metadata import PackageNotFoundError, version

from ._core import generate_waveforms
from .kernels import (
    adaptive_centroid,
    deconvolve,
    deconvolve_valid_range,
    extract_around_peak,
    neighbor_peak_indices,
    pos_soft_clip,
    preprocess,
    preprocess_valid_range,
)

__all__ = [
    "generate_waveforms",
    "deconvolve",
    "preprocess",
    "preprocess_valid_range",
    "pos_soft_clip",
    "neighbor_peak_indices",
    "extract_around_peak",
    "adaptive_centroid",
    "deconvolve_valid_range",
]
try:
    __version__ = version("phepex")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"
