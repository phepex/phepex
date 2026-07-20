# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Max-Planck-Institut für Kernphysik
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.

"""phepex -- photo-electron pulse extraction.

C++-accelerated kernels for extracting charge and timing from digitised PMT/SiPM (e.g.
Cherenkov telescopes or Water Cherenkov Detectors) waveforms, plus a fast waveform
generator for testing.

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
)

__all__ = [
    "generate_waveforms",
    "deconvolve",
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
