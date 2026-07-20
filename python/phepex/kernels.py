# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Max-Planck-Institut für Kernphysik
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.

"""NumPy-friendly wrappers over the compiled ``phepex._core`` kernels.

These depend only on numpy and the C++ extension (no ctapipe), so ``import phepex`` stays
lightweight. Each mirrors the corresponding ctapipe FlashCamExtractor step.
"""

from __future__ import annotations

import numpy as np

from . import _core


def _as_3d(waveforms):
    """Promote ``waveforms`` to a contiguous float32 (n_channels, n_pix, n_samples) array.

    Missing leading axes are *prepended* (a 2D ``(n_pix, n_samples)`` array becomes
    ``(1, n_pix, n_samples)``), unlike ``np.atleast_3d`` which appends a trailing axis.
    """
    wf = np.asarray(waveforms, dtype=np.float32)
    while wf.ndim < 3:
        wf = wf[np.newaxis, ...]
    return np.ascontiguousarray(wf, dtype=np.float32)


__all__ = [
    "deconvolve",
    "pos_soft_clip",
    "neighbor_peak_indices",
    "extract_around_peak",
    "adaptive_centroid",
    "deconvolve_valid_range",
]


def deconvolve(waveforms, baselines, upsampling, pole_zero):
    """Pole-zero deconvolution + upsampling; drop-in for ctapipe's ``deconvolve``.

    ``waveforms`` (n_channels, n_pix, n_samples); ``baselines`` scalar or per-pixel;
    ``upsampling`` >= 1. Returns float32 (n_channels, n_pix, n_samples*upsampling).
    """
    wf = _as_3d(waveforms)

    baselines = np.asarray(baselines, dtype=np.float32)
    if baselines.ndim == 0:
        offset = float(baselines)
    else:
        wf = np.ascontiguousarray(wf - baselines.reshape(1, -1, 1), dtype=np.float32)
        offset = 0.0

    if upsampling <= 1:
        # rare path (FlashCam default is 4): replicate ctapipe's numpy version exactly
        d = wf - np.float32(offset)
        d[..., 1:] -= np.float32(pole_zero) * d[..., :-1]
        d[..., 0] = 0
        return d

    return _core.deconvolve_upsample(wf, int(upsampling), float(pole_zero), offset)


def deconvolve_valid_range(upsampling, n_samples, pole_zero):
    """Trustworthy (non-edge) output-sample range ``(lo, hi)`` of ``deconvolve``.

    Delegates to the C++ ``deconvolve_valid_range`` so the edge-trim rule has a single
    source of truth (the same one the C++/test path uses).
    """
    return _core.deconvolve_valid_range(int(upsampling), int(n_samples), float(pole_zero))


def pos_soft_clip(waveforms, scale, sample_lo=0, sample_hi=0):
    """Positive soft clip ``max(clip(waveforms/scale), 0)`` over ``[sample_lo, sample_hi)``.

    ``(0, 0)`` means the full trace. The soft clip already bounds the result to
    ``(-1, 1)``, so only negatives are clamped (to 0). Equivalent to ctapipe
    ``FlashCamExtractor.clip(waveforms / scale)``.
    """  # noqa: E501
    wf = _as_3d(waveforms)
    return _core.pos_soft_clip(wf, float(scale), int(sample_lo), int(sample_hi))


def neighbor_peak_indices(
    waveforms, neighbors, local_weight, broken_pixels, sample_lo=0, sample_hi=0
):
    """Per-pixel peak sample index of the neighbour-summed waveform.

    Equivalent to ctapipe's ``neighbor_average_maximum``. ``neighbors`` is a scipy CSR
    matrix (``geometry.neighbor_matrix_sparse``); its ``indices``/``indptr`` are cast to
    int32. Returns int64 (n_channels, n_pix).
    """
    wf = np.ascontiguousarray(waveforms, dtype=np.float32)
    indptr = np.ascontiguousarray(neighbors.indptr, dtype=np.int32)
    indices = np.ascontiguousarray(neighbors.indices, dtype=np.int32)
    bp = np.ascontiguousarray(broken_pixels, dtype=bool)
    return _core.neighbor_peak_indices(
        wf, indptr, indices, int(local_weight), bp, int(sample_lo), int(sample_hi)
    )


def extract_around_peak(waveforms, peak_index, width, shift, sampling_rate_ghz):
    """Window integration + weighted peak time; ctapipe's ``extract_around_peak``.

    Returns ``(charge, peak_time)`` float32; ``peak_time`` in ns.
    """
    wf = np.ascontiguousarray(waveforms, dtype=np.float32)
    pk = np.ascontiguousarray(peak_index, dtype=np.int64)
    return _core.extract_around_peak(
        wf, pk, int(width), int(shift), float(sampling_rate_ghz)
    )


def adaptive_centroid(waveforms, peak_index, rel_descend_limit):
    """Leading-edge centroid in *sample* units; ctapipe's ``adaptive_centroid``.

    float32 (n_channels, n_pix).
    """
    wf = np.ascontiguousarray(waveforms, dtype=np.float32)
    pk = np.ascontiguousarray(peak_index, dtype=np.int64)
    return _core.adaptive_centroid(wf, pk, float(rel_descend_limit))
