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


def _per_pixel(x, n_ch, n_pix):
    """Broadcast a scalar / ``(n_pix,)`` / ``(n_ch, n_pix)`` value to contiguous float32.

    Returns a length ``n_ch*n_pix`` array (row-major over ``(channel, pixel)``). A scalar
    fills every entry; a ``(n_pix,)`` array is broadcast across channels.
    ``np.broadcast_to`` raises ``ValueError`` for any other shape.
    """
    a = np.asarray(x, dtype=np.float32)
    a = np.broadcast_to(a, (n_ch, n_pix))
    return np.ascontiguousarray(a, dtype=np.float32).reshape(-1)


__all__ = [
    "deconvolve",
    "preprocess",
    "preprocess_valid_range",
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


def preprocess(
    waveforms, upsampling, pole_zero, smoothing_fwhm=0.0, baseline=0.0, scale=1.0
):
    """Upsample + pole-zero deconvolution + optional smoothing, per (channel, pixel).

    The full single-waveform preprocessing step of ctapipe's ``FlashCamExtractor`` applied
    to every ``(channel, pixel)`` row: repeat each sample ``upsampling`` times, subtract
    ``baseline`` and multiply by ``scale``, apply a single-pole (``pole_zero``) decay
    correction, and smooth with two ``upsampling``-wide moving averages. When
    ``smoothing_fwhm > 0`` a delay-compensated Deriche (1992) IIR pass of that FWHM (in
    upsampled samples) is applied on top; ``0`` or ``None`` disables it.

    ``waveforms`` (n_channels, n_pix, n_samples); ``upsampling`` >= 1. ``pole_zero``,
    ``baseline`` and ``scale`` are each independently a scalar, a per-pixel ``(n_pix,)``
    array, or a full ``(n_channels, n_pix)`` array. Returns float32 (n_channels, n_pix,
    n_samples*upsampling).

    With ``smoothing_fwhm=0`` and ``upsampling>1`` the result is bit-identical to
    ``deconvolve`` (same underlying upsample+pole-zero kernel). Integer inputs (e.g.
    uint16 ADC samples) are cast to float32.
    """
    wf = _as_3d(waveforms)
    n_ch, n_pix, _ = wf.shape
    return _core.preprocess(
        wf,
        int(upsampling),
        _per_pixel(pole_zero, n_ch, n_pix),
        float(smoothing_fwhm or 0.0),
        _per_pixel(baseline, n_ch, n_pix),
        _per_pixel(scale, n_ch, n_pix),
    )


def preprocess_valid_range(upsampling, pole_zero, smoothing_fwhm, n_samples):
    """Trustworthy (non-edge) output-sample range ``(lo, hi)`` of ``preprocess``.

    Depends on ``pole_zero`` only through whether it is nonzero, so a scalar suffices even
    when ``preprocess`` is called with per-pixel values. ``n_samples`` is the RAW
    (pre-upsample) sample count; the returned bounds index the length
    ``n_samples*upsampling`` output. Uses the DVR convention ``n_samples - right`` for the
    upper bound, which differs from ``deconvolve_valid_range``. Returns ``(0, 0)`` if the
    edge margins exceed ``n_samples``.
    """
    return _core.preprocess_valid_range(
        int(upsampling), float(pole_zero), float(smoothing_fwhm or 0.0), int(n_samples)
    )


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
