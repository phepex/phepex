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


def _as_3d(waveforms, keep_uint16=False):
    """Promote ``waveforms`` to a contiguous (n_channels, n_pix, n_samples) array.

    Missing leading axes are *prepended* (a 2D ``(n_pix, n_samples)`` array becomes
    ``(1, n_pix, n_samples)``), unlike ``np.atleast_3d`` which appends a trailing axis.

    The dtype becomes float32, except that ``keep_uint16=True`` passes a uint16 array
    through unconverted: ``_core.preprocess`` has a uint16 overload that reads raw ADC
    samples directly, so casting here would add a full copy of the input for nothing.
    """
    wf = np.asarray(waveforms)
    dtype = wf.dtype if (keep_uint16 and wf.dtype == np.uint16) else np.float32
    while wf.ndim < 3:
        wf = wf[np.newaxis, ...]
    return np.ascontiguousarray(wf, dtype=dtype)


def _per_pixel(x, n_ch, n_pix):
    """Broadcast a scalar / ``(n_pix,)`` / ``(n_ch, n_pix)`` value to contiguous float32.

    A scalar returns a length-1 array; the ``_core`` binding applies it to every row with
    row stride 0, so a scalar-argument call does not allocate a full ``n_ch*n_pix`` array.
    A ``(n_pix,)`` or ``(n_ch, n_pix)`` value is broadcast to a length ``n_ch*n_pix``
    array (row-major over ``(channel, pixel)``). ``np.broadcast_to`` raises ``ValueError``
    for any other shape.
    """
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 0:
        return np.ascontiguousarray(a.reshape(1), dtype=np.float32)
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
    """Pole-zero deconvolution + upsampling.

    Deconvolution is preprocessing without the optional smoothing pass, so this is
    `preprocess` with ``smoothing_fwhm=0`` (same arguments and broadcasting, with
    ``baselines`` as ``baseline``). Matches ctapipe's ``deconvolve`` in the trustworthy
    (non-edge) region; see `deconvolve_valid_range` for the edge margins.

    At ``upsampling == 1`` with ``pole_zero == 0`` there is no deconvolution, so every
    sample is valid (`deconvolve_valid_range` returns ``lo == 0``).
    """
    return preprocess(
        waveforms, upsampling, pole_zero, smoothing_fwhm=0.0, baseline=baselines
    )


def deconvolve_valid_range(upsampling, n_samples, pole_zero):
    """Trustworthy (non-edge) output-sample range ``[lo, hi)`` of ``deconvolve``.

    Deconvolution is preprocessing without smoothing, so this is
    ``preprocess_valid_range`` with ``smoothing_fwhm == 0``.
    """
    return preprocess_valid_range(upsampling, pole_zero, 0.0, n_samples)


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

    ``waveforms`` is (n_channels, n_pix, n_samples); leading axes are prepended if
    missing. ``upsampling`` >= 1 (``_core.preprocess`` raises ``ValueError`` otherwise).
    ``pole_zero``, ``baseline`` and ``scale`` are each independently a scalar, a per-pixel
    ``(n_pix,)`` array, or a full ``(n_channels, n_pix)`` array. Returns float32
    (n_channels, n_pix, n_samples*upsampling).

    uint16 input (raw ADC samples) reaches the kernel's uint16 overload uncopied; every
    other dtype is cast to float32. uint16 widens to float32 exactly, so both paths give
    bit-identical results.
    """
    wf = _as_3d(waveforms, keep_uint16=True)
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
    """Trustworthy (non-edge) output-sample range ``[lo, hi)`` of `preprocess`.

    Depends on ``pole_zero`` only through whether it is nonzero, so a scalar suffices even
    when `preprocess` is called with per-pixel values. ``n_samples`` is the RAW
    (pre-upsample) sample count; the returned bounds index the length
    ``n_samples*upsampling`` output. Returns ``(0, 0)`` if the edge margins leave no
    trustworthy sample.
    """
    return _core.preprocess_valid_range(
        int(upsampling), float(pole_zero), float(smoothing_fwhm or 0.0), int(n_samples)
    )


def pos_soft_clip(waveforms, scale, sample_lo=0, sample_hi=0):
    """Positive soft clip ``max(y / (1 + |y|), 0)``, ``y = waveforms / scale``.

    Applied over ``[sample_lo, sample_hi)``, 0 elsewhere; ``(0, 0)`` means the full trace.
    The soft clip already bounds the result to ``(-1, 1)``, so only negatives are clamped
    (to 0). Equivalent to ctapipe ``FlashCamExtractor.clip(waveforms / scale)``.
    """
    wf = _as_3d(waveforms)
    return _core.pos_soft_clip(wf, float(scale), int(sample_lo), int(sample_hi))


def neighbor_peak_indices(
    waveforms, neighbors, local_weight, broken_pixels, sample_lo=0, sample_hi=0
):
    """Per-pixel peak sample index of the neighbour-summed waveform.

    Equivalent to ctapipe's ``neighbor_average_maximum``: sums each pixel's own trace with
    weight ``local_weight`` and the traces of its non-broken neighbours, then takes the
    argmax over ``[sample_lo, sample_hi)`` (``(0, 0)`` means the full trace) as an
    absolute index into the trace. No normalisation by the neighbour count, which would
    not move the argmax anyway.

    ``neighbors`` is a scipy CSR matrix (``geometry.neighbor_matrix_sparse``); its
    ``indptr``/``indices`` are cast to int32. ``broken_pixels`` is
    ``(n_channels, n_pix)``. Returns int64 (n_channels, n_pix).
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

    Integrates over ``[peak_index - shift, peak_index - shift + width)``, clamped to the
    trace. ``sampling_rate_ghz`` must include any upsampling applied to ``waveforms``.
    Returns ``(charge, peak_time)`` float32 (n_channels, n_pix); ``peak_time`` in ns.
    """
    wf = np.ascontiguousarray(waveforms, dtype=np.float32)
    pk = np.ascontiguousarray(peak_index, dtype=np.int64)
    return _core.extract_around_peak(
        wf, pk, int(width), int(shift), float(sampling_rate_ghz)
    )


def adaptive_centroid(waveforms, peak_index, rel_descend_limit):
    """Leading-edge centroid in *sample* units; ctapipe's ``adaptive_centroid``.

    Walks left then right from ``peak_index`` while samples exceed
    ``rel_descend_limit * waveforms[peak_index]``, returning the amplitude-weighted index
    centroid. Falls back to ``peak_index`` where that window is empty. float32
    (n_channels, n_pix).
    """
    wf = np.ascontiguousarray(waveforms, dtype=np.float32)
    pk = np.ascontiguousarray(peak_index, dtype=np.int64)
    return _core.adaptive_centroid(wf, pk, float(rel_descend_limit))
