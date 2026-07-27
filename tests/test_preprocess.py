# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Max-Planck-Institut für Kernphysik
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.

"""Tests for the batched ``preprocess`` / ``preprocess_valid_range`` bindings.

These need only numpy and the compiled extension (no ctapipe). The single-waveform C++
kernel is already validated bit-for-bit against the frozen libdvr oracle in
``tests/cpp/test_phepex.cpp``; these cover the Python binding layer: the batched row loop,
the scalar/per-pixel broadcasting of ``pole_zero``/``baseline``/``scale``, shape/dtype of
the result, and the valid-range formula.

``deconvolve`` is now a thin wrapper over ``preprocess`` (smoothing disabled), so
``preprocess(..., smoothing_fwhm=0)`` and ``deconvolve`` share the same kernel; the
cross-checks here assert that equivalence.
"""

import math

import numpy as np
import pytest

from phepex import (
    deconvolve,
    deconvolve_valid_range,
    preprocess,
    preprocess_valid_range,
)


def _wf(n_ch=2, n_pix=6, n_samples=20, seed=0):
    """Random float32 (n_ch, n_pix, n_samples) waveforms."""
    rng = np.random.default_rng(seed)
    return rng.normal(50.0, 10.0, (n_ch, n_pix, n_samples)).astype(np.float32)


def test_output_shape_and_dtype():
    """preprocess returns float32 (n_ch, n_pix, n_samples*upsampling)."""
    wf = _wf(n_ch=2, n_pix=6, n_samples=20)
    up = 4
    out = preprocess(wf, up, 0.9, baseline=3.0)
    assert out.shape == (2, 6, 20 * up)
    assert out.dtype == np.float32


def test_2d_input_promotes_leading_axis():
    """A 2D (n_pix, n_samples) input becomes (1, n_pix, ...), == explicit 3D."""
    wf2d = _wf(n_ch=1, n_pix=6, n_samples=20)[0]
    out2d = preprocess(wf2d, 4, 0.9, baseline=3.0)
    out3d = preprocess(wf2d[None], 4, 0.9, baseline=3.0)
    assert out2d.shape == (1, 6, 80)
    assert np.array_equal(out2d, out3d)


@pytest.mark.parametrize("up", [2, 4, 10])
@pytest.mark.parametrize("pz", [0.0, 0.9])
def test_no_smoothing_matches_deconvolve(up, pz):
    """smoothing_fwhm=0 is bit-identical to deconvolve (shared upsample kernel)."""
    wf = _wf(seed=1)
    baseline = 4.0
    got = preprocess(wf, up, pz, smoothing_fwhm=0.0, baseline=baseline)
    ref = deconvolve(wf, baseline, up, pz)
    assert np.array_equal(got, ref)


def test_none_smoothing_equals_zero():
    """smoothing_fwhm=None disables smoothing, same as 0.0."""
    wf = _wf(seed=2)
    a = preprocess(wf, 4, 0.9, smoothing_fwhm=None, baseline=3.0)
    b = preprocess(wf, 4, 0.9, smoothing_fwhm=0.0, baseline=3.0)
    assert np.array_equal(a, b)


def test_upsampling_one_no_smoothing_is_scale_minus_baseline():
    """upsampling=1, no smoothing, pole_zero=0 degenerates to scale*(wf - baseline)."""
    wf = _wf(seed=3)
    baseline, scale = 5.0, 0.5
    out = preprocess(wf, 1, 0.0, smoothing_fwhm=0.0, baseline=baseline, scale=scale)
    expected = (np.float32(scale) * (wf - np.float32(baseline))).astype(np.float32)
    assert np.array_equal(out, expected)


def test_upsampling_one_applies_pole_zero():
    """upsampling=1 applies the pole-zero correction (regression: it was skipped).

    Independent numpy reference of the closed form the kernel computes at up==1:
    out[0] = scale*(wf[0]-baseline); out[i] = scale*((wf[i]-baseline) -
    pole_zero*(wf[i-1]-baseline)). Compared with a tolerance rather than bit-for-bit: the
    C++ kernel contracts `cur - pole_zero*prev` into a fused multiply-add (single
    rounding), which numpy's separate multiply/subtract does not reproduce (~1 ULP).
    """
    wf = _wf(seed=3)
    baseline, scale, pole_zero = 5.0, 0.5, 0.9
    out = preprocess(wf, 1, pole_zero, smoothing_fwhm=0.0, baseline=baseline, scale=scale)

    d = (wf - np.float32(baseline)).astype(np.float32)
    expected = np.empty_like(d)
    expected[..., 0] = np.float32(scale) * d[..., 0]
    expected[..., 1:] = np.float32(scale) * (
        d[..., 1:] - np.float32(pole_zero) * d[..., :-1]
    )
    assert np.allclose(out, expected, rtol=1e-6, atol=1e-6)
    # The pole-zero term must actually change the result (guards against a silent skip):
    # the previous behaviour returned scale*(wf-baseline) and ignored pole_zero.
    plain = (np.float32(scale) * d).astype(np.float32)
    assert not np.allclose(out[..., 1:], plain[..., 1:])


def test_per_pixel_matches_per_row_scalar_calls():
    """Per-pixel (n_pix,) pole_zero/baseline/scale == scalar calls on each pixel."""
    wf = _wf(n_ch=1, n_pix=5, n_samples=20, seed=4)
    n_pix = wf.shape[1]
    rng = np.random.default_rng(5)
    pz = rng.uniform(0.0, 0.95, n_pix).astype(np.float32)
    bl = rng.uniform(-2.0, 8.0, n_pix).astype(np.float32)
    sc = rng.uniform(0.3, 1.5, n_pix).astype(np.float32)
    up = 4

    batched = preprocess(wf, up, pz, smoothing_fwhm=0.0, baseline=bl, scale=sc)
    for p in range(n_pix):
        row = preprocess(
            wf[:, p : p + 1], up, float(pz[p]), baseline=float(bl[p]), scale=float(sc[p])
        )
        assert np.array_equal(batched[:, p : p + 1], row)


@pytest.mark.parametrize(
    # Row counts (n_ch * n_pix) chosen against the default tile width
    # (PHEPEX_PREPROCESS_TILE_WIDTH == 24) to exercise the path split; the equivalence
    # assertion below holds for any width, so these still pass under a non-default build.
    "n_ch, n_pix",
    [
        (1, 3),  # fewer rows than a tile: pure scalar-remainder path (0 tiles)
        (1, 24),  # 24 rows = exactly one tile, no remainder (n_rows % 24 == 0 boundary)
        (2, 19),  # 38 rows = one full tile (24) + a 14-row remainder
    ],
)
@pytest.mark.parametrize(
    "up, fwhm",
    [
        (4, 3.0),  # upsampling + smoothing: tiled, deconvolve/upsample + Deriche IIR
        (4, 0.0),  # upsampling, no smoothing: tiled (running sums are latency-bound)
        (1, 3.0),  # no upsampling + smoothing: tiled (Deriche IIR)
        (1, 0.0),  # no upsampling, no smoothing: per-row scalar path (stencil)
    ],
)
def test_batched_matches_per_row_scalar_calls(n_ch, n_pix, up, fwhm):
    """The batched kernel must be bit-identical to scalar preprocess calls on each pixel
    across all four gate branches (tile when upsampling > 1 and/or smoothing, else
    scalar), including the remainder rows when n_ch*n_pix is not a multiple of the tile
    width."""
    wf = _wf(n_ch=n_ch, n_pix=n_pix, n_samples=20, seed=11)
    rng = np.random.default_rng(12)
    pz = rng.uniform(0.0, 0.95, n_pix).astype(np.float32)
    bl = rng.uniform(-2.0, 8.0, n_pix).astype(np.float32)
    sc = rng.uniform(0.3, 1.5, n_pix).astype(np.float32)

    batched = preprocess(wf, up, pz, smoothing_fwhm=fwhm, baseline=bl, scale=sc)
    for p in range(n_pix):
        row = preprocess(
            wf[:, p : p + 1],
            up,
            float(pz[p]),
            smoothing_fwhm=fwhm,
            baseline=float(bl[p]),
            scale=float(sc[p]),
        )
        assert np.array_equal(batched[:, p : p + 1], row)


def test_deconvolve_accepts_per_pixel_pole_zero():
    """deconvolve takes a per-pixel pole_zero (like preprocess).

    Verified two ways: the per-pixel call equals looping a scalar deconvolve over each
    pixel, and it equals preprocess with the same per-pixel pole_zero and no smoothing.
    """
    wf = _wf(n_ch=1, n_pix=5, n_samples=20, seed=8)
    n_pix = wf.shape[1]
    pz = np.random.default_rng(9).uniform(0.0, 0.95, n_pix).astype(np.float32)
    bl = np.random.default_rng(10).uniform(-2.0, 8.0, n_pix).astype(np.float32)
    up = 4

    batched = deconvolve(wf, bl, up, pz)
    for p in range(n_pix):
        row = deconvolve(wf[:, p : p + 1], float(bl[p]), up, float(pz[p]))
        assert np.array_equal(batched[:, p : p + 1], row)

    ref = preprocess(wf, up, pz, smoothing_fwhm=0.0, baseline=bl)
    assert np.array_equal(batched, ref)


def test_mixed_scalar_and_per_pixel():
    """Each arg is normalized independently: scalar pole_zero + per-pixel scale works."""
    wf = _wf(n_ch=1, n_pix=5, n_samples=20, seed=6)
    n_pix = wf.shape[1]
    scale = np.random.default_rng(7).uniform(0.5, 1.5, n_pix).astype(np.float32)
    up = 4
    mixed = preprocess(wf, up, 0.8, smoothing_fwhm=0.0, baseline=3.0, scale=scale)
    # same result as spelling out every argument per-pixel
    full = preprocess(
        wf,
        up,
        np.full(n_pix, 0.8, np.float32),
        smoothing_fwhm=0.0,
        baseline=np.full(n_pix, 3.0, np.float32),
        scale=scale,
    )
    assert np.array_equal(mixed, full)


@pytest.mark.parametrize("up", [1, 4])
@pytest.mark.parametrize("fwhm", [0.0, 3.0])
def test_uint16_input_matches_float32(up, fwhm):
    """uint16 ADC input takes the kernel's uint16 overload and agrees bit-for-bit.

    The wrapper hands uint16 through uncopied; uint16 widens to float32 exactly, so the
    two dtypes must give identical results on all four kernel branches (tiled/scalar x
    smoothing on/off).
    """
    adc = np.random.default_rng(11).integers(0, 4096, (2, 40, 24), dtype=np.uint16)
    kw = {
        "pole_zero": 0.75,
        "smoothing_fwhm": fwhm,
        "baseline": 200.0,
        "scale": 0.02,
    }
    out16 = preprocess(adc, up, **kw)
    out32 = preprocess(adc.astype(np.float32), up, **kw)
    assert out16.dtype == np.float32
    assert np.array_equal(out16, out32)


def test_signed_dtype_is_not_truncated_to_uint16():
    """Non-float32, non-uint16 dtypes convert to float32, never to the uint16 overload.

    A float64 array of negative samples would wrap to ~65531 if the uint16 overload were
    reachable by dtype conversion; it is registered noconvert, so it is not.
    """
    wf = np.full((1, 3, 8), -5.0, np.float64)
    out = preprocess(wf, 1, 0.0, baseline=0.0, scale=1.0)
    assert np.all(out == -5.0)


def test_broken_per_pixel_shape_raises():
    """A per-pixel array whose length is not n_pix raises (np.broadcast_to)."""
    wf = _wf(n_ch=1, n_pix=5, n_samples=20)
    with pytest.raises(ValueError):
        preprocess(wf, 4, np.ones(4, np.float32), baseline=0.0)


@pytest.mark.parametrize("bad", [0, -1])
def test_preprocess_rejects_upsampling_below_one(bad):
    """upsampling < 1 is rejected: the kernel divides by upsampling^2 and reads out of
    bounds for upsampling == 0, so it must not reach the C++ side."""
    wf = _wf(n_ch=1, n_pix=4, n_samples=20)
    with pytest.raises(ValueError):
        preprocess(wf, bad, 0.5)


@pytest.mark.parametrize("fwhm", [2.0, 4.0, 8.0])
def test_smoothing_preserves_dc_level(fwhm):
    """A constant input is preserved (unity DC gain) inside the valid range."""
    n_samples, up = 40, 4
    level = 7.0
    wf = np.full((1, 3, n_samples), level, np.float32)
    out = preprocess(wf, up, 0.0, smoothing_fwhm=fwhm, baseline=0.0, scale=1.0)
    lo, hi = preprocess_valid_range(up, 0.0, fwhm, n_samples)  # upsampled-output indices
    assert hi > lo
    interior = out[:, :, lo:hi]
    # The floor(fwhm) trim is a coarse margin; the Deriche IIR transient is not fully dead
    # right at the valid-range boundary, so a 1e-2 tolerance is used (as in the C++ unity
    # DC-gain test) rather than exact equality.
    assert np.abs(interior - level).max() < 1e-2


def _ref_valid_range(up, pole_zero, fwhm, n_samples):
    """Expected preprocess_valid_range: margins (upsampled samples) trimming the
    up*n_samples output."""
    up = max(up, 1)
    right = 2 * up - 2
    left = max(right, (3 * up - 2) if pole_zero != 0.0 else 0)
    if fwhm and fwhm > 0.0:
        f = int(math.floor(fwhm))
        right += f
        left += f
    n_up = up * n_samples
    if left >= n_up - right:
        return (0, 0)
    return (left, n_up - right)


@pytest.mark.parametrize("up", [1, 4, 10])
@pytest.mark.parametrize("pz", [0.0, 0.9])
@pytest.mark.parametrize("fwhm", [0.0, 4.0])
@pytest.mark.parametrize("n_samples", [40, 8, 3])
def test_valid_range_trims_upsampled_output(up, pz, fwhm, n_samples):
    """preprocess_valid_range indexes the up*n_samples output, incl. the (0,0) case."""
    got = preprocess_valid_range(up, pz, fwhm, n_samples)
    assert tuple(got) == _ref_valid_range(up, pz, fwhm, n_samples)


@pytest.mark.parametrize("up", [1, 4, 10])
@pytest.mark.parametrize("pz", [0.0, 0.9])
@pytest.mark.parametrize("n_samples", [40, 8])
def test_valid_range_no_smoothing_matches_deconvolve(up, pz, n_samples):
    """With no smoothing, a non-empty preprocess range equals deconvolve_valid_range."""
    pp = tuple(preprocess_valid_range(up, pz, 0.0, n_samples))
    if pp[0] < pp[1]:
        assert pp == tuple(deconvolve_valid_range(up, n_samples, pz))


if __name__ == "__main__":
    test_output_shape_and_dtype()
    test_2d_input_promotes_leading_axis()
    for up in (2, 4, 10):
        for pz in (0.0, 0.9):
            test_no_smoothing_matches_deconvolve(up, pz)
    test_none_smoothing_equals_zero()
    test_upsampling_one_no_smoothing_is_scale_minus_baseline()
    test_per_pixel_matches_per_row_scalar_calls()
    test_mixed_scalar_and_per_pixel()
    test_broken_per_pixel_shape_raises()
    for fwhm in (2.0, 4.0, 8.0):
        test_smoothing_preserves_dc_level(fwhm)
    for up in (1, 4, 10):
        for pz in (0.0, 0.9):
            for fwhm in (0.0, 4.0):
                for n_samples in (40, 8, 3):
                    test_valid_range_trims_upsampled_output(up, pz, fwhm, n_samples)
    for up in (1, 4, 10):
        for pz in (0.0, 0.9):
            for n_samples in (40, 8):
                test_valid_range_no_smoothing_matches_deconvolve(up, pz, n_samples)
    print("OK")
