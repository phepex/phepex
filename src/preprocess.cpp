// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Max-Planck-Institut für Kernphysik
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

#include "phepex/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "internal.hpp"

// Row-tile width for the batched preprocessing path (preprocess_waveforms). The CMake
// build defines this explicitly (cache variable PHEPEX_PREPROCESS_TILE_WIDTH); this guard
// only supplies a default for compilation outside CMake. See TILE_WIDTH for the tuning
// rationale.
#ifndef PHEPEX_PREPROCESS_TILE_WIDTH
#define PHEPEX_PREPROCESS_TILE_WIDTH 24
#endif

namespace phepex {
namespace {

/** @brief Apply Gaussian-like smoothing with coefficients from
 * calculate_smoothing_coefficients().
 *
 * A delay-compensated second-order IIR filter is used, see [1].
 *
 * [1] Deriche, R., 1992, Recursively implementing the Gaussian and its derivatives:
 * Proceedings of the 2nd International Conference on Image Processing, Singapore,
 * pp263—267.
 */
template <typename InputType, typename OutputType>
void smooth_waveform(const InputType *x, OutputType *y, int n,
                     const SmoothingCoefficients &c) {
    if (n == 0 || x == nullptr || y == nullptr)
        return;

    // Forward step: y[i] = n[0]*x[i] + n[1]*x[i-1] - d[0]*y[i-1] - d[1]*y[i-2]
    y[0] = c.n[0] * x[0];

    if (n == 1)
        return;

    y[1] = c.n[0] * x[1] + c.n[1] * x[0] - c.d[0] * y[0];
    for (int i = 2; i < n; i++)
        y[i] = c.n[0] * x[i] + c.n[1] * x[i - 1] - c.d[0] * y[i - 1] - c.d[1] * y[i - 2];

    // Backward step: y[i] = m[0]*x[i+1] + m[1]*x[i+2] - d[0]*y[i+1] - d[1]*y[i+2]
    double y_2 = 0;
    double y_1 = 0;
    double y_0 = c.m[0] * x[n - 1];
    y[n - 2] += y_0;
    for (int i = n - 3; i >= 0; i--) {
        y_2 = y_1;
        y_1 = y_0;
        y_0 = c.m[0] * x[i + 1] + c.m[1] * x[i + 2] - c.d[0] * y_1 - c.d[1] * y_2;
        y[i] += y_0;
    }
}

// Unity-upsampling (upsampling == 1) pole-zero deconvolution. At upsampling == 1 the two
// moving-average boxcars in detail::upsample_waveform are width-1 (identity), so its
// runsum machinery reduces to the plain pole-zero difference computed directly here; the
// closed form also skips the two accumulating running sums, yielding
// scale*((src[i]-offset) - pole_zero*(src[i-1]-offset)) without their intermediate
// accumulation. dst[0] has no predecessor and so carries no pole-zero term (matching the
// upsampling>1 kernel, whose leading samples are trimmed by preprocess_valid_range).
template <typename InputType>
void deconvolve_unit_upsampling(const InputType *src, int n_samples, float pole_zero,
                                float offset, float scale, float *dst) {
    if (n_samples <= 0)
        return;
    // pole_zero == 0 is a pure map dst[i] = scale*(src[i]-offset) with no loop-carried
    // dependency, so the compiler auto-vectorises it. The general loop below cannot prove
    // pole_zero == 0 at compile time, so it keeps the `prev` recurrence and stays scalar.
    // Bit-identical to the general path here, since cur - 0.0f*prev == cur, and free for
    // pole_zero != 0 (one branch, no measurable regression).
    if (pole_zero == 0.0f) {
        for (int i = 0; i < n_samples; i++)
            dst[i] = scale * (static_cast<float>(src[i]) - offset);
        return;
    }
    float prev = static_cast<float>(src[0]) - offset;
    dst[0] = scale * prev;
    for (int i = 1; i < n_samples; i++) {
        const float cur = static_cast<float>(src[i]) - offset;
        dst[i] = scale * (cur - pole_zero * prev);
        prev = cur;
    }
}

// Row-tile width for the batched, per-lane preprocessing path (preprocess_waveforms): the
// number of waveforms mapped onto SIMD lanes per tile. A compile-time width lets the
// compiler vectorise the inner `k` loops. Configurable via PHEPEX_PREPROCESS_TILE_WIDTH
// (CMake).
//
// Tuning: a wider tile exposes more independent rows to the reorder window (helping the
// latency-bound recurrences) at the cost of a larger L1 tile footprint (~3 * width * n_up
// floats live at once, so keep width * n_up well inside L1). The optimum is not simply
// the target's float SIMD width: 24 (the default) measured fastest on both Apple Silicon
// (NEON, 4-wide) and Zen 4 (AVX-512, 16-wide), so the reorder-window occupancy dominates
// lane count. Tune empirically with the micro-benchmark. The value is not
// correctness-bearing: any width >= 1 produces a bit-identical result.
constexpr int TILE_WIDTH = PHEPEX_PREPROCESS_TILE_WIDTH;
static_assert(TILE_WIDTH >= 1, "PHEPEX_PREPROCESS_TILE_WIDTH must be >= 1");

// Floats of scratch the batched (tiled) path needs: in_tile (n_samples) + up_tile (n_up)
// + out_tile (n_up, smoothing only), each times the tile width. 0 when the scalar path is
// used (upsampling == 1 and no smoothing). Single source of the size formula, shared by
// preprocess_waveforms (partitioning its buffer) and preprocess_waveforms_scratch_size.
std::size_t tiled_scratch_size(int n_samples, int upsampling, bool smoothing) {
    if (upsampling == 1 && !smoothing)
        return 0;
    const std::size_t n_up = static_cast<std::size_t>(upsampling) * n_samples;
    return static_cast<std::size_t>(TILE_WIDTH) *
           (static_cast<std::size_t>(n_samples) + n_up + (smoothing ? n_up : 0));
}

// Batched counterparts of the three single-waveform kernels above, each operating on K
// rows held sample-major in a tile (index [sample*K + lane]) so the inner `for k` loop
// over lanes carries no dependency and vectorises. The per-lane arithmetic is identical
// to the scalar kernels, so the tiled output is bit-for-bit equal to
// preprocess_waveform() per row.

// Batched deconvolve_unit_upsampling (upsampling == 1) over K lanes. The scalar kernel's
// pole_zero == 0 fast path is not special-cased here: cur - 0*prev == cur bit-for-bit, so
// the general recurrence reproduces it exactly while allowing pole_zero to differ per
// lane.
template <int K>
void deconvolve_batched(const float *xin, int n, const float *pole_zero,
                        const float *offset, const float *scale, float *tile) {
    if (n <=
        0)  // guard xin[k] read below, matching the scalar kernel's n_samples<=0 exit
        return;
    float prev[K];
    for (int k = 0; k < K; k++) {
        prev[k] = xin[k] - offset[k];
        tile[k] = scale[k] * prev[k];
    }
    for (int i = 1; i < n; i++)
        for (int k = 0; k < K; k++) {
            const float cur = xin[i * K + k] - offset[k];
            tile[i * K + k] = scale[k] * (cur - pole_zero[k] * prev[k]);
            prev[k] = cur;
        }
}

// Batched detail::upsample_waveform (upsampling > 1) over K lanes. Mirrors the scalar
// kernel's two interleaved running sums (sum1, sum2); the leading `out1` pointer becomes
// the index j1 and the lagging `out2` pointer j2, shared across lanes since they advance
// deterministically.
template <int K>
void upsample_batched(int n, int up, const float *xin, const float *pole_zero,
                      const float *offset, const float *scale, float *tile) {
    if (n <= 0)  // guard xin[k] / tile[nn-1] access below, matching smooth_batched's exit
        return;
    float mult[K], v1[K], pzc1[K], pzc2[K], sum1[K], sum2[K];
    for (int k = 0; k < K; k++) {
        mult[k] = scale[k] / static_cast<float>(up * up);
        const float v2 = (xin[k] - offset[k]) * mult[k];
        v1[k] = v2;
        pzc1[k] = v2;
        sum1[k] = v2 * up;
        sum2[k] = sum1[k] * up;
    }
    int j1 = 0, j2 = 0;
    for (int i1 = 0; i1 < up; i1++, j1++)
        for (int k = 0; k < K; k++)
            tile[j1 * K + k] = sum1[k];
    for (int i = 1; i < n; i++) {
        for (int k = 0; k < K; k++) {
            const float v2 = (xin[i * K + k] - offset[k]) * mult[k];
            pzc2[k] = v2 - v1[k];
            v1[k] = v2;
        }
        for (int i1 = 0; i1 < up; i1++, j1++, j2++) {
            for (int k = 0; k < K; k++) {
                sum1[k] += pzc2[k] - pzc1[k] * pole_zero[k];
                tile[j1 * K + k] = sum1[k];
            }
            for (int k = 0; k < K; k++) {
                const float tmp = tile[j2 * K + k];
                tile[j2 * K + k] = sum2[k];
                sum2[k] += sum1[k] - tmp;
            }
        }
        for (int k = 0; k < K; k++)
            pzc1[k] = pzc2[k];
    }
    const int nn = n * up;
    float last[K];
    for (int k = 0; k < K; k++)
        last[k] = tile[(nn - 1) * K + k];
    for (; j2 < nn; j2++)
        for (int k = 0; k < K; k++) {
            const float tmp = tile[j2 * K + k];
            tile[j2 * K + k] = sum2[k];
            sum2[k] += last[k] - tmp;
        }
}

// Batched smooth_waveform over K lanes. The forward/backward recurrences accumulate in
// double per lane, matching the scalar kernel's precision (coefficients are double, y/x
// are float); `y[i] += nv` with double nv promotes as in the scalar `y[i] += y0`.
template <int K>
void smooth_batched(const float *x, float *y, int n, const SmoothingCoefficients &c) {
    if (n <= 0)
        return;
    const double n0 = c.n[0], n1 = c.n[1], m0 = c.m[0], m1 = c.m[1], d0 = c.d[0],
                 d1 = c.d[1];
    for (int k = 0; k < K; k++)
        y[k] = n0 * x[k];
    if (n == 1)
        return;
    for (int k = 0; k < K; k++)
        y[K + k] = n0 * x[K + k] + n1 * x[k] - d0 * y[k];
    for (int i = 2; i < n; i++)
        for (int k = 0; k < K; k++)
            y[i * K + k] = n0 * x[i * K + k] + n1 * x[(i - 1) * K + k] -
                           d0 * y[(i - 1) * K + k] - d1 * y[(i - 2) * K + k];

    double pr0[K], pr1[K];
    for (int k = 0; k < K; k++) {
        pr0[k] = m0 * x[(n - 1) * K + k];
        pr1[k] = 0.0;
        y[(n - 2) * K + k] += pr0[k];
    }
    for (int i = n - 3; i >= 0; i--)
        for (int k = 0; k < K; k++) {
            const double o0 = pr0[k], o1 = pr1[k];
            const double nv =
                m0 * x[(i + 1) * K + k] + m1 * x[(i + 2) * K + k] - d0 * o0 - d1 * o1;
            y[i * K + k] += nv;
            pr1[k] = o0;
            pr0[k] = nv;
        }
}

// Shared body of the two preprocess_waveform overloads (uint16 and float input).
template <typename InputType>
void preprocess_impl(const InputType *src, int n_samples, int upsampling, float pole_zero,
                     const SmoothingCoefficients *smoothing, float offset, float scale,
                     float *out, float *scratch) {
    const std::size_t dst_samples = static_cast<std::size_t>(upsampling) * n_samples;

    if (smoothing) {
        std::vector<float> owned;
        float *tmp = scratch;
        if (tmp == nullptr) {
            owned.resize(dst_samples);
            tmp = owned.data();
        }
        if (upsampling == 1)
            deconvolve_unit_upsampling(src, n_samples, pole_zero, offset, scale, tmp);
        else
            detail::upsample_waveform<InputType, float>(n_samples, upsampling, src,
                                                        offset, pole_zero, tmp, scale);

        smooth_waveform(tmp, out, static_cast<int>(dst_samples), *smoothing);
    } else {
        if (upsampling == 1)
            deconvolve_unit_upsampling(src, n_samples, pole_zero, offset, scale, out);
        else
            detail::upsample_waveform<InputType, float>(n_samples, upsampling, src,
                                                        offset, pole_zero, out, scale);
    }
}

}  // namespace

SmoothingCoefficients calculate_smoothing_coefficients(double smoothing_fwhm) {
    // The coefficients from Deriche 1992, g1=0.9629, g2=1.942, w=0.8448, b=1.26, create
    // an undesirable double peak and undershoot, which is not the case with these
    // coefficients:
    const double g1 = 1.06003468;
    const double g2 = 2.85668332;
    const double w = 0.54103265;
    const double b = 1.44605019;

    const double sigma = smoothing_fwhm / 2.1;

    // Cf. Deriche 1992 for these equations
    const double d0 = -2.0 * std::cos(w / sigma) * std::exp(-b / sigma);
    const double d1 = std::exp(-2.0 * b / sigma);
    const double n0 = g1;
    const double n1 =
        (-g1 * std::cos(w / sigma) + g2 * std::sin(w / sigma)) * std::exp(-b / sigma);
    const double m0 = n1 - d0 * n0;
    const double m1 = -d1 * n0;

    // Calculate normalization factor for the numerator coefficients
    const double c = (1.0 + d0 + d1) / (n0 + n1 + m0 + m1);

    return {smoothing_fwhm, {n0 * c, n1 * c}, {m0 * c, m1 * c}, {d0, d1}};
}

void preprocess_waveform(const std::uint16_t *src, int n_samples, int upsampling,
                         float pole_zero, const SmoothingCoefficients *smoothing,
                         float offset, float scale, float *out, float *scratch) {
    preprocess_impl(src, n_samples, upsampling, pole_zero, smoothing, offset, scale, out,
                    scratch);
}

void preprocess_waveform(const float *src, int n_samples, int upsampling, float pole_zero,
                         const SmoothingCoefficients *smoothing, float offset,
                         float scale, float *out, float *scratch) {
    preprocess_impl(src, n_samples, upsampling, pole_zero, smoothing, offset, scale, out,
                    scratch);
}

void preprocess_waveforms(const float *src, int n_rows, int n_samples, int upsampling,
                          const float *pole_zero, std::ptrdiff_t pole_zero_stride,
                          const SmoothingCoefficients *smoothing, const float *offset,
                          std::ptrdiff_t offset_stride, const float *scale,
                          std::ptrdiff_t scale_stride, float *out, float *scratch) {
    const int n_up = upsampling * n_samples;
    const auto row = [&](int r, float *scratch) {
        preprocess_waveform(src + static_cast<std::size_t>(r) * n_samples, n_samples,
                            upsampling, pole_zero[r * pole_zero_stride], smoothing,
                            offset[r * offset_stride], scale[r * scale_stride],
                            out + static_cast<std::size_t>(r) * n_up, scratch);
    };

    // Scalar per-row path when nothing latency-bound is present to batch: at upsampling
    // == 1 with no smoothing the kernel is deconvolve_unit_upsampling, a stencil that
    // already vectorises across samples (no loop-carried output dependency), so tiling
    // would only add transpose overhead. Every other case carries a latency-bound
    // recurrence -- the upsampling running sums (upsampling > 1) and/or the Deriche IIR
    // (smoothing) -- whose throughput the tiling raises by more than the transpose costs.
    if (upsampling == 1 && smoothing == nullptr) {
        for (int r = 0; r < n_rows; r++)
            row(r, nullptr);
        return;
    }

    constexpr int K = TILE_WIDTH;
    // Sample-major tiles carved from one buffer: in_tile (transposed input) | up_tile
    // (upsampled/deconvolved intermediate) | out_tile (smoothed output, only when
    // smoothing runs; otherwise up_tile is transposed out directly). The buffer is
    // caller-provided via `scratch` (sized by tiled_scratch_size) or allocated once here;
    // either way it is reused across all tiles, so the intermediate stays resident
    // instead of a per-row round-trip.
    const std::size_t in_sz = static_cast<std::size_t>(n_samples) * K;
    const std::size_t up_sz = static_cast<std::size_t>(n_up) * K;
    std::vector<float> owned;
    float *buf = scratch;
    if (buf == nullptr) {
        owned.resize(tiled_scratch_size(n_samples, upsampling, smoothing != nullptr));
        buf = owned.data();
    }
    float *in_tile = buf;
    float *up_tile = buf + in_sz;
    float *out_tile = (smoothing != nullptr) ? up_tile + up_sz : nullptr;
    float pz[K], off[K], scl[K];

    int p = 0;
    for (; p + K <= n_rows; p += K) {
        for (int k = 0; k < K; k++) {
            const int r = p + k;
            pz[k] = pole_zero[r * pole_zero_stride];
            off[k] = offset[r * offset_stride];
            scl[k] = scale[r * scale_stride];
            const float *s = src + static_cast<std::size_t>(r) * n_samples;
            for (int i = 0; i < n_samples; i++)
                in_tile[i * K + k] = s[i];
        }
        if (upsampling == 1)
            deconvolve_batched<K>(in_tile, n_samples, pz, off, scl, up_tile);
        else
            upsample_batched<K>(n_samples, upsampling, in_tile, pz, off, scl, up_tile);
        const float *result = up_tile;
        if (smoothing != nullptr) {
            smooth_batched<K>(up_tile, out_tile, n_up, *smoothing);
            result = out_tile;
        }
        for (int k = 0; k < K; k++) {
            float *d = out + static_cast<std::size_t>(p + k) * n_up;
            for (int i = 0; i < n_up; i++)
                d[i] = result[i * K + k];
        }
    }
    // Remainder rows (n_rows % K): scalar path, bit-identical to the tiled body. up_tile
    // doubles as smoothing scratch (n_up floats); it is unused by the no-smoothing
    // kernels.
    for (; p < n_rows; p++)
        row(p, up_tile);
}

std::size_t preprocess_waveforms_scratch_size(int n_samples, int upsampling,
                                              const SmoothingCoefficients *smoothing) {
    return tiled_scratch_size(n_samples, upsampling, smoothing != nullptr);
}

SampleRange preprocess_valid_range(int upsampling, float pole_zero,
                                   const SmoothingCoefficients *smoothing,
                                   int num_samples) {
    if (upsampling < 1)
        upsampling = 1;

    int right = 2 * upsampling - 2;
    int left = std::max(right, (pole_zero != 0.0f) ? (3 * upsampling - 2) : 0);

    if (smoothing) {
        const int fwhm = static_cast<int>(std::floor(smoothing->fwhm));
        right += fwhm;
        left += fwhm;
    }

    // `left`/`right` are margins in UPSAMPLED samples, so they trim the
    // upsampling*num_samples output, not the raw num_samples input.
    const int n_up = upsampling * num_samples;
    if (left >= n_up - right)  // margins meet or cross => nothing trustworthy
        return {0, 0};

    return {left, n_up - right};
}

}  // namespace phepex
