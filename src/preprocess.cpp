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
