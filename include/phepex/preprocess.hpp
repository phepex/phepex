// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Max-Planck-Institut für Kernphysik
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

#ifndef PHEPEX_PREPROCESS_HPP
#define PHEPEX_PREPROCESS_HPP

#include <array>
#include <cstdint>

#include "phepex/deconvolve.hpp"  // for SampleRange

namespace phepex {

/// Coefficients for the delay-compensated second-order IIR (Deriche 1992) Gaussian-like
/// smoothing applied by preprocess_waveform().
struct SmoothingCoefficients {
    double fwhm;  ///< target full width at half maximum of the impulse response
    std::array<double, 2> n;  ///< numerator coefficients for the forward filter step
    std::array<double, 2> m;  ///< numerator coefficients for the backward filter step
    std::array<double, 2> d;  ///< denominator coefficients
};

/// Compute the smoothing coefficients for a given impulse-response FWHM (in samples).
SmoothingCoefficients calculate_smoothing_coefficients(double smoothing_fwhm);

/// Upsampling + pole-zero deconvolution of a single waveform, with optional Gaussian
/// smoothing. Repeats each of the `n_samples` inputs `upsampling` times, subtracts
/// `offset` and applies `scale`, corrects a single-pole decay `pole_zero`, and smooths
/// with two `upsampling`-wide moving averages; when `smoothing != nullptr` an additional
/// Deriche IIR pass is applied. `upsampling == 1` degenerates to scale*(src - offset).
///
/// Writes `n_samples*upsampling` floats to `out` (caller-allocated). All arithmetic is
/// float32; `offset`/`scale`/`pole_zero` are float on purpose (bit-exactness).
///
/// @param src        input waveform of `n_samples` samples
/// @param smoothing  optional Deriche coefficients; nullptr disables smoothing
/// @param out        caller-allocated output of n_samples*upsampling floats
/// @param scratch    optional caller-provided workspace of at least n_samples*upsampling
///                   floats, used only when `smoothing != nullptr`; if null a buffer is
///                   allocated internally for the duration of the call
void preprocess_waveform(const std::uint16_t *src, int n_samples, int upsampling,
                         float pole_zero, const SmoothingCoefficients *smoothing,
                         float offset, float scale, float *out, float *scratch = nullptr);
void preprocess_waveform(const float *src, int n_samples, int upsampling, float pole_zero,
                         const SmoothingCoefficients *smoothing, float offset,
                         float scale, float *out, float *scratch = nullptr);

/// Trustworthy (non-edge) sample range of preprocess_waveform(): the boxcar smoothing
/// contaminates 2*(upsampling-1) samples at each end, a non-zero pole_zero adds
/// `upsampling` more invalid samples at the start, and smoothing widens both margins by
/// floor(fwhm). The upper bound follows the DVR convention `num_samples - right` (using
/// the RAW, pre-upsample `num_samples`) — this differs from deconvolve_valid_range().
/// Returns {0, 0} if the margins exceed `num_samples`.
SampleRange preprocess_valid_range(int upsampling, float pole_zero,
                                   const SmoothingCoefficients *smoothing,
                                   int num_samples);

}  // namespace phepex

#endif  // PHEPEX_PREPROCESS_HPP
