// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Max-Planck-Institut für Kernphysik
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

#ifndef PHEPEX_PREPROCESS_HPP
#define PHEPEX_PREPROCESS_HPP

#include <array>
#include <cstddef>
#include <cstdint>

namespace phepex {

/// Half-open sample range [lo, hi).
struct SampleRange {
    int lo;  ///< lower limit (inclusive)
    int hi;  ///< upper limit (exclusive)
};

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
/// Deriche IIR pass is applied. At `upsampling == 1` the two moving averages are width-1
/// (identity), so the result is the pole-zero-corrected signal
/// scale*((src[i]-offset) - pole_zero*(src[i-1]-offset)) with out[0] =
/// scale*(src[0]-offset); it degenerates to scale*(src - offset) only when `pole_zero ==
/// 0`.
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

/// Apply preprocess_waveform() to each of `n_rows` consecutive waveforms of `n_samples`
/// samples, writing `n_rows * n_samples * upsampling` floats to `out`. `pole_zero`,
/// `offset` and `scale` are read per row as `array[row * stride]`; a stride of 0
/// broadcasts a single value to every row, a stride of 1 selects a distinct per-row
/// value.
///
/// Rows are processed in fixed-width tiles, one row per SIMD lane, whenever the per-row
/// work contains a loop-carried, latency-bound recurrence: the upsampling running sums
/// (when `upsampling > 1`) and/or the Deriche IIR (when `smoothing != nullptr`; see
/// calculate_smoothing_coefficients()). Tiling fills that recurrence with independent
/// rows, raising throughput by more than the tile transpose costs. The result is
/// bit-identical to calling preprocess_waveform() once per row: intermediate arithmetic
/// order and precision (double accumulators in the smoothing pass) are unchanged.
///
/// The single case with no such recurrence -- `upsampling == 1` and `smoothing ==
/// nullptr`, where the kernel is a pole-zero stencil that already vectorises across
/// samples -- uses the per-row scalar path directly, as tiling would only add transpose
/// overhead.
///
/// Float input only. `out` is caller-allocated. `scratch` is an optional workspace for
/// the tile buffers of the batched path: pass a buffer of at least
/// preprocess_waveforms_scratch_size(n_samples, upsampling, smoothing) floats to avoid
/// the per-call allocation (e.g. when preprocessing many events in a loop); if null a
/// buffer is allocated internally for the duration of the call. Used only on the tiled
/// path; ignored when `upsampling == 1` and `smoothing == nullptr` (where the size query
/// returns 0). The caller-provided scratch must not be shared between concurrent calls;
/// the null path is reentrant.
void preprocess_waveforms(const float *src, int n_rows, int n_samples, int upsampling,
                          const float *pole_zero, std::ptrdiff_t pole_zero_stride,
                          const SmoothingCoefficients *smoothing, const float *offset,
                          std::ptrdiff_t offset_stride, const float *scale,
                          std::ptrdiff_t scale_stride, float *out,
                          float *scratch = nullptr);

/// Number of floats a preprocess_waveforms() call needs in its `scratch` buffer for the
/// given parameters. Returns 0 when `upsampling == 1` and `smoothing == nullptr` (the
/// per-row scalar path uses no tiles). Callers must obtain the size from this function
/// rather than compute it, as it depends on the library's build-time tile width.
/// `smoothing` is read only for null-ness (the size does not depend on the coefficient
/// values), matching the pointer passed to preprocess_waveforms().
std::size_t preprocess_waveforms_scratch_size(int n_samples, int upsampling,
                                              const SmoothingCoefficients *smoothing);

/// Trustworthy (non-edge) sample range of preprocess_waveform(), as indices into the
/// `upsampling*num_samples` output. The boxcar smoothing contaminates 2*(upsampling-1)
/// samples at each end, a non-zero pole_zero adds `upsampling` more invalid samples at
/// the start, and an optional Deriche pass widens both margins by floor(fwhm). The
/// margins are in upsampled samples, so the range is
/// {left, upsampling*num_samples - right}. Returns {0, 0} when the margins leave no
/// trustworthy samples.
SampleRange preprocess_valid_range(int upsampling, float pole_zero,
                                   const SmoothingCoefficients *smoothing,
                                   int num_samples);

}  // namespace phepex

#endif  // PHEPEX_PREPROCESS_HPP
