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
/// Requires `smoothing_fwhm > 0` (it appears in a denominator, as sigma = fwhm/2.1);
/// unchecked.
SmoothingCoefficients calculate_smoothing_coefficients(double smoothing_fwhm);

/// Upsampling + pole-zero deconvolution of a single waveform, with optional Gaussian
/// smoothing. Repeats each of the `n_samples` inputs `upsampling` times, subtracts
/// `offset` and applies `scale`, corrects a single-pole decay `pole_zero`, and smooths
/// with two `upsampling`-wide moving averages; when `smoothing != nullptr` an additional
/// Deriche IIR pass is applied. At `upsampling == 1` both moving averages are width-1
/// (identity), leaving out[i] = scale*((src[i]-offset) - pole_zero*(src[i-1]-offset)) and
/// out[0] = scale*(src[0]-offset); that degenerates to scale*(src - offset) only when
/// `pole_zero == 0`.
///
/// Requires `upsampling >= 1`; smaller values are unchecked and read out of bounds. All
/// arithmetic is float32; `offset`/`scale`/`pole_zero` are float on purpose
/// (bit-exactness against ctapipe).
///
/// @param src        input waveform of `n_samples` samples
/// @param smoothing  optional Deriche coefficients (see
///                   calculate_smoothing_coefficients()); nullptr disables smoothing
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
/// samples, writing `n_rows * n_samples * upsampling` floats to caller-allocated `out`.
/// Rows are contiguous on both sides (row `r` reads `src[r * n_samples]` and writes
/// `out[r * n_samples * upsampling]`); strided views must be copied by the caller.
/// `pole_zero`, `offset` and `scale` are read per row as `array[row * stride]`, so a
/// stride of 0 broadcasts one value to every row and a stride of 1 selects a distinct
/// per-row value.
///
/// Rows are processed in fixed-width tiles, one row per SIMD lane, whenever the per-row
/// work carries a latency-bound recurrence -- the upsampling running sums
/// (`upsampling > 1`) and/or the Deriche IIR (`smoothing != nullptr`) -- so that the
/// recurrence is filled with independent rows; the gain exceeds the cost of the tile
/// transpose. uint16 input is widened inside that transpose and so costs no more than
/// float input. The one case without such a recurrence, `upsampling == 1` with
/// `smoothing == nullptr`, is a pole-zero stencil that already vectorises across samples
/// and takes the per-row scalar path.
///
/// Output is bit-identical to calling preprocess_waveform() once per row: arithmetic
/// order and intermediate precision (float32 throughout, including the smoothing pass)
/// are unchanged. Requires `upsampling >= 1`, unchecked as in preprocess_waveform().
///
/// @param scratch  optional workspace for the tile buffers, at least
///                 preprocess_waveforms_scratch_size(n_samples, upsampling, smoothing)
///                 floats; avoids the per-call allocation when preprocessing many events
///                 in a loop. If null a buffer is allocated internally for the duration
///                 of the call. Unused on the scalar path. A caller-provided buffer must
///                 not be shared between concurrent calls; the null path is reentrant.
void preprocess_waveforms(const std::uint16_t *src, int n_rows, int n_samples,
                          int upsampling, const float *pole_zero,
                          std::ptrdiff_t pole_zero_stride,
                          const SmoothingCoefficients *smoothing, const float *offset,
                          std::ptrdiff_t offset_stride, const float *scale,
                          std::ptrdiff_t scale_stride, float *out,
                          float *scratch = nullptr);
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

/// Trustworthy (non-edge) sample range of preprocess_waveform(), as indices into its
/// length `upsampling*num_samples` output. The two boxcars contaminate 2*(upsampling-1)
/// samples at each end, a non-zero `pole_zero` adds `upsampling` more at the start, and a
/// Deriche pass widens both margins by floor(fwhm). Returns {0, 0} when the margins meet
/// or cross, i.e. when no sample is trustworthy.
SampleRange preprocess_valid_range(int upsampling, float pole_zero,
                                   const SmoothingCoefficients *smoothing,
                                   int num_samples);

}  // namespace phepex

#endif  // PHEPEX_PREPROCESS_HPP
