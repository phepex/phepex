// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Max-Planck-Institut für Kernphysik
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

#ifndef PHEPEX_DECONVOLVE_HPP
#define PHEPEX_DECONVOLVE_HPP

namespace phepex {

/// Pole-zero deconvolution + upsampling of PMT/SiPM waveforms.
///
/// Applies, independently to every (channel, pixel) waveform: upsampling by
/// `upsampling`, baseline subtraction and scaling, a single-pole-decay (pole-zero)
/// correction, and two `upsampling`-wide moving-average smoothings (a forward+backward
/// boxcar pass, so the smoothing introduces no net time shift).
///
/// `pole_zero`, `baseline` and `scale` are per-(channel, pixel) arrays of `n_ch*n_pix`
/// floats, indexed row-major like `waveforms`. Each may be null, in which case a constant
/// default is used for every pixel: pole_zero = 0 (no pole-zero correction), baseline =
/// 0, scale = 1.
///
/// The smoothing produces 2*(upsampling-1) invalid samples at each end of every output
/// waveform; where pole_zero != 0, the first 3*upsampling-2 samples are invalid
/// (see deconvolve_valid_range() below).
///
/// @param waveforms  input, row-major (n_ch, n_pix, n_samples), float32
/// @param pole_zero  per-(channel, pixel) pole-zero factors, or null for 0 (no
/// correction)
/// @param baseline   per-(channel, pixel) baselines subtracted before scaling, or null
/// for 0
/// @param scale      per-(channel, pixel) scale factors, or null for 1
/// @param out        caller-allocated output of n_ch*n_pix*n_samples*upsampling floats,
///                   row-major (n_ch, n_pix, n_samples*upsampling)
void deconvolve_upsample(const float *waveforms, int n_ch, int n_pix, int n_samples,
                         int upsampling, const float *pole_zero, const float *baseline,
                         const float *scale, float *out);

/// Half-open sample range [lo, hi).
struct SampleRange {
    int lo;  ///< lower limit (inclusive)
    int hi;  ///< upper limit (exclusive)
};

/// Trustworthy (non-edge) output-sample range of deconvolve_upsample(): the
/// upsample+filtfilt boxcar contaminates 2*(upsampling-1) samples at each end, and a
/// non-zero pole_zero adds `upsampling` more invalid samples at the start. Returns the
/// range into the length upsampling*n_samples deconvolution output.
SampleRange deconvolve_valid_range(int upsampling, int n_samples, double pole_zero);

}  // namespace phepex

#endif  // PHEPEX_DECONVOLVE_HPP
