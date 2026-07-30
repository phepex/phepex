// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Max-Planck-Institut für Kernphysik
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

#ifndef PHEPEX_EXTRACT_HPP
#define PHEPEX_EXTRACT_HPP

#include <cstdint>

namespace phepex {

/// Window integration + amplitude-weighted peak time, per (channel, pixel). Sums the
/// waveform over [peak-shift, peak-shift+width) (clamped to [0, n_up)) and computes the
/// amplitude-weighted index centroid over the strictly-positive samples of that window,
/// converted to ns by dividing by sampling_rate_ghz. Falls back to `peak_index` itself
/// when the window holds no positive sample. float64 accumulation; float32 outputs.
///
/// @param waveforms   input (n_ch, n_pix, n_up) float32
/// @param peak_index  (n_ch, n_pix) int64 peak sample index per pixel
/// @param width       window length in samples
/// @param shift       samples the window starts before the peak
/// @param sampling_rate_ghz  sample rate of `waveforms`, i.e. including any upsampling
/// @param charge      caller-allocated n_ch*n_pix float32 (integrated charge)
/// @param peak_time   caller-allocated n_ch*n_pix float32 (peak time, ns)
void extract_around_peak(const float *waveforms, int n_ch, int n_pix, int n_up,
                         const std::int64_t *peak_index, int width, int shift,
                         double sampling_rate_ghz, float *charge, float *peak_time);

/// Leading-edge weighted centroid (in sample units), per (channel, pixel): walks left
/// then right from the peak while samples exceed rel_descend_limit*waveforms[peak],
/// accumulating an amplitude-weighted index centroid. The threshold is fixed at the
/// initial peak amplitude and not re-derived if a larger sample is met. Falls back to
/// `peak_index` itself when it lies outside [0, n_up), the peak amplitude is negative, or
/// the accumulated weight is 0. float64 accumulation; float32 output.
///
/// @param waveforms   input (n_ch, n_pix, n_up) float32
/// @param peak_index  (n_ch, n_pix) int64 peak sample index per pixel
/// @param rel_descend_limit  threshold as a fraction of the peak amplitude
/// @param centroids   caller-allocated n_ch*n_pix float32 (centroid, sample units)
void adaptive_centroid(const float *waveforms, int n_ch, int n_pix, int n_up,
                       const std::int64_t *peak_index, double rel_descend_limit,
                       float *centroids);

}  // namespace phepex

#endif  // PHEPEX_EXTRACT_HPP
