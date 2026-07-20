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
/// waveform over [peak-shift, peak-shift+width) (clamped to the trace) and computes the
/// amplitude-weighted centroid over strictly-positive samples, in ns (divided by
/// sampling_rate_ghz). float64 accumulation; float32 outputs.
///
/// @param waveforms   input (n_ch, n_pix, n_up) float32
/// @param peak_index  (n_ch, n_pix) int64 peak sample index per pixel
/// @param charge      caller-allocated n_ch*n_pix float32 (integrated charge)
/// @param peak_time   caller-allocated n_ch*n_pix float32 (peak time, ns)
void extract_around_peak(const float *waveforms, int n_ch, int n_pix, int n_up,
                         const std::int64_t *peak_index, int width, int shift,
                         double sampling_rate_ghz, float *charge, float *peak_time);

/// Leading-edge weighted centroid (in sample units), per (channel, pixel): walks left
/// then right from peak while samples exceed rel_descend_limit*peak_amplitude,
/// accumulating an amplitude-weighted index centroid. float64 accumulation; float32
/// output.
///
/// @param waveforms   input (n_ch, n_pix, n_up) float32
/// @param peak_index  (n_ch, n_pix) int64 peak sample index per pixel
/// @param centroids   caller-allocated n_ch*n_pix float32 (centroid, sample units)
void adaptive_centroid(const float *waveforms, int n_ch, int n_pix, int n_up,
                       const std::int64_t *peak_index, double rel_descend_limit,
                       float *centroids);

}  // namespace phepex

#endif  // PHEPEX_EXTRACT_HPP
