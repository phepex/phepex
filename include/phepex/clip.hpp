// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Max-Planck-Institut für Kernphysik
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

#ifndef PHEPEX_CLIP_HPP
#define PHEPEX_CLIP_HPP

namespace phepex {

/// Positive soft clip: max(y/(1+|y|), 0) with y = x/scale, applied over samples
/// [sample_lo, sample_hi) of each (channel, pixel) waveform; samples outside that range
/// are set to 0. The soft clip already bounds the result to (-1, 1), so only negatives
/// are clamped (to 0).
///
/// @param waveforms  input (n_ch, n_pix, n_up) float32
/// @param scale      divisor applied before the clip; 0 yields +-inf/NaN, unchecked
/// @param sample_lo, sample_hi  half-open sample window; (0, 0) means the full trace.
///                   Bounds are used unchecked, so the caller must keep them within
///                   [0, n_up].
/// @param out        caller-allocated n_ch*n_pix*n_up floats (fully written; 0 outside
///                   the window)
void pos_soft_clip(const float *waveforms, int n_ch, int n_pix, int n_up, float scale,
                   int sample_lo, int sample_hi, float *out);

}  // namespace phepex

#endif  // PHEPEX_CLIP_HPP
