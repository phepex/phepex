// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Max-Planck-Institut für Kernphysik
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

#include "phepex/clip.hpp"

#include <cstddef>

#include "internal.hpp"

namespace phepex {

void pos_soft_clip(const float *waveforms, int n_ch, int n_pix, int n_up, float scale,
                   int sample_lo, int sample_hi, float *out) {
    int lo, hi;
    detail::resolve_range(sample_lo, sample_hi, n_up, lo, hi);
    const std::size_t n_rows = static_cast<std::size_t>(n_ch) * n_pix;

    for (std::size_t r = 0; r < n_rows; ++r) {
        const float *src = waveforms + r * n_up;
        float *dst = out + r * n_up;
        // Zero only outside [lo,hi); the clip loop writes every sample inside it.
        for (int s = 0; s < lo; ++s)
            dst[s] = 0.0f;
        for (int s = hi; s < n_up; ++s)
            dst[s] = 0.0f;
        for (int s = lo; s < hi; ++s)
            dst[s] = detail::pos_soft_clip_value(src[s], scale);
    }
}

}  // namespace phepex
