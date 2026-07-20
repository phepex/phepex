// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Max-Planck-Institut für Kernphysik
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

#include "phepex/deconvolve.hpp"

#include <cstddef>

#include "internal.hpp"

namespace phepex {

void deconvolve_upsample(const float *waveforms, int n_ch, int n_pix, int n_samples,
                         int upsampling, const float *pole_zero, const float *baseline,
                         const float *scale, float *out) {
    const std::size_t n_rows = static_cast<std::size_t>(n_ch) * n_pix;
    const std::size_t n_out = static_cast<std::size_t>(n_samples) * upsampling;
    for (std::size_t r = 0; r < n_rows; ++r) {
        const float pz = pole_zero ? pole_zero[r] : 0.0f;
        const float bl = baseline ? baseline[r] : 0.0f;
        const float sc = scale ? scale[r] : 1.0f;
        detail::upsample_waveform<float, float>(n_samples, upsampling,
                                                waveforms + r * n_samples, bl, pz,
                                                out + r * n_out, sc);
    }
}

SampleRange deconvolve_valid_range(int upsampling, int n_samples, double pole_zero) {
    const int n_up = upsampling * n_samples;
    const int tail = 2 * (upsampling - 1);
    const int lo = (pole_zero != 0.0) ? (3 * upsampling - 2) : tail;
    return SampleRange{lo, n_up - tail};
}

}  // namespace phepex
