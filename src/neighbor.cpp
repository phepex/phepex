// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Max-Planck-Institut für Kernphysik
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

#include "phepex/neighbor.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "internal.hpp"

namespace phepex {

void neighbor_peak_indices(const float *waveforms, int n_ch, int n_pix, int n_up,
                           const std::int32_t *indptr, const std::int32_t *indices,
                           int local_weight, const std::uint8_t *broken_pixels,
                           int sample_lo, int sample_hi, std::int64_t *peak_out,
                           std::int32_t *neighbor_count, float *scratch) {
    int lo, hi;
    detail::resolve_range(sample_lo, sample_hi, n_up, lo, hi);
    const int w = hi - lo;
    const float lw = static_cast<float>(local_weight);
    std::vector<float> owned;
    float *buf = scratch;
    if (buf == nullptr) {
        owned.resize(w > 0 ? w : 1);
        buf = owned.data();
    }

    for (std::size_t ch = 0; ch < static_cast<std::size_t>(n_ch); ++ch) {
        for (std::size_t pix = 0; pix < static_cast<std::size_t>(n_pix); ++pix) {
            const float *self = waveforms + (ch * n_pix + pix) * n_up;
            for (int j = 0; j < w; ++j)
                buf[j] = self[lo + j] * lw;
            std::int32_t count = 0;
            for (std::int32_t k = indptr[pix]; k < indptr[pix + 1]; ++k) {
                const std::int32_t nb_pix = indices[k];
                if (broken_pixels[ch * n_pix + nb_pix])
                    continue;
                ++count;
                const float *nw = waveforms + (ch * n_pix + nb_pix) * n_up;
                for (int j = 0; j < w; ++j)
                    buf[j] += nw[lo + j];
            }
            if (neighbor_count)
                neighbor_count[ch * n_pix + pix] = count;
            int best = 0;
            for (int j = 1; j < w; ++j)
                if (buf[j] > buf[best])
                    best = j;
            peak_out[ch * n_pix + pix] = static_cast<std::int64_t>(lo + best);
        }
    }
}

}  // namespace phepex
