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

// Neighbour-sum accumulation strategy for neighbor_peak_indices. 1 = pairwise (default),
// 0 = sequential. The CMake build always defines this explicitly (option
// PHEPEX_NEIGHBOR_PAIRWISE_SUM); this guard only supplies a default for compilation
// outside CMake.
#ifndef PHEPEX_NEIGHBOR_PAIRWISE_SUM
#define PHEPEX_NEIGHBOR_PAIRWISE_SUM 1
#endif

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
            const std::size_t row = ch * n_pix + pix;
            std::int32_t count = 0;

#if PHEPEX_NEIGHBOR_PAIRWISE_SUM
            // Accumulate neighbours in pairs (buf[j] += a[j] + b[j]). This halves the
            // read-modify-write traffic on buf, which the per-kernel micro-benchmark
            // identifies as the bottleneck (neighbour reads are L1/L2 resident and cheap;
            // the repeated buf load+store is not): ~20% faster for local_weight != 0 on
            // Zen 4. For local_weight == 0 the self term contributes nothing, so buf is
            // seeded from the first neighbour pair instead of a zeroed self*0 pass,
            // skipping one full sweep over the trace (~26% faster). Pairwise grouping
            // changes the float32 summation order, so the argmax differs from a strictly
            // sequential sum only in near-ties within ~1 ULP.
            bool seeded = false;
            if (local_weight != 0) {
                const float *self = waveforms + row * n_up + lo;
                for (int j = 0; j < w; ++j)
                    buf[j] = self[j] * lw;
                seeded = true;
            }
            const float *pend = nullptr;  // one neighbour held back to pair with the next
            for (std::int32_t k = indptr[pix]; k < indptr[pix + 1]; ++k) {
                const std::int32_t nb_pix = indices[k];
                if (broken_pixels[ch * n_pix + nb_pix])
                    continue;
                ++count;
                const float *nw = waveforms + (ch * n_pix + nb_pix) * n_up + lo;
                if (pend == nullptr) {
                    pend = nw;
                } else if (seeded) {
                    for (int j = 0; j < w; ++j)
                        buf[j] += pend[j] + nw[j];
                    pend = nullptr;
                } else {
                    for (int j = 0; j < w; ++j)
                        buf[j] = pend[j] + nw[j];
                    seeded = true;
                    pend = nullptr;
                }
            }
            if (pend != nullptr) {  // odd neighbour left over
                if (seeded)
                    for (int j = 0; j < w; ++j)
                        buf[j] += pend[j];
                else {
                    for (int j = 0; j < w; ++j)
                        buf[j] = pend[j];
                    seeded = true;
                }
            }
            if (!seeded)  // local_weight == 0 with no non-broken neighbours: sum is zero
                for (int j = 0; j < w; ++j)
                    buf[j] = 0.0f;
#else
            // Sequential sum: buf = self*local_weight, then add each non-broken neighbour
            // one at a time, in CSR index order.
            const float *self = waveforms + row * n_up + lo;
            for (int j = 0; j < w; ++j)
                buf[j] = self[j] * lw;
            for (std::int32_t k = indptr[pix]; k < indptr[pix + 1]; ++k) {
                const std::int32_t nb_pix = indices[k];
                if (broken_pixels[ch * n_pix + nb_pix])
                    continue;
                ++count;
                const float *nw = waveforms + (ch * n_pix + nb_pix) * n_up + lo;
                for (int j = 0; j < w; ++j)
                    buf[j] += nw[j];
            }
#endif

            if (neighbor_count)
                neighbor_count[row] = count;
            int best = 0;
            for (int j = 1; j < w; ++j)
                if (buf[j] > buf[best])
                    best = j;
            peak_out[row] = static_cast<std::int64_t>(lo + best);
        }
    }
}

}  // namespace phepex
