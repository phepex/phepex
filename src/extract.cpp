// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Max-Planck-Institut für Kernphysik
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

#include "phepex/extract.hpp"

#include <cstddef>

namespace phepex {

void extract_around_peak(const float *waveforms, int n_ch, int n_pix, int n_up,
                         const std::int64_t *peak_index, int width, int shift,
                         double sampling_rate_ghz, float *charge, float *peak_time) {
    const std::size_t n_rows = static_cast<std::size_t>(n_ch) * n_pix;
    for (std::size_t r = 0; r < n_rows; ++r) {
        const float *w = waveforms + r * n_up;
        const long peak = static_cast<long>(peak_index[r]);
        long start = peak - shift;
        long end = start + width;
        if (start < 0)
            start = 0;
        if (end > n_up)
            end = n_up;

        double i_sum = 0.0, num = 0.0, den = 0.0;
        for (long s = start; s < end; ++s) {
            const double v = w[s];
            i_sum += v;
            if (v > 0.0) {
                num += v * static_cast<double>(s);
                den += v;
            }
        }
        // Weighted peak time computed fully in double, rounded once on store
        const double pt =
            (den > 0.0 ? num / den : static_cast<double>(peak)) / sampling_rate_ghz;
        charge[r] = static_cast<float>(i_sum);
        peak_time[r] = static_cast<float>(pt);
    }
}

void adaptive_centroid(const float *waveforms, int n_ch, int n_pix, int n_up,
                       const std::int64_t *peak_index, double rel_descend_limit,
                       float *centroids) {
    const std::size_t n_rows = static_cast<std::size_t>(n_ch) * n_pix;
    for (std::size_t r = 0; r < n_rows; ++r) {
        const float *w = waveforms + r * n_up;
        const long peak = static_cast<long>(peak_index[r]);
        centroids[r] = static_cast<float>(peak);  // preload in case of early return
        if (n_up == 0 || peak < 0 || peak > n_up - 1)
            continue;

        const float peak_amp = w[peak];
        if (peak_amp < 0.0f)
            continue;

        // Fixed descend threshold; do not recompute this limit even if a sample larger
        // than peak_amp is found during the traversal
        const auto descend = static_cast<float>(rel_descend_limit * peak_amp);

        double sum = 0.0, jsum = 0.0;
        long j = peak;
        while (j >= 0 && w[j] > descend) {
            sum += w[j];
            jsum += static_cast<double>(j) * static_cast<double>(w[j]);
            --j;
        }
        j = peak + 1;
        while (j < n_up && w[j] > descend) {
            sum += w[j];
            jsum += static_cast<double>(j) * static_cast<double>(w[j]);
            ++j;
        }
        if (sum != 0.0)
            centroids[r] = static_cast<float>(jsum / sum);
    }
}

}  // namespace phepex
