// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Max-Planck-Institut für Kernphysik
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

// Minimal standalone C++ user of libphepex: deconvolve -> clip -> neighbour peak ->
// extract.
//
// Build with CMake:   find_package(phepex REQUIRED); target_link_libraries(app
// phepex::phepex) or with pkg-config: g++ -std=c++17 example.cpp $(pkg-config --cflags
// --libs phepex)

#include <cstdint>
#include <cstdio>
#include <vector>

#include <phepex/phepex.hpp>

int main() {
    // A tiny camera: 3 pixels in a line, one channel, 8 samples each.
    const int n_ch = 1, n_pix = 3, n_samples = 8, upsampling = 4;
    const int n_up = n_samples * upsampling;

    // A single Gaussian-ish pulse on the middle pixel; neighbours see a smaller echo.
    std::vector<float> wf(n_ch * n_pix * n_samples, 0.0f);
    const float pulse[n_samples] = {0, 1, 4, 9, 6, 2, 0, 0};
    for (int s = 0; s < n_samples; ++s) {
        wf[0 * n_samples + s] = 0.3f * pulse[s];  // pixel 0
        wf[1 * n_samples + s] = 1.0f * pulse[s];  // pixel 1 (brightest)
        wf[2 * n_samples + s] = 0.3f * pulse[s];  // pixel 2
    }

    // Pole-zero deconvolution + upsampling. pole_zero/baseline/scale are per-(channel,
    // pixel); here every pixel shares pole_zero=0.75 (baseline/scale left at the
    // defaults 0/1 via nullptr).
    std::vector<float> deconv(static_cast<std::size_t>(n_ch) * n_pix * n_up);
    std::vector<float> pole_zero(static_cast<std::size_t>(n_ch) * n_pix, 0.75f);
    phepex::deconvolve_upsample(wf.data(), n_ch, n_pix, n_samples, upsampling,
                                pole_zero.data(), /*baseline=*/nullptr, /*scale=*/nullptr,
                                deconv.data());

    // Neighbour-sum peak search over the valid sample range.
    const phepex::SampleRange vr =
        phepex::deconvolve_valid_range(upsampling, n_samples, 0.75);
    // CSR line graph: 0-1-2 (pixel 1 neighbours 0 and 2).
    const std::int32_t indptr[4] = {0, 1, 3, 4};
    const std::int32_t indices[4] = {1, 0, 2, 1};
    // std::vector<bool> is bit-packed, so use a plain byte buffer for the bool* API.
    std::vector<unsigned char> broken(n_ch * n_pix, 0);
    std::vector<std::int64_t> peak(n_ch * n_pix);
    phepex::neighbor_peak_indices(deconv.data(), n_ch, n_pix, n_up, indptr, indices,
                                  /*local_weight=*/1,
                                  reinterpret_cast<const bool *>(broken.data()), vr.lo,
                                  vr.hi, peak.data());

    // Window integration + weighted peak time.
    std::vector<float> charge(n_ch * n_pix), peak_time(n_ch * n_pix);
    phepex::extract_around_peak(deconv.data(), n_ch, n_pix, n_up, peak.data(),
                                /*width=*/7, /*shift=*/3,
                                /*sampling_rate_ghz=*/0.25 * upsampling, charge.data(),
                                peak_time.data());

    std::printf("phepex %s  (valid samples [%d, %d) of %d)\n", PHEPEX_VERSION, vr.lo,
                vr.hi, n_up);
    for (int p = 0; p < n_pix; ++p)
        std::printf("  pixel %d: peak=%2lld  charge=%8.3f  peak_time=%7.3f ns\n", p,
                    static_cast<long long>(peak[p]), charge[p], peak_time[p]);
    return 0;
}
