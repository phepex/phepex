// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Max-Planck-Institut für Kernphysik
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

// Internal (non-installed) helpers shared across phepex source files.
#ifndef PHEPEX_SRC_INTERNAL_HPP
#define PHEPEX_SRC_INTERNAL_HPP

#include <cmath>

namespace phepex {
namespace detail {

// Soft clip x/scale to (-1,1) via y/(1+|y|), then clamp negatives to 0. The soft clip
// already bounds the result to (-1,1), so no upper clip is needed. Float32 arithmetic
// throughout (matches a numpy float32 reference exactly).
inline float pos_soft_clip_value(float x, float scale) {
    float y = x / scale;
    y = y / (1.0f + std::fabs(y));
    return y < 0.0f ? 0.0f : y;
}

// Resolve a [lo, hi) sample window; (0,0) => the full trace [0, n_up).
inline void resolve_range(int sample_lo, int sample_hi, int n_up, int &lo, int &hi) {
    if (sample_lo == 0 && sample_hi == 0) {
        lo = 0;
        hi = n_up;
    } else {
        lo = sample_lo;
        hi = sample_hi;
    }
}

// Apply upsampling and first order corrections to one waveform.
//
// Steps: repeat the n input values upsampling_factor times each; subtract baseline offset
// and apply a scaling factor; correct for a single pole decay with decay time pole_zero;
// smooth with two upsampling_factor-wide moving averages. Produces n*upsampling_factor
// samples with 2*(upsampling_factor-1) invalid samples at each end (3*upsampling_factor-2
// at the start if pole_zero != 0).
//
// Note: a "single-write" two-pass variant was tried and measured ~1.45x SLOWER — the
// per-pixel output is L1-resident, so this tight single-pass loop is already the faster
// form. Do not "optimize" it into two passes.
template <typename InputType, typename OutputType>
void upsample_waveform(int n, int upsampling_factor, const InputType *input,
                       OutputType offset, OutputType pole_zero, OutputType *output,
                       OutputType scale) {
    OutputType v2, v1;          // the next and prev. input samples
    OutputType sum1, sum2;      // the running sum of 1.st and 2.nd average
    OutputType tmp;             // a temp var for intermediate copy
    OutputType pzc2, pzc1;      // the next and prev. pz corrected value
    OutputType *out1 = output;  // the out pointer of the first runsum
    OutputType *out2 = output;  // the out pointer of the second runsum
    OutputType mult =
        scale / (upsampling_factor *
                 upsampling_factor);  // the multiplier to correct the two runsums

    v1 = v2 = (input[0] - offset) * mult;
    pzc2 = pzc1 = v2;
    sum1 = pzc2 * upsampling_factor;
    sum2 = sum1 * upsampling_factor;
    for (int i = 0; i < upsampling_factor; i++)
        *out1++ = sum1;
    for (int i = 1; i < n; i++) {
        v2 = (input[i] - offset) * mult;
        pzc2 = (v2 - v1);
        v1 = v2;
        for (int i1 = 0; i1 < upsampling_factor; i1++) {
            sum1 += pzc2 - pzc1 * pole_zero;
            *out1++ = sum1;
            tmp = *out2;
            *out2++ = sum2;
            sum2 += sum1 - tmp;
        }
        pzc1 = pzc2;
    }
    n *= upsampling_factor;
    for (v2 = output[n - 1]; out2 < (output + n);) {
        tmp = *out2;
        *out2++ = sum2;
        sum2 += v2 - tmp;
    }
}

}  // namespace detail
}  // namespace phepex

#endif  // PHEPEX_SRC_INTERNAL_HPP
