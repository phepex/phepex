// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Max-Planck-Institut für Kernphysik
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

#ifndef PHEPEX_NEIGHBOR_HPP
#define PHEPEX_NEIGHBOR_HPP

#include <cstdint>

namespace phepex {

/// Per-pixel argmax of the neighbour-summed waveform
/// `waveforms[pixel]*local_weight + sum over non-broken neighbours`, searched over
/// samples [sample_lo, sample_hi) and returned as an absolute index into the trace.
/// First-max tie-break, float32 accumulation, no normalisation by the neighbour count (so
/// this is a sum, not the average its ctapipe counterpart's name suggests; the argmax is
/// unaffected).
///
/// By default (compile-time option PHEPEX_NEIGHBOR_PAIRWISE_SUM=1) neighbours are
/// accumulated in pairs to halve the read-modify-write traffic on the accumulator (the
/// micro-benchmark bottleneck), and for local_weight == 0 the accumulator is seeded from
/// the first neighbour pair rather than a zeroed self*0 pass. Both change the float32
/// summation order, so the peak index is not bit-identical to a strictly sequential sum;
/// it differs only where two samples are within ~1 ULP of the maximum. Building with
/// PHEPEX_NEIGHBOR_PAIRWISE_SUM=0 selects the sequential sum (slower on the benchmarked
/// aarch64 core, but bit-identical to a left-to-right accumulation in CSR order).
///
/// Neighbours are given as a CSR adjacency (compressed sparse row: row pointers +
/// column indices): pixel p's neighbours are indices[indptr[p] .. indptr[p+1]).
///
/// @param waveforms  input (n_ch, n_pix, n_up) float32
/// @param indptr     CSR row pointers, length n_pix+1
/// @param indices    CSR column indices (neighbour pixel ids), length indptr[n_pix]
/// @param local_weight  weight of the pixel's own trace in the sum; 0 sums neighbours
/// only
/// @param broken_pixels  (n_ch, n_pix) byte mask (nonzero => broken); broken neighbours
///                        are skipped. A byte mask (rather than bool*) lets callers pass
///                        contiguous byte storage directly without a bool aliasing
///                        hazard. The pixel's own flag is not consulted -- a broken pixel
///                        still gets a peak index.
/// @param sample_lo, sample_hi  half-open search window; (0, 0) means the full trace.
///                        Bounds are used unchecked, so the caller must keep them within
///                        [0, n_up].
/// @param peak_out   caller-allocated n_ch*n_pix int64 (argmax sample index per pixel)
/// @param neighbor_count  optional n_ch*n_pix int32 (caller-allocated); if non-null, gets
///                        the number of non-broken neighbours summed per pixel
/// @param scratch    optional caller-provided workspace of at least n_up floats; if null,
///                   a buffer is allocated internally for the duration of the call
void neighbor_peak_indices(const float *waveforms, int n_ch, int n_pix, int n_up,
                           const std::int32_t *indptr, const std::int32_t *indices,
                           int local_weight, const std::uint8_t *broken_pixels,
                           int sample_lo, int sample_hi, std::int64_t *peak_out,
                           std::int32_t *neighbor_count = nullptr,
                           float *scratch = nullptr);

}  // namespace phepex

#endif  // PHEPEX_NEIGHBOR_HPP
