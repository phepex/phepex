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

/// Per-pixel peak index (over samples in [sample_lo, sample_hi)) of the neighbour-summed
/// waveform: average = waveforms[pixel]*local_weight + sum over non-broken neighbours.
/// First-max tie-break, float32 accumulation, no normalisation. Passing sample_lo ==
/// sample_hi == 0 means the full trace.
///
/// Neighbours are given as a CSR adjacency (compressed sparse row: row pointers +
/// column indices): pixel p's neighbours are indices[indptr[p] .. indptr[p+1]).
///
/// @param waveforms  input (n_ch, n_pix, n_up) float32
/// @param indptr     CSR row pointers, length n_pix+1
/// @param indices    CSR column indices (neighbour pixel ids), length indptr[n_pix]
/// @param broken_pixels  (n_ch, n_pix) byte mask (nonzero => broken); broken neighbours
///                        are skipped. A byte mask (rather than bool*) lets callers pass
///                        contiguous byte storage directly without a bool aliasing
///                        hazard.
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
