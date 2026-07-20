// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Max-Planck-Institut für Kernphysik
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

#ifndef PHEPEX_GENERATE_HPP
#define PHEPEX_GENERATE_HPP

#include <cstdint>

namespace phepex {

/// Synthesize sampled waveforms from per-pixel (charge, peak_time): for the signal
/// component, each pixel's charge is deposited at its sub-sample peak time and convolved
/// with the reference pulse shape; a temporally-distributed Poisson night-sky-background
/// (NSB) process at nsb_rate_ghz is added on top. Intended as a fast generator
/// for testing/benchmarking extractors.
///
/// @param charge          (n_events, n_pix) float64 photo-electron charge per pixel
/// @param time_ns         (n_events, n_pix) float64 pulse peak time (ns) per pixel
/// @param reference_pulse (n_ref) float64 single-channel reference pulse shape
/// @param ref_sample_width_ns  sample width of the reference pulse (ns)
/// @param sample_width_ns      readout sample width (ns) = 1 / sampling_rate
/// @param upsampling      internal upsampling used to place sub-sample deposits (e.g. 10)
/// @param nsb_rate_ghz    NSB rate in GHz (0 disables NSB)
/// @param seed            RNG seed (per-event deterministic)
/// @param out             caller-allocated n_events*n_pix*n_samples float32,
///                        row-major (n_events, n_pix, n_samples)
void generate_waveforms(const double *charge, const double *time_ns, int n_events,
                        int n_pix, const double *reference_pulse, int n_ref,
                        double ref_sample_width_ns, double sample_width_ns, int n_samples,
                        int upsampling, double nsb_rate_ghz, std::uint64_t seed,
                        float *out);

}  // namespace phepex

#endif  // PHEPEX_GENERATE_HPP
