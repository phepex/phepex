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

/// Parameters of a single artificial shower image, in camera-plane metres / radians / ns.
struct ShowerModel {
    double centroid_x_m = 0;  ///< image centroid, x
    double centroid_y_m = 0;  ///< image centroid, y
    double length_m = 0;      ///< major-axis width (standard deviation, incl. any skew)
    double width_m = 0;       ///< minor-axis width (Gaussian sigma)
    double psi_rad = 0;       ///< major-axis rotation (0 = +x axis)
    double skewness = 0;      ///< skew along the major axis only; 0 = symmetric,
                              ///< |skewness| < 0.995271746431 (the skew-normal's limit)
    double intensity_pe = 0;  ///< total expected photo-electrons in the image
    double time_gradient_ns_per_m = 0;  ///< d(peak time) / d(longitudinal distance)
    double time_intercept_ns = 0;       ///< peak time at the centroid
    double time_jitter_ns = 0;  ///< half-width of the uniform per-pixel time jitter
};

/// Fill per-pixel (charge, time_ns) for one artificial shower image, as input for
/// generate_waveforms().
///
/// Charges are Poisson draws around `intensity_pe * pdf(x, y) * pixel_area_m2`, where pdf
/// is a Gaussian of sigma `width_m` in the transverse coordinate times a skew-normal of
/// scale/location derived from (`length_m`, `skewness`) in the longitudinal one. Times
/// are linear in the longitudinal coordinate plus jitter drawn from
/// [-time_jitter_ns, +time_jitter_ns]; they are written for every pixel, including those
/// whose charge is 0 (which generate_waveforms skips). Assumes a uniform pixel area, and
/// that the pixel size is small against `width_m` (the pdf is sampled at the pixel centre
/// rather than integrated over the pixel).
///
/// Throws std::invalid_argument unless `width_m > 0`, `length_m > 0` (both appear in a
/// denominator), `intensity_pe >= 0`, `pixel_area_m2 > 0` and
/// `|skewness| < 0.995271746431`; each would otherwise give a NaN or negative Poisson
/// mean, which is indistinguishable from a genuinely dim image in the output.
///
/// @param model          shower parameters
/// @param pix_x, pix_y   (n_pix) pixel centre coordinates (m)
/// @param pixel_area_m2  area of one pixel (m^2)
/// @param seed           RNG seed (deterministic, independent of generate_waveforms())
/// @param charge         caller-allocated (n_pix) float64 output, p.e. per pixel
/// @param time_ns        caller-allocated (n_pix) float64 output, peak time (ns) per
/// pixel
void generate_shower_image(const ShowerModel &model, const double *pix_x,
                           const double *pix_y, int n_pix, double pixel_area_m2,
                           std::uint64_t seed, double *charge, double *time_ns);

}  // namespace phepex

#endif  // PHEPEX_GENERATE_HPP
