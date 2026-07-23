// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Max-Planck-Institut für Kernphysik
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

// phepex._core -- nanobind bindings over the phepex C++ kernels.
//
// Thin wrappers: read numpy array shapes, allocate the output array, and call the
// raw-pointer phepex:: functions. All heavy lifting lives in libphepex.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "phepex/phepex.hpp"

namespace nb = nanobind;

namespace {

template <typename T> nb::capsule make_deleter(T *p) {
    return nb::capsule(p, [](void *q) noexcept { delete[] static_cast<T *>(q); });
}

// Validate a [sample_lo, sample_hi) window against the trace length. (0, 0) means the
// full trace and is always valid; otherwise the window must lie within [0, n_up]. Kernels
// index the raw buffer with these bounds unchecked, so an out-of-range window would read
// and write past the row -- guard it here, at the Python boundary, with a clear error.
void validate_range(int sample_lo, int sample_hi, int n_up) {
    if (sample_lo == 0 && sample_hi == 0)
        return;
    if (sample_lo < 0 || sample_hi > n_up || sample_lo > sample_hi)
        throw std::invalid_argument(
            "sample window [sample_lo, sample_hi) must satisfy 0 <= sample_lo <= "
            "sample_hi <= n_samples");
}

nb::ndarray<nb::numpy, float>
pos_soft_clip(nb::ndarray<const float, nb::ndim<3>, nb::c_contig> waveforms, float scale,
              int sample_lo, int sample_hi) {
    const int n_ch = static_cast<int>(waveforms.shape(0));
    const int n_pix = static_cast<int>(waveforms.shape(1));
    const int n_up = static_cast<int>(waveforms.shape(2));
    validate_range(sample_lo, sample_hi, n_up);
    float *out = new float[static_cast<std::size_t>(n_ch) * n_pix * n_up];
    phepex::pos_soft_clip(waveforms.data(), n_ch, n_pix, n_up, scale, sample_lo,
                          sample_hi, out);
    std::size_t shape[3] = {static_cast<std::size_t>(n_ch),
                            static_cast<std::size_t>(n_pix),
                            static_cast<std::size_t>(n_up)};
    return nb::ndarray<nb::numpy, float>(out, 3, shape, make_deleter(out));
}

nb::ndarray<nb::numpy, std::int64_t>
neighbor_peak_indices(nb::ndarray<const float, nb::ndim<3>, nb::c_contig> waveforms,
                      nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> indptr,
                      nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> indices,
                      int local_weight,
                      nb::ndarray<const bool, nb::ndim<2>, nb::c_contig> broken_pixels,
                      int sample_lo, int sample_hi) {
    const int n_ch = static_cast<int>(waveforms.shape(0));
    const int n_pix = static_cast<int>(waveforms.shape(1));
    const int n_up = static_cast<int>(waveforms.shape(2));
    validate_range(sample_lo, sample_hi, n_up);
    // The kernel dereferences CSR neighbour ids and broken_pixels using n_pix taken from
    // the waveforms array. Confirm the neighbour matrix and broken_pixels were built for
    // the same pixel count, otherwise a neighbour id >= n_pix reads past the buffers.
    // (For empty waveforms no pixel is dereferenced, so the matrix size is irrelevant.)
    if (n_pix > 0 && static_cast<int>(indptr.shape(0)) != n_pix + 1)
        throw std::invalid_argument(
            "neighbour matrix indptr length must be n_pix + 1 (neighbour matrix and "
            "waveforms disagree on pixel count)");
    if (static_cast<int>(broken_pixels.shape(0)) != n_ch ||
        static_cast<int>(broken_pixels.shape(1)) != n_pix)
        throw std::invalid_argument(
            "broken_pixels shape must be (n_channels, n_pix) matching waveforms");
    std::int64_t *out = new std::int64_t[static_cast<std::size_t>(n_ch) * n_pix];
    // The kernel takes a byte mask; a numpy bool array is genuine 1-byte bool storage,
    // and reading a bool object through `unsigned char` is permitted by the aliasing
    // rules, so this reinterpret is well-defined (unlike char storage read as bool*).
    phepex::neighbor_peak_indices(
        waveforms.data(), n_ch, n_pix, n_up, indptr.data(), indices.data(), local_weight,
        reinterpret_cast<const std::uint8_t *>(broken_pixels.data()), sample_lo,
        sample_hi, out);
    std::size_t shape[2] = {static_cast<std::size_t>(n_ch),
                            static_cast<std::size_t>(n_pix)};
    return nb::ndarray<nb::numpy, std::int64_t>(out, 2, shape, make_deleter(out));
}

std::pair<nb::ndarray<nb::numpy, float>, nb::ndarray<nb::numpy, float>>
extract_around_peak(nb::ndarray<const float, nb::ndim<3>, nb::c_contig> waveforms,
                    nb::ndarray<const std::int64_t, nb::ndim<2>, nb::c_contig> peak_index,
                    int width, int shift, double sampling_rate_ghz) {
    const int n_ch = static_cast<int>(waveforms.shape(0));
    const int n_pix = static_cast<int>(waveforms.shape(1));
    const int n_up = static_cast<int>(waveforms.shape(2));
    const std::size_t n = static_cast<std::size_t>(n_ch) * n_pix;
    float *charge = new float[n];
    float *ptime = new float[n];
    phepex::extract_around_peak(waveforms.data(), n_ch, n_pix, n_up, peak_index.data(),
                                width, shift, sampling_rate_ghz, charge, ptime);
    std::size_t shape[2] = {static_cast<std::size_t>(n_ch),
                            static_cast<std::size_t>(n_pix)};
    return {nb::ndarray<nb::numpy, float>(charge, 2, shape, make_deleter(charge)),
            nb::ndarray<nb::numpy, float>(ptime, 2, shape, make_deleter(ptime))};
}

nb::ndarray<nb::numpy, float>
adaptive_centroid(nb::ndarray<const float, nb::ndim<3>, nb::c_contig> waveforms,
                  nb::ndarray<const std::int64_t, nb::ndim<2>, nb::c_contig> peak_index,
                  double rel_descend_limit) {
    const int n_ch = static_cast<int>(waveforms.shape(0));
    const int n_pix = static_cast<int>(waveforms.shape(1));
    const int n_up = static_cast<int>(waveforms.shape(2));
    float *out = new float[static_cast<std::size_t>(n_ch) * n_pix];
    phepex::adaptive_centroid(waveforms.data(), n_ch, n_pix, n_up, peak_index.data(),
                              rel_descend_limit, out);
    std::size_t shape[2] = {static_cast<std::size_t>(n_ch),
                            static_cast<std::size_t>(n_pix)};
    return nb::ndarray<nb::numpy, float>(out, 2, shape, make_deleter(out));
}

// preprocess_waveform is a single-waveform (1-D) kernel; this wrapper applies it to every
// (channel, pixel) row of a 3-D batch. pole_zero/baseline/scale each arrive as a float32
// array of length 1 (one value applied to every row) or n_ch*n_pix (a distinct
// per-(channel, pixel) value); the Python wrapper produces the length-1 form for scalar
// inputs, so the common scalar call does not allocate a full per-row array. A length-1
// array is read with row stride 0. smoothing_fwhm > 0 enables the Deriche IIR pass with
// coefficients computed once and shared across rows.
nb::ndarray<nb::numpy, float>
preprocess(nb::ndarray<const float, nb::ndim<3>, nb::c_contig> waveforms, int upsampling,
           nb::ndarray<const float, nb::ndim<1>, nb::c_contig> pole_zero,
           double smoothing_fwhm,
           nb::ndarray<const float, nb::ndim<1>, nb::c_contig> baseline,
           nb::ndarray<const float, nb::ndim<1>, nb::c_contig> scale) {
    // upsampling < 1 would divide by upsampling^2 == 0 and read output[-1] in
    // detail::upsample_waveform (undefined). Guard here, at the Python boundary, so no
    // _core.preprocess caller reaches the kernel with it.
    if (upsampling < 1)
        throw std::invalid_argument("upsampling must be >= 1");
    const int n_ch = static_cast<int>(waveforms.shape(0));
    const int n_pix = static_cast<int>(waveforms.shape(1));
    const int n_samples = static_cast<int>(waveforms.shape(2));
    const std::size_t n_rows = static_cast<std::size_t>(n_ch) * n_pix;
    // Each per-row array is length 1 (broadcast to all rows, row stride 0) or n_rows (one
    // value per row, stride 1).
    auto row_stride = [n_rows](std::size_t len) -> std::size_t {
        if (len == 1)
            return 0;
        if (len == n_rows)
            return 1;
        throw std::invalid_argument(
            "pole_zero, baseline and scale must each have length 1 or n_channels*n_pix");
    };
    const std::size_t pz_stride = row_stride(pole_zero.shape(0));
    const std::size_t bl_stride = row_stride(baseline.shape(0));
    const std::size_t scl_stride = row_stride(scale.shape(0));

    // smoothing_fwhm == 0 disables smoothing; calculate_smoothing_coefficients divides by
    // fwhm/2.1, so it must not be called with a non-positive width.
    phepex::SmoothingCoefficients sc;
    const phepex::SmoothingCoefficients *sm = nullptr;
    if (smoothing_fwhm > 0.0) {
        sc = phepex::calculate_smoothing_coefficients(smoothing_fwhm);
        sm = &sc;
    }

    const std::size_t out_row = static_cast<std::size_t>(n_samples) * upsampling;
    float *out = new float[n_rows * out_row];

    // The scratch buffer is only read/written within a single preprocess_waveform call
    // (smoothing path), so one buffer can be reused across the sequential row loop.
    std::vector<float> scratch;
    float *scratch_ptr = nullptr;
    if (sm != nullptr) {
        scratch.resize(out_row);
        scratch_ptr = scratch.data();
    }

    const float *src = waveforms.data();
    const float *pz = pole_zero.data();
    const float *bl = baseline.data();
    const float *scl = scale.data();
    for (std::size_t r = 0; r < n_rows; r++)
        phepex::preprocess_waveform(src + r * n_samples, n_samples, upsampling,
                                    pz[r * pz_stride], sm, bl[r * bl_stride],
                                    scl[r * scl_stride], out + r * out_row, scratch_ptr);

    std::size_t shape[3] = {static_cast<std::size_t>(n_ch),
                            static_cast<std::size_t>(n_pix), out_row};
    return nb::ndarray<nb::numpy, float>(out, 3, shape, make_deleter(out));
}

std::pair<int, int> preprocess_valid_range(int upsampling, double pole_zero,
                                           double smoothing_fwhm, int n_samples) {
    // Reject upsampling < 1 rather than let phepex::preprocess_valid_range clamp it to 1
    // and return a range for an upsampling that preprocess() itself refuses.
    if (upsampling < 1)
        throw std::invalid_argument("upsampling must be >= 1");
    phepex::SmoothingCoefficients sc;
    const phepex::SmoothingCoefficients *sm = nullptr;
    if (smoothing_fwhm > 0.0) {
        sc = phepex::calculate_smoothing_coefficients(smoothing_fwhm);
        sm = &sc;
    }
    const phepex::SampleRange r = phepex::preprocess_valid_range(
        upsampling, static_cast<float>(pole_zero), sm, n_samples);
    return {r.lo, r.hi};
}

nb::ndarray<nb::numpy, float>
generate_waveforms(nb::ndarray<const double, nb::ndim<2>, nb::c_contig> charge,
                   nb::ndarray<const double, nb::ndim<2>, nb::c_contig> time_ns,
                   nb::ndarray<const double, nb::ndim<1>, nb::c_contig> reference_pulse,
                   double ref_sample_width_ns, double sample_width_ns, int n_samples,
                   int upsampling, double nsb_rate_ghz, std::uint64_t seed) {
    const int n_events = static_cast<int>(charge.shape(0));
    const int n_pix = static_cast<int>(charge.shape(1));
    if (time_ns.shape(0) != charge.shape(0) || time_ns.shape(1) != charge.shape(1))
        throw std::runtime_error("charge and time_ns must have the same shape");
    float *out = new float[static_cast<std::size_t>(n_events) * n_pix * n_samples];
    phepex::generate_waveforms(
        charge.data(), time_ns.data(), n_events, n_pix, reference_pulse.data(),
        static_cast<int>(reference_pulse.shape(0)), ref_sample_width_ns, sample_width_ns,
        n_samples, upsampling, nsb_rate_ghz, seed, out);
    std::size_t shape[3] = {static_cast<std::size_t>(n_events),
                            static_cast<std::size_t>(n_pix),
                            static_cast<std::size_t>(n_samples)};
    return nb::ndarray<nb::numpy, float>(out, 3, shape, make_deleter(out));
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "phepex C++ kernels (photo-electron pulse extraction).";
    m.def("preprocess", &preprocess, nb::arg("waveforms"), nb::arg("upsampling"),
          nb::arg("pole_zero"), nb::arg("smoothing_fwhm"), nb::arg("baseline"),
          nb::arg("scale"),
          "Upsample + pole-zero deconvolution + optional Deriche smoothing per (channel, "
          "pixel); float32 (n_ch,n_pix,n_samples*upsampling). upsampling must be >= 1; "
          "pole_zero/baseline/scale are length-1 or (n_ch*n_pix,) arrays; smoothing_fwhm "
          "<= "
          "0 disables smoothing.");
    m.def("preprocess_valid_range", &preprocess_valid_range, nb::arg("upsampling"),
          nb::arg("pole_zero"), nb::arg("smoothing_fwhm"), nb::arg("n_samples"),
          "Trustworthy (non-edge) output-sample range (lo, hi) of preprocess (DVR "
          "convention on the raw n_samples).");
    m.def("pos_soft_clip", &pos_soft_clip, nb::arg("waveforms"), nb::arg("scale"),
          nb::arg("sample_lo") = 0, nb::arg("sample_hi") = 0,
          "Positive soft clip max(soft_clip(waveforms/scale), 0) over [sample_lo, "
          "sample_hi).");
    m.def("neighbor_peak_indices", &neighbor_peak_indices, nb::arg("waveforms"),
          nb::arg("indptr"), nb::arg("indices"), nb::arg("local_weight"),
          nb::arg("broken_pixels"), nb::arg("sample_lo") = 0, nb::arg("sample_hi") = 0,
          "Per-pixel peak sample index of the neighbour-summed waveform over [sample_lo, "
          "sample_hi).");
    m.def(
        "extract_around_peak", &extract_around_peak, nb::arg("waveforms"),
        nb::arg("peak_index"), nb::arg("width"), nb::arg("shift"),
        nb::arg("sampling_rate_ghz"),
        "Window integration + weighted peak time. Returns (charge, peak_time) float32.");
    m.def("adaptive_centroid", &adaptive_centroid, nb::arg("waveforms"),
          nb::arg("peak_index"), nb::arg("rel_descend_limit"),
          "Leading-edge centroid (sample units) per pixel.");
    m.def(
        "generate_waveforms", &generate_waveforms, nb::arg("charge"), nb::arg("time_ns"),
        nb::arg("reference_pulse"), nb::arg("ref_sample_width_ns"),
        nb::arg("sample_width_ns"), nb::arg("n_samples"), nb::arg("upsampling") = 10,
        nb::arg("nsb_rate_ghz") = 0.0, nb::arg("seed") = 0,
        R"doc(Synthesize (n_events, n_pixels, n_samples) float32 waveforms from per-pixel
(charge, peak_time): a physically-placed signal deposit plus Poisson NSB. Matches ctapipe
WaveformModel for deposits inside the readout window, but is physically correct at the
edges (floor-snapped deposit time, and pulses centred just outside the window still
contribute their in-window tail rather than being dropped as WaveformModel does).)doc");
}
