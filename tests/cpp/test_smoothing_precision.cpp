// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Max-Planck-Institut für Kernphysik
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

// Numerical-stability tests for the Deriche smoothing pass of preprocess_waveform().
//
// Motivation: the pass keeps its intermediate state (and its coefficients) in float32.
// These tests quantify what that costs against a long double evaluation of the same
// recursion, for worst-case but realistic Cherenkov waveforms and for traces longer than
// a nominal FlashCam readout window.
//
// Two error sources are measured separately, because they behave differently:
//
//  1. State precision -- rounding noise injected into a recursive filter. Its gain is set
//     by the pole radius r = exp(-b/sigma), sigma = fwhm/2.1, and scales like 1/(1-r)^2
//     (noise_gain() below). Because the filter forgets in ~1/(1-r) samples, this
//     error does NOT accumulate with trace length; the length sweep verifies that rather
//     than assuming it.
//  2. Coefficient precision -- float32 coefficients move the poles, which perturbs the
//     filter itself (DC gain, impulse-response width). That is a systematic bias, not
//     noise, so it is tested on the impulse and step responses where it is not masked by
//     the state noise.
//
// The references are exact emulations of the kernel's arithmetic, verified bit-for-bit
// against the library in "reference recursions mirror the kernel"; every other number
// here is meaningless if that test fails, so it is checked first.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "phepex/phepex.hpp"

using Catch::Approx;

namespace {

// FlashCam-like configuration: 0.25 GHz readout (4 ns samples), 4x upsampling in the
// extractor, single-pole decay correction as used by FlashCamExtractor.
constexpr double SAMPLE_WIDTH_NS = 4.0;
constexpr double REF_SAMPLE_WIDTH_NS = 0.25;
constexpr int UPSAMPLING = 4;
constexpr float POLE_ZERO = 0.76f;

// Smoothing FWHM sweep, in UPSAMPLED samples (the unit the kernel uses). 3-4 is the
// operating region for FlashCam; 20 is far past it and included to expose the trend.
const std::vector<double> FWHM_SWEEP = {1.0, 2.0, 3.0, 4.0, 8.0, 20.0};

// Raw (pre-upsampling) trace lengths: nominal FlashCam readout window, and 3.2x / 6.4x
// longer to probe error accumulation with length.
const std::vector<int> LENGTH_SWEEP = {40, 128, 256};

constexpr float EPS_F32 = 1.1920929e-7f;  // 2^-23

// Noise gain of the two-pole recursion, 1/(1-r)^2 with r = exp(-b/sigma). Rounding
// injected at one sample is amplified by at most ~this much in the output, so it sets the
// scale every tolerance below is expressed in.
double noise_gain(double fwhm) {
    const double r = std::exp(-1.44605019 * 2.1 / fwhm);
    return 1.0 / ((1.0 - r) * (1.0 - r));
}

// ---------------------------------------------------------------------------
// Reference implementations of the smoothing recursion
// ---------------------------------------------------------------------------

template <typename Coef> struct DericheCoeffs {
    Coef n0, n1, m0, m1, d0, d1;
};

// Mirrors calculate_smoothing_coefficients() at the requested precision. The library
// evaluates this formula in double; passing Coef = long double gives the unquantised
// coefficients (the "true" filter), Coef = double what the library stores.
template <typename Coef> DericheCoeffs<Coef> reference_coefficients(double fwhm) {
    const Coef g1 = static_cast<Coef>(1.06003468);
    const Coef g2 = static_cast<Coef>(2.85668332);
    const Coef w = static_cast<Coef>(0.54103265);
    const Coef b = static_cast<Coef>(1.44605019);
    const Coef sigma = static_cast<Coef>(fwhm) / static_cast<Coef>(2.1);

    const Coef d0 = static_cast<Coef>(-2) * std::cos(w / sigma) * std::exp(-b / sigma);
    const Coef d1 = std::exp(static_cast<Coef>(-2) * b / sigma);
    const Coef n0 = g1;
    const Coef n1 =
        (-g1 * std::cos(w / sigma) + g2 * std::sin(w / sigma)) * std::exp(-b / sigma);
    const Coef m0 = n1 - d0 * n0;
    const Coef m1 = -d1 * n0;
    const Coef c = (static_cast<Coef>(1) + d0 + d1) / (n0 + n1 + m0 + m1);

    return {n0 * c, n1 * c, m0 * c, m1 * c, d0, d1};
}

// The library's coefficients as the kernel uses them: stored in double, narrowed to float
// once on entry to the recursion so the loop products stay float32. The narrowing is the
// coefficient-quantisation error source, and it is present whatever the struct holds.
DericheCoeffs<float> library_coefficients(double fwhm) {
    const auto c = phepex::calculate_smoothing_coefficients(fwhm);
    return {static_cast<float>(c.n[0]), static_cast<float>(c.n[1]),
            static_cast<float>(c.m[0]), static_cast<float>(c.m[1]),
            static_cast<float>(c.d[0]), static_cast<float>(c.d[1])};
}

// The kernel's recursion with the storage type of the forward pass (Fwd) and the state
// type of the backward pass (Bwd) made explicit. Products evaluate in the wider of the
// coefficient and operand types, exactly as in the kernel, so
//   <float,  float,  float>  with library coefficients == the current kernel, bit-for-bit
//   <float,  double, double> with double  coefficients == the kernel before the change
//   <long double, long double, long double> with long double coefficients == truth.
// The input is always float32: it is the output of the deconvolution stage, which is
// float32 by design (bit-exactness against ctapipe).
template <typename Fwd, typename Bwd, typename Coef>
void smooth_reference(const float *x, Fwd *y, int n, const DericheCoeffs<Coef> &c) {
    if (n == 0)
        return;

    y[0] = c.n0 * x[0];
    if (n == 1)
        return;

    y[1] = c.n0 * x[1] + c.n1 * x[0] - c.d0 * y[0];
    for (int i = 2; i < n; i++)
        y[i] = c.n0 * x[i] + c.n1 * x[i - 1] - c.d0 * y[i - 1] - c.d1 * y[i - 2];

    Bwd y_2 = 0;
    Bwd y_1 = 0;
    Bwd y_0 = c.m0 * x[n - 1];
    y[n - 2] += y_0;
    for (int i = n - 3; i >= 0; i--) {
        y_2 = y_1;
        y_1 = y_0;
        y_0 = c.m0 * x[i + 1] + c.m1 * x[i + 2] - c.d0 * y_1 - c.d1 * y_2;
        y[i] += y_0;
    }
}

// The three pipelines under comparison, each fed the same float32 deconvolved input.
// NARROW is what the library computes today; LEGACY is what it computed with double
// state and double coefficients; TRUTH is the same recursion evaluated in long double.
std::vector<float> smooth_narrow(const std::vector<float> &x, double fwhm) {
    std::vector<float> y(x.size());
    smooth_reference<float, float>(x.data(), y.data(), static_cast<int>(x.size()),
                                   library_coefficients(fwhm));
    return y;
}

std::vector<float> smooth_legacy(const std::vector<float> &x, double fwhm) {
    std::vector<float> y(x.size());
    smooth_reference<float, double>(x.data(), y.data(), static_cast<int>(x.size()),
                                    reference_coefficients<double>(fwhm));
    return y;
}

std::vector<long double> smooth_truth(const std::vector<float> &x, double fwhm) {
    std::vector<long double> y(x.size());
    smooth_reference<long double, long double>(x.data(), y.data(),
                                               static_cast<int>(x.size()),
                                               reference_coefficients<long double>(fwhm));
    return y;
}

// ---------------------------------------------------------------------------
// Waveform construction
// ---------------------------------------------------------------------------

// FlashCam-like reference pulse: a log-normal of ~4 ns FWHM on the 0.25 ns reference
// grid, i.e. wide compared to the 4 ns readout sampling. Amplitude is irrelevant --
// generate_waveforms normalises to the requested charge.
std::vector<double> flashcam_pulse() {
    std::vector<double> ref;
    for (double t = REF_SAMPLE_WIDTH_NS; t < 24.0; t += REF_SAMPLE_WIDTH_NS) {
        const double s = std::log(t / 5.0) / 0.45;
        ref.push_back(std::exp(-0.5 * s * s) / t);
    }
    return ref;
}

// One event of `n_pix` generated waveforms, row-major (n_pix, n_samples).
std::vector<float> generate(const std::vector<double> &charge,
                            const std::vector<double> &time_ns, int n_samples,
                            double nsb_rate_ghz, double electronic_noise,
                            std::uint64_t seed) {
    const auto ref = flashcam_pulse();
    const int n_pix = static_cast<int>(charge.size());
    std::vector<float> out(static_cast<std::size_t>(n_pix) * n_samples);
    phepex::generate_waveforms(charge.data(), time_ns.data(), 1, n_pix, ref.data(),
                               static_cast<int>(ref.size()), REF_SAMPLE_WIDTH_NS,
                               SAMPLE_WIDTH_NS, n_samples, 10, nsb_rate_ghz,
                               electronic_noise, seed, out.data());
    return out;
}

// The deconvolution stage alone (upsampling + pole-zero), i.e. exactly the float32 signal
// the smoothing pass receives. Obtained from the library with smoothing disabled so the
// tests never re-implement the stage that is not under study.
std::vector<float> deconvolved(const float *src, int n_samples, float baseline = 0.0f,
                               float scale = 1.0f) {
    std::vector<float> out(static_cast<std::size_t>(n_samples) * UPSAMPLING);
    phepex::preprocess_waveform(src, n_samples, UPSAMPLING, POLE_ZERO, nullptr, baseline,
                                scale, out.data());
    return out;
}

phepex::SampleRange valid_range(double fwhm, int n_samples) {
    const auto sc = phepex::calculate_smoothing_coefficients(fwhm);
    return phepex::preprocess_valid_range(UPSAMPLING, POLE_ZERO, &sc, n_samples);
}

// Largest |a - b| over [lo, hi).
template <typename A, typename B>
double max_abs_diff(const A &a, const B &b, const phepex::SampleRange &r) {
    double worst = 0.0;
    for (int i = r.lo; i < r.hi; i++)
        worst = std::max(
            worst, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    return worst;
}

template <typename A> double max_abs(const A &a, const phepex::SampleRange &r) {
    double peak = 0.0;
    for (int i = r.lo; i < r.hi; i++)
        peak = std::max(peak, std::fabs(static_cast<double>(a[i])));
    return peak;
}

}  // namespace

// ---------------------------------------------------------------------------

TEST_CASE("reference recursions mirror the kernel", "[smoothing][precision]") {
    // Guards every other test in this file: the NARROW reference must reproduce the
    // library bit-for-bit, otherwise the measured deltas describe the reference, not the
    // kernel. Compared through preprocess_waveform with upsampling == 1, pole_zero == 0,
    // unit scale and zero offset, which reduces the deconvolution stage to the identity
    // (out[i] = src[i]) and leaves the smoothing pass alone under test.
    std::mt19937 rng(20260813);
    std::uniform_real_distribution<float> amp(-500.0f, 3000.0f);

    for (double fwhm : FWHM_SWEEP) {
        for (int n : {8, 40, 512, 1024}) {
            std::vector<float> x(n);
            for (float &v : x)
                v = amp(rng);

            const auto sc = phepex::calculate_smoothing_coefficients(fwhm);
            std::vector<float> got(n);
            phepex::preprocess_waveform(x.data(), n, 1, 0.0f, &sc, 0.0f, 1.0f,
                                        got.data());

            const auto ref = smooth_narrow(x, fwhm);
            INFO("fwhm=" << fwhm << " n=" << n);
            REQUIRE(std::memcmp(got.data(), ref.data(), n * sizeof(float)) == 0);
        }
    }
}

TEST_CASE("coefficient quantisation: DC gain and impulse-response width",
          "[smoothing][precision]") {
    // Isolates error source 2. The float32 coefficients define a slightly different
    // filter; on a step and on an impulse there is no competing state noise, so the shift
    // in DC gain and in the effective FWHM is measured directly.
    for (double fwhm : FWHM_SWEEP) {
        const int n = 4096;  // >> the 1/(1-r) memory of the filter at every fwhm here
        const int mid = n / 2;

        SECTION("DC gain is unity to float32 precision, fwhm=" + std::to_string(fwhm)) {
            // The normalisation factor c in calculate_smoothing_coefficients enforces
            // sum(h) == 1; quantising the coefficients perturbs that sum.
            const float level = 1000.0f;
            std::vector<float> step(n, level);
            const auto narrow = smooth_narrow(step, fwhm);
            const auto truth = smooth_truth(step, fwhm);

            const double gain_narrow = narrow[mid] / level;
            const double gain_truth = static_cast<double>(truth[mid]) / level;
            INFO("gain narrow=" << gain_narrow << " truth=" << gain_truth);
            REQUIRE(gain_truth == Approx(1.0).margin(1e-12));
            // Tolerance: a few ulp of float32 amplified by the recursion's noise gain.
            REQUIRE(std::fabs(gain_narrow - 1.0) < 10.0 * EPS_F32 * noise_gain(fwhm));
        }

        SECTION("effective FWHM is unchanged, fwhm=" + std::to_string(fwhm)) {
            std::vector<float> impulse(n, 0.0f);
            impulse[mid] = 1000.0f;
            const auto narrow = smooth_narrow(impulse, fwhm);
            const auto truth = smooth_truth(impulse, fwhm);

            // Second moment of the impulse response, a continuous proxy for the width
            // (the half-maximum crossing is quantised to whole samples and too coarse to
            // resolve a coefficient shift at fwhm ~ 1).
            auto width = [&](const auto &h) {
                long double sum = 0.0L, moment = 0.0L;
                for (int i = 0; i < n; i++) {
                    const long double v = static_cast<long double>(h[i]);
                    sum += v;
                    moment += v * (i - mid) * (i - mid);
                }
                return std::sqrt(static_cast<double>(moment / sum));
            };
            const double w_narrow = width(narrow), w_truth = width(truth);
            INFO("rms width narrow=" << w_narrow << " truth=" << w_truth);
            REQUIRE(std::fabs(w_narrow - w_truth) / w_truth < 1e-4);
        }
    }
}

TEST_CASE("state precision: error does not grow with trace length",
          "[smoothing][precision]") {
    // Error source 1, swept over the axis the analytic model says should NOT matter. A
    // 6.4x longer trace must not carry a materially larger error: the filter's memory is
    // ~1/(1-r) samples, so rounding injected early is forgotten long before the end.
    for (double fwhm : FWHM_SWEEP) {
        std::vector<double> errors;
        for (int n_samples : LENGTH_SWEEP) {
            // A pulse train filling the whole trace: a longer trace then injects strictly
            // more rounding, instead of differing only in the length of its quiet tail.
            // generate_waveforms places one deposit per pixel, so the train is the sum of
            // one single-deposit waveform per pulse position.
            std::vector<float> raw(n_samples, 0.0f);
            for (int k = 0; k * 12 + 6 < n_samples; k++) {
                std::vector<double> charge = {500.0},
                                    time_ns = {(k * 12 + 6) * SAMPLE_WIDTH_NS};
                const auto pulse = generate(charge, time_ns, n_samples, 0.0, 0.0, 7);
                for (int i = 0; i < n_samples; i++)
                    raw[i] += pulse[i];
            }
            const auto x = deconvolved(raw.data(), n_samples);
            const auto r = valid_range(fwhm, n_samples);
            const auto truth = smooth_truth(x, fwhm);
            errors.push_back(max_abs_diff(smooth_narrow(x, fwhm), truth, r) /
                             max_abs(truth, r));
        }
        INFO("fwhm=" << fwhm << " rel errors by length: " << errors[0] << " " << errors[1]
                     << " " << errors[2]);
        // Absolute bound from the analytic model, at every length.
        for (double e : errors)
            REQUIRE(e < 20.0 * EPS_F32 * noise_gain(fwhm));
        // Length-independence: the 6.4x longer trace stays within 2x of the shortest
        // (a genuinely accumulating error would grow with sqrt(n) at least, i.e. 2.5x).
        REQUIRE(errors[2] < 2.0 * errors[0] + 1e-9);
    }
}

TEST_CASE("dynamic-range worst case: small late pulse after a saturating early pulse",
          "[smoothing][precision]") {
    // The backward pass carries the state of the large pulse into the region of the small
    // one, so this is where the relative error on the small pulse is maximal -- a case
    // uniform-amplitude random waveforms never produce.
    const int n_samples = 256;
    for (double fwhm : FWHM_SWEEP) {
        // 1e4 p.e. is at the top of the FlashCam dynamic range; 1 p.e. is the smallest
        // signal the extractor is expected to resolve.
        std::vector<double> charge = {1.0e4}, time_ns = {8 * SAMPLE_WIDTH_NS};
        auto raw = generate(charge, time_ns, n_samples, 0.0, 0.0, 11);
        std::vector<double> small_charge = {1.0}, small_time = {200 * SAMPLE_WIDTH_NS};
        const auto small = generate(small_charge, small_time, n_samples, 0.0, 0.0, 12);
        for (int i = 0; i < n_samples; i++)
            raw[i] += small[i];

        const auto x = deconvolved(raw.data(), n_samples);
        const auto narrow = smooth_narrow(x, fwhm);
        const auto truth = smooth_truth(x, fwhm);

        // Restrict to the late window holding the 1 p.e. pulse.
        phepex::SampleRange late{190 * UPSAMPLING, 210 * UPSAMPLING};
        const double err = max_abs_diff(narrow, truth, late);
        const double small_peak = max_abs(truth, late);
        const double large_peak = max_abs(truth, valid_range(fwhm, n_samples));

        INFO("fwhm=" << fwhm << " err=" << err << " small peak=" << small_peak
                     << " large peak=" << large_peak);
        // The error scales with the LARGE amplitude (it is the state being rounded), so
        // that is what it must be compared against; the requirement is that it stays a
        // negligible fraction of the SMALL pulse it sits on.
        REQUIRE(err < 20.0 * EPS_F32 * noise_gain(fwhm) * large_peak);
        REQUIRE(err < 1e-3 * small_peak);
    }
}

TEST_CASE("baseline-residual worst case: DC offset through the filter",
          "[smoothing][precision]") {
    // A few ADC counts of residual baseline survive subtraction in practice. The
    // deconvolution suppresses DC by (1 - pole_zero), but what is left enters the filter
    // as a constant term whose absolute rounding error is proportional to it.
    const int n_samples = 256;
    for (double fwhm : FWHM_SWEEP) {
        for (float residual : {-5.0f, 3.0f}) {
            std::vector<double> charge = {200.0}, time_ns = {40 * SAMPLE_WIDTH_NS};
            auto raw = generate(charge, time_ns, n_samples, 0.0, 0.0, 13);
            for (float &v : raw)
                v += residual;

            const auto x = deconvolved(raw.data(), n_samples);
            const auto narrow = smooth_narrow(x, fwhm);
            const auto truth = smooth_truth(x, fwhm);
            const auto r = valid_range(fwhm, n_samples);

            INFO("fwhm=" << fwhm << " residual=" << residual);
            REQUIRE(max_abs_diff(narrow, truth, r) <
                    20.0 * EPS_F32 * noise_gain(fwhm) * max_abs(truth, r));
        }
    }
}

namespace {

// One full-camera event, measured three ways: sample-level error against the long double
// reference, and the shift in the two extracted observables between the current (NARROW)
// and previous (LEGACY) arithmetic.
struct CameraMeasurement {
    std::vector<double> rel_error;    ///< per-pixel max |narrow - truth| / max|truth|
    double charge_delta_pe = 0.0;     ///< worst |q_narrow - q_legacy|, in p.e.
    double centroid_delta = 0.0;      ///< worst |c_narrow - c_legacy|, upsampled samples
    double peak_time_delta_ns = 0.0;  ///< worst |t_narrow - t_legacy|, ns
};

CameraMeasurement measure_camera(int n_pix, int n_samples, double fwhm,
                                 std::uint64_t seed) {
    const int n_up = n_samples * UPSAMPLING;

    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    std::lognormal_distribution<double> charge_dist(std::log(30.0), 1.6);
    std::uniform_real_distribution<double> time_dist(30.0 * SAMPLE_WIDTH_NS,
                                                     90.0 * SAMPLE_WIDTH_NS);
    std::vector<double> charge(n_pix), time_ns(n_pix);
    for (int p = 0; p < n_pix; p++) {
        charge[p] = std::min(charge_dist(rng), 1.0e4);  // clipped at the dynamic range
        time_ns[p] = time_dist(rng);
    }
    // 0.4 GHz NSB is a bright-moon FlashCam rate; 0.6 p.e./sample electronic noise.
    const auto raw = generate(charge, time_ns, n_samples, 0.4, 0.6, seed);

    const auto r = valid_range(fwhm, n_samples);
    std::vector<double> rel_errors;
    rel_errors.reserve(n_pix);
    std::vector<float> narrow_all(static_cast<std::size_t>(n_pix) * n_up);
    std::vector<float> legacy_all(static_cast<std::size_t>(n_pix) * n_up);

    for (int p = 0; p < n_pix; p++) {
        const auto x =
            deconvolved(&raw[static_cast<std::size_t>(p) * n_samples], n_samples);
        const auto narrow = smooth_narrow(x, fwhm);
        const auto legacy = smooth_legacy(x, fwhm);
        const auto truth = smooth_truth(x, fwhm);
        rel_errors.push_back(max_abs_diff(narrow, truth, r) / max_abs(truth, r));
        std::copy(narrow.begin(), narrow.end(),
                  narrow_all.begin() + static_cast<std::size_t>(p) * n_up);
        std::copy(legacy.begin(), legacy.end(),
                  legacy_all.begin() + static_cast<std::size_t>(p) * n_up);
    }

    // Downstream observables. NARROW vs LEGACY (both float32 waveforms through the same
    // library kernels) answers the operational question directly: does the change move
    // the physics relative to the shipped behaviour?
    std::vector<std::int64_t> peak(n_pix);
    for (int p = 0; p < n_pix; p++) {
        const float *w = &legacy_all[static_cast<std::size_t>(p) * n_up];
        peak[p] = std::max_element(w + r.lo, w + r.hi) - w;
    }
    std::vector<float> q_narrow(n_pix), q_legacy(n_pix), t_narrow(n_pix), t_legacy(n_pix);
    std::vector<float> c_narrow(n_pix), c_legacy(n_pix);
    const double rate_ghz = UPSAMPLING / SAMPLE_WIDTH_NS;
    phepex::extract_around_peak(narrow_all.data(), 1, n_pix, n_up, peak.data(), 8, 4,
                                rate_ghz, q_narrow.data(), t_narrow.data());
    phepex::extract_around_peak(legacy_all.data(), 1, n_pix, n_up, peak.data(), 8, 4,
                                rate_ghz, q_legacy.data(), t_legacy.data());
    phepex::adaptive_centroid(narrow_all.data(), 1, n_pix, n_up, peak.data(), 0.5,
                              c_narrow.data());
    phepex::adaptive_centroid(legacy_all.data(), 1, n_pix, n_up, peak.data(), 0.5,
                              c_legacy.data());

    // Charge is reported in the generator's p.e. units scaled by the deconvolution gain;
    // calibrate that gain per pixel from the extracted value so the delta below is a
    // delta in p.e. Pixels below 1 p.e. are skipped: they carry no charge information and
    // their gain estimate is dominated by NSB and noise.
    CameraMeasurement m;
    m.rel_error = std::move(rel_errors);
    for (int p = 0; p < n_pix; p++) {
        m.centroid_delta = std::max(
            m.centroid_delta, std::fabs(static_cast<double>(c_narrow[p]) - c_legacy[p]));
        m.peak_time_delta_ns =
            std::max(m.peak_time_delta_ns,
                     std::fabs(static_cast<double>(t_narrow[p]) - t_legacy[p]));
        const double gain = q_legacy[p] / charge[p];  // extracted units per p.e.
        if (charge[p] >= 1.0 && gain > 0.0)
            m.charge_delta_pe =
                std::max(m.charge_delta_pe, std::fabs(q_narrow[p] - q_legacy[p]) / gain);
    }
    return m;
}

}  // namespace

TEST_CASE("realistic camera load: error distribution and extracted charge/time",
          "[smoothing][precision]") {
    // Full-camera sample with NSB pile-up and electronic noise, at a trace length 3.2x
    // the nominal readout window. Three independent events are drawn so the reported tail
    // (max and 99.9th percentile over ~5300 waveforms) is not a single-seed artefact.
    const int n_pix = 1764;  // FlashCam
    const int n_samples = 128;
    const double fwhm = 3.0;  // FlashCam operating point

    std::vector<double> rel_errors;
    double charge_delta_pe = 0.0, centroid_delta = 0.0, peak_time_delta_ns = 0.0;
    for (std::uint64_t seed : {2026u, 2027u, 2028u}) {
        const auto m = measure_camera(n_pix, n_samples, fwhm, seed);
        rel_errors.insert(rel_errors.end(), m.rel_error.begin(), m.rel_error.end());
        charge_delta_pe = std::max(charge_delta_pe, m.charge_delta_pe);
        centroid_delta = std::max(centroid_delta, m.centroid_delta);
        peak_time_delta_ns = std::max(peak_time_delta_ns, m.peak_time_delta_ns);
    }

    std::sort(rel_errors.begin(), rel_errors.end());
    const double p999 =
        rel_errors[static_cast<std::size_t>(0.999 * (rel_errors.size() - 1))];
    INFO("relative sample error: p99.9=" << p999 << " max=" << rel_errors.back());
    REQUIRE(rel_errors.back() < 20.0 * EPS_F32 * noise_gain(fwhm));

    // The bounds that decide the change: the shift must be far below the single-p.e.
    // charge resolution and far below the timing resolution (~0.1 upsampled samples).
    INFO("worst charge delta=" << charge_delta_pe << " p.e., worst centroid delta="
                               << centroid_delta << " upsampled samples, worst peak time="
                               << peak_time_delta_ns << " ns");
    REQUIRE(charge_delta_pe < 1.0e-3);
    REQUIRE(centroid_delta < 1.0e-3);
    REQUIRE(peak_time_delta_ns < 1.0e-3);
}
