// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Max-Planck-Institut für Kernphysik
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

// Standalone C++ unit tests for the phepex kernels.
// Uses vendored Catch2 v3 (third_party/catch2). Each check is self-verifying:
// hand-computed values or algebraic invariants, so libphepex can be validated in a
// pure-C++ toolchain.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <random>
#include <utility>
#include <vector>

#include "catch_amalgamated.hpp"
#include "phepex/phepex.hpp"

using Catch::Approx;

TEST_CASE("preprocess_valid_range (no smoothing): hand-computed", "[preprocess]") {
    // With smoothing == nullptr this is the deconvolution edge range.
    // pole_zero != 0 : lo = 3U-2, hi = U*n - 2(U-1)
    auto r = phepex::preprocess_valid_range(4, 0.76f, nullptr, 22);
    REQUIRE(r.lo == 10);
    REQUIRE(r.hi == 82);
    // pole_zero == 0 : lo = 2(U-1)
    auto r0 = phepex::preprocess_valid_range(4, 0.0f, nullptr, 22);
    REQUIRE(r0.lo == 6);
    REQUIRE(r0.hi == 82);
}

TEST_CASE("preprocess_waveform (no smoothing): shape, DC gain, linearity",
          "[preprocess]") {
    const int n = 12, up = 4;
    const float c = 5.0f;
    std::vector<float> x(n, c);
    std::vector<float> out(static_cast<std::size_t>(n) * up);

    SECTION("DC input -> interior == c*(1-pole_zero)") {
        for (double pz : {0.0, 0.5, 0.75}) {
            const float pzf = static_cast<float>(pz);
            phepex::preprocess_waveform(x.data(), n, up, pzf, nullptr, 0.0f, 1.0f,
                                        out.data());
            auto vr = phepex::preprocess_valid_range(up, pzf, nullptr, n);
            const int mid = (vr.lo + vr.hi) / 2;
            REQUIRE(out[mid] == Approx(c * (1.0 - pz)).margin(1e-4));
            for (float v : out)
                REQUIRE(std::isfinite(v));
        }
    }
    SECTION("linearity: deconv(2x) == 2*deconv(x)") {
        std::vector<float> x2(n, 2 * c), o1(n * up), o2(n * up);
        const float pzf = 0.5f;
        phepex::preprocess_waveform(x.data(), n, up, pzf, nullptr, 0.0f, 1.0f, o1.data());
        phepex::preprocess_waveform(x2.data(), n, up, pzf, nullptr, 0.0f, 1.0f,
                                    o2.data());
        for (std::size_t i = 0; i < o1.size(); ++i)
            REQUIRE(o2[i] == Approx(2.0f * o1[i]).margin(1e-5));
    }
}

TEST_CASE("pos_soft_clip: max(y/(1+|y|), 0)", "[clip]") {
    const int n_up = 5;
    std::vector<float> x = {0.0f, 5.0f, -5.0f, 15.0f, 2.5f};  // scale = 5
    std::vector<float> out(n_up);

    SECTION("full range (0,0)") {
        phepex::pos_soft_clip(x.data(), 1, 1, n_up, 5.0f, 0, 0, out.data());
        REQUIRE(out[0] == Approx(0.0));
        REQUIRE(out[1] == Approx(0.5));      // y=1  -> 0.5
        REQUIRE(out[2] == Approx(0.0));      // y=-1 -> -0.5 -> clamp 0
        REQUIRE(out[3] == Approx(0.75));     // y=3  -> 0.75
        REQUIRE(out[4] == Approx(1.0 / 3));  // y=0.5 -> 1/3
    }
    SECTION("restricted [1,4): outside is zero") {
        phepex::pos_soft_clip(x.data(), 1, 1, n_up, 5.0f, 1, 4, out.data());
        REQUIRE(out[0] == 0.0f);
        REQUIRE(out[4] == 0.0f);
        REQUIRE(out[1] == Approx(0.5));
    }
}

TEST_CASE("neighbor_peak_indices: CSR line graph 0-1-2", "[neighbor]") {
    const int n_ch = 1, n_pix = 3, n_up = 5;
    std::vector<float> wf(n_pix * n_up, 0.0f);
    wf[0 * n_up + 2] = 9;  // pixel 0 peak @2
    wf[1 * n_up + 3] = 9;  // pixel 1 peak @3
    wf[2 * n_up + 4] = 9;  // pixel 2 peak @4
    std::vector<std::int32_t> indptr = {0, 1, 3, 4}, indices = {1, 0, 2, 1};
    std::vector<std::uint8_t> broken(n_pix, 0);  // byte mask: nonzero => broken
    std::vector<std::int64_t> peak(n_pix);
    const std::uint8_t *bp = broken.data();

    SECTION("local_weight 0") {
        phepex::neighbor_peak_indices(wf.data(), n_ch, n_pix, n_up, indptr.data(),
                                      indices.data(), 0, bp, 0, 0, peak.data());
        REQUIRE(peak[0] == 3);  // nb{1}: wf1 peak @3
        REQUIRE(peak[1] == 2);  // nb{0,2}: [0,0,9,0,9] first max @2
        REQUIRE(peak[2] == 3);
    }
    SECTION("local_weight 1 includes self") {
        phepex::neighbor_peak_indices(wf.data(), n_ch, n_pix, n_up, indptr.data(),
                                      indices.data(), 1, bp, 0, 0, peak.data());
        REQUIRE(peak[0] == 2);  // self@2 + nb1@3 -> [0,0,9,9,0] first max @2
        REQUIRE(peak[1] == 2);  // [0,0,9,9,9] first max @2
        REQUIRE(peak[2] == 3);
    }
    SECTION("broken neighbour skipped") {
        broken[0] = 1;  // pixel 0 broken -> excluded as a neighbour
        phepex::neighbor_peak_indices(wf.data(), n_ch, n_pix, n_up, indptr.data(),
                                      indices.data(), 0, bp, 0, 0, peak.data());
        REQUIRE(peak[1] == 4);  // only nb2 -> peak @4
    }
}

TEST_CASE("extract_around_peak: window sum + weighted time", "[extract]") {
    const int n_ch = 1, n_pix = 1, n_up = 8;
    std::vector<float> w = {0, 0, 1, 4, 2, 0, 0, 0};
    std::vector<std::int64_t> pk = {3};
    std::vector<float> charge(1), ptime(1);
    // width=3, shift=1 -> window [2,5): 1+4+2 = 7; time = (1*2+4*3+2*4)/7 = 22/7,
    // /rate(1)
    phepex::extract_around_peak(w.data(), n_ch, n_pix, n_up, pk.data(), 3, 1, 1.0,
                                charge.data(), ptime.data());
    REQUIRE(charge[0] == Approx(7.0));
    REQUIRE(ptime[0] == Approx(22.0 / 7.0));
    // sampling_rate scales the time
    phepex::extract_around_peak(w.data(), n_ch, n_pix, n_up, pk.data(), 3, 1, 2.0,
                                charge.data(), ptime.data());
    REQUIRE(ptime[0] == Approx(22.0 / 7.0 / 2.0));
}

TEST_CASE("adaptive_centroid: leading-edge centroid + fallback", "[extract]") {
    const int n_ch = 1, n_pix = 1, n_up = 8;
    std::vector<float> w = {0, 0, 1, 4, 2, 0, 0, 0};
    std::vector<std::int64_t> pk = {3};
    std::vector<float> out(1);
    // rel=0.4 -> descend limit 0.4*4=1.6; window {3,4}: (3*4+4*2)/(4+2) = 20/6
    phepex::adaptive_centroid(w.data(), n_ch, n_pix, n_up, pk.data(), 0.4, out.data());
    REQUIRE(out[0] == Approx(20.0 / 6.0));

    SECTION("negative peak amplitude -> returns peak_index") {
        std::vector<float> wn(n_up, -1.0f);
        phepex::adaptive_centroid(wn.data(), n_ch, n_pix, n_up, pk.data(), 0.4,
                                  out.data());
        REQUIRE(out[0] == Approx(3.0));
    }
}

namespace {
std::vector<double> gaussian_pulse() {  // ref_sample_width = 0.25 ns, ~10 ns long
    std::vector<double> ref;
    for (double t = 0.0; t < 10.0; t += 0.25)
        ref.push_back(std::exp(-0.5 * std::pow((t - 4.0) / 0.8, 2)));
    return ref;
}
}  // namespace

TEST_CASE("generate_waveforms: charge conservation, zero, reproducibility, NSB rate",
          "[generate]") {
    const auto ref = gaussian_pulse();
    const double ref_sw = 0.25, sw = 1.0;
    const int up = 4, n = 32;

    SECTION("signal-only integral == charge (charge-conserving), zero->zero") {
        std::vector<double> ch = {100.0}, tm = {static_cast<double>(n / 2) * sw};
        std::vector<float> out(n);
        phepex::generate_waveforms(ch.data(), tm.data(), 1, 1, ref.data(),
                                   (int)ref.size(), ref_sw, sw, n, up, 0.0, 0.0, 0,
                                   out.data());
        double integ = std::accumulate(out.begin(), out.end(), 0.0);
        REQUIRE(integ == Approx(100.0).margin(0.5));
        for (float v : out)
            REQUIRE(std::isfinite(v));

        std::vector<double> z0 = {0.0}, zt = {0.0};
        std::vector<float> zout(n);
        phepex::generate_waveforms(z0.data(), zt.data(), 1, 1, ref.data(),
                                   (int)ref.size(), ref_sw, sw, n, up, 0.0, 0.0, 0,
                                   zout.data());
        for (float v : zout)
            REQUIRE(v == 0.0f);
    }

    SECTION("same seed -> identical output") {
        const int ne = 5, np = 40;
        std::vector<double> ch(ne * np, 0.0), tm(ne * np, 0.0);
        std::vector<float> a(ne * np * n), b(ne * np * n);
        phepex::generate_waveforms(ch.data(), tm.data(), ne, np, ref.data(),
                                   (int)ref.size(), ref_sw, sw, n, up, 0.5, 0.0, 42,
                                   a.data());
        phepex::generate_waveforms(ch.data(), tm.data(), ne, np, ref.data(),
                                   (int)ref.size(), ref_sw, sw, n, up, 0.5, 0.0, 42,
                                   b.data());
        REQUIRE(a == b);
    }

    SECTION("NSB mean integral/pixel ~ rate * n_samples * sample_width") {
        const int ne = 200, np = 200;
        std::vector<double> ch(ne * np, 0.0), tm(ne * np, 0.0);
        std::vector<float> out(static_cast<std::size_t>(ne) * np * n);
        const double rate = 0.5;
        phepex::generate_waveforms(ch.data(), tm.data(), ne, np, ref.data(),
                                   (int)ref.size(), ref_sw, sw, n, up, rate, 0.0, 1,
                                   out.data());
        double tot = 0.0;
        for (float v : out)
            tot += v;
        const double mean_per_pixel = tot / (static_cast<double>(ne) * np);
        REQUIRE(mean_per_pixel == Approx(rate * n * sw).epsilon(0.02));
    }

    SECTION("electronic noise: zero-mean, sigma reproduced, additive on the signal") {
        const int ne = 50, np = 200;
        const double sigma = 0.3;
        std::vector<double> ch(static_cast<std::size_t>(ne) * np, 0.0),
            tm(static_cast<std::size_t>(ne) * np, 0.0);
        std::vector<float> out(static_cast<std::size_t>(ne) * np * n);
        phepex::generate_waveforms(ch.data(), tm.data(), ne, np, ref.data(),
                                   (int)ref.size(), ref_sw, sw, n, up, 0.0, sigma, 7,
                                   out.data());
        double sum = 0.0, sq = 0.0;
        for (float v : out) {
            sum += v;
            sq += static_cast<double>(v) * v;
        }
        const double count = static_cast<double>(out.size());
        const double mean = sum / count;
        // 8e5 samples of sigma 0.3 => standard error 3.4e-4, so 5e-3 is ~15 sigma.
        REQUIRE(std::fabs(mean) < 5e-3);
        REQUIRE(std::sqrt(sq / count - mean * mean) == Approx(sigma).epsilon(0.02));

        // Noise is drawn after the signal deposit, so it only displaces the samples: the
        // difference against the noiseless output has the noise statistics, and the
        // signal integral survives.
        std::vector<double> sch(np, 100.0), stm(np, static_cast<double>(n / 2) * sw);
        std::vector<float> quiet(np * n), noisy(np * n);
        phepex::generate_waveforms(sch.data(), stm.data(), 1, np, ref.data(),
                                   (int)ref.size(), ref_sw, sw, n, up, 0.0, 0.0, 7,
                                   quiet.data());
        phepex::generate_waveforms(sch.data(), stm.data(), 1, np, ref.data(),
                                   (int)ref.size(), ref_sw, sw, n, up, 0.0, sigma, 7,
                                   noisy.data());
        double dsq = 0.0, quiet_integ = 0.0, noisy_integ = 0.0;
        for (std::size_t i = 0; i < quiet.size(); i++) {
            const double d = static_cast<double>(noisy[i]) - quiet[i];
            dsq += d * d;
            quiet_integ += quiet[i];
            noisy_integ += noisy[i];
        }
        REQUIRE(std::sqrt(dsq / static_cast<double>(quiet.size())) ==
                Approx(sigma).epsilon(0.05));
        REQUIRE(noisy_integ == Approx(quiet_integ).epsilon(0.01));
    }

    SECTION("electronic noise: reproducible per seed, and rejected when negative") {
        const int np = 32;
        std::vector<double> ch(np, 0.0), tm(np, 0.0);
        std::vector<float> a(np * n), b(np * n), c(np * n);
        phepex::generate_waveforms(ch.data(), tm.data(), 1, np, ref.data(),
                                   (int)ref.size(), ref_sw, sw, n, up, 0.0, 0.5, 3,
                                   a.data());
        phepex::generate_waveforms(ch.data(), tm.data(), 1, np, ref.data(),
                                   (int)ref.size(), ref_sw, sw, n, up, 0.0, 0.5, 3,
                                   b.data());
        phepex::generate_waveforms(ch.data(), tm.data(), 1, np, ref.data(),
                                   (int)ref.size(), ref_sw, sw, n, up, 0.0, 0.5, 4,
                                   c.data());
        REQUIRE(a == b);
        REQUIRE(a != c);

        REQUIRE_THROWS_AS(phepex::generate_waveforms(ch.data(), tm.data(), 1, np,
                                                     ref.data(), (int)ref.size(), ref_sw,
                                                     sw, n, up, 0.0, -1.0, 0, a.data()),
                          std::invalid_argument);
    }
}

namespace {
// Square grid of pixel centres covering [-lim, lim]^2 at `pitch` spacing. A hexagonal
// layout would only change which points the pdf is sampled at, which none of the checks
// below depend on.
struct PixelGrid {
    std::vector<double> x, y;
    double area;
    PixelGrid(double pitch, double lim) : area(pitch * pitch) {
        const int n = static_cast<int>(std::lround(lim / pitch));
        for (int i = -n; i <= n; ++i)
            for (int j = -n; j <= n; ++j) {
                x.push_back(i * pitch);
                y.push_back(j * pitch);
            }
    }
    int size() const { return static_cast<int>(x.size()); }
};

phepex::ShowerModel test_shower() {
    phepex::ShowerModel m;
    m.length_m = 0.20;
    m.width_m = 0.07;
    m.psi_rad = 0.3;
    m.skewness = 0.3;
    m.intensity_pe = 20000.0;
    m.time_gradient_ns_per_m = 20.0;
    m.time_intercept_ns = 44.0;
    return m;
}
}  // namespace

TEST_CASE("generate_shower_image: intensity, time model, reproducibility", "[generate]") {
    const PixelGrid grid(0.02, 1.2);  // covers the image out to many sigma
    const auto model = test_shower();
    std::vector<double> q(grid.size()), t(grid.size());

    SECTION("charge sums to the model intensity, and vanishes far from the image") {
        phepex::generate_shower_image(model, grid.x.data(), grid.y.data(), grid.size(),
                                      grid.area, 1, q.data(), t.data());
        const double tot = std::accumulate(q.begin(), q.end(), 0.0);
        // Poisson on the total: sigma = sqrt(intensity); allow 5 sigma.
        REQUIRE(tot ==
                Approx(model.intensity_pe).margin(5.0 * std::sqrt(model.intensity_pe)));
        // The pdf is > 15 sigma out at the grid corner in both axes.
        for (int p = 0; p < grid.size(); ++p)
            if (std::hypot(grid.x[p], grid.y[p]) > 1.5)
                REQUIRE(q[p] == 0.0);
    }

    SECTION("time is linear in the longitudinal coordinate; jitter is bounded") {
        auto m = model;
        m.psi_rad = 0.0;  // longitudinal coordinate == x - centroid_x
        m.centroid_x_m = 0.13;
        phepex::generate_shower_image(m, grid.x.data(), grid.y.data(), grid.size(),
                                      grid.area, 1, q.data(), t.data());
        for (int p = 0; p < grid.size(); ++p) {
            const double expected =
                (grid.x[p] - m.centroid_x_m) * m.time_gradient_ns_per_m +
                m.time_intercept_ns;
            REQUIRE(t[p] == Approx(expected));
        }

        m.time_jitter_ns = 0.5;
        std::vector<double> tj(grid.size());
        phepex::generate_shower_image(m, grid.x.data(), grid.y.data(), grid.size(),
                                      grid.area, 1, q.data(), tj.data());
        double max_dev = 0.0;
        for (int p = 0; p < grid.size(); ++p)
            max_dev = std::max(max_dev, std::fabs(tj[p] - t[p]));
        REQUIRE(max_dev <= m.time_jitter_ns);
        // With thousands of pixels the jitter must actually populate its range.
        REQUIRE(max_dev > 0.9 * m.time_jitter_ns);
    }

    SECTION("photon_statistics == false: rounded expectation, seed-independent") {
        PixelGrid grid(0.05, 1.0);
        auto model = test_shower();
        model.photon_statistics = false;
        std::vector<double> q1(grid.size()), t1(grid.size()), q2(grid.size()),
            t2(grid.size());
        phepex::generate_shower_image(model, grid.x.data(), grid.y.data(), grid.size(),
                                      grid.area, 1, q1.data(), t1.data());
        phepex::generate_shower_image(model, grid.x.data(), grid.y.data(), grid.size(),
                                      grid.area, 999, q2.data(), t2.data());
        REQUIRE(q1 == q2);  // consumes no random numbers for the charge
        for (double q : q1)
            REQUIRE(q == std::round(q));
        // Rounding sheds the sub-0.5 tail, so the total falls short of intensity_pe; it
        // must still recover the bulk of it.
        const double total = std::accumulate(q1.begin(), q1.end(), 0.0);
        REQUIRE(total > 0.7 * model.intensity_pe);
        REQUIRE(total <= model.intensity_pe);
    }

    SECTION("same seed -> identical image, different seed -> different charges") {
        auto m = model;
        m.time_jitter_ns = 0.5;
        std::vector<double> q2(grid.size()), t2(grid.size());
        phepex::generate_shower_image(m, grid.x.data(), grid.y.data(), grid.size(),
                                      grid.area, 7, q.data(), t.data());
        phepex::generate_shower_image(m, grid.x.data(), grid.y.data(), grid.size(),
                                      grid.area, 7, q2.data(), t2.data());
        REQUIRE(q == q2);
        REQUIRE(t == t2);
        phepex::generate_shower_image(m, grid.x.data(), grid.y.data(), grid.size(),
                                      grid.area, 8, q2.data(), t2.data());
        REQUIRE(q != q2);
    }

    SECTION("invalid model parameters throw instead of yielding an empty image") {
        auto call = [&](const phepex::ShowerModel &m, double area) {
            phepex::generate_shower_image(m, grid.x.data(), grid.y.data(), grid.size(),
                                          area, 1, q.data(), t.data());
        };
        auto m = model;
        m.width_m = 0.0;
        REQUIRE_THROWS_AS(call(m, grid.area), std::invalid_argument);
        m = model;
        m.length_m = 0.0;
        REQUIRE_THROWS_AS(call(m, grid.area), std::invalid_argument);
        m = model;
        m.intensity_pe = -1.0;
        REQUIRE_THROWS_AS(call(m, grid.area), std::invalid_argument);
        REQUIRE_THROWS_AS(call(model, 0.0), std::invalid_argument);

        // Beyond the skew-normal's attainable |skewness| (~0.995271746) the shape
        // parameter diverges; just below it the image must still be well-formed.
        m = model;
        m.skewness = 0.9953;
        REQUIRE_THROWS_AS(call(m, grid.area), std::invalid_argument);
        m.skewness = 2.0;
        REQUIRE_THROWS_AS(call(m, grid.area), std::invalid_argument);
        m.skewness = 0.9952;
        REQUIRE_NOTHROW(call(m, grid.area));
        REQUIRE(std::accumulate(q.begin(), q.end(), 0.0) ==
                Approx(model.intensity_pe).margin(5.0 * std::sqrt(model.intensity_pe)));
    }
}

// preprocess_waveform / preprocess_valid_range: bit-exact against a FROZEN copy of
// libdvr's DSP kernels (the oracle), so that any divergence from libdvr's math fails
// here. The reference code below is a verbatim copy of libdvr's src/DSP.cpp and must NOT
// be "cleaned up", with ONE deliberate exception: refDeconvolveUnit applies the pole-zero
// correction at upsampling == 1, where libdvr's DSP.cpp computes scale*(src-offset)
// instead. phepex applies the correction there, so the oracle is patched identically to
// keep the bit-exact comparison meaningful; for upsampling > 1 it is unmodified.
namespace {

template <typename InputType, typename OutputType>
void refUpsample(int n, int upsampling_factor, const InputType *input, OutputType offset,
                 OutputType pole_zero, OutputType *output, OutputType scale) {
    OutputType v2, v1;
    OutputType sum1, sum2;
    OutputType tmp;
    OutputType pzc2, pzc1;
    OutputType *out1 = output;
    OutputType *out2 = output;
    OutputType mult = scale / (upsampling_factor * upsampling_factor);
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

template <typename InputType, typename OutputType>
void refSmooth(const InputType *x, OutputType *y, int n,
               const phepex::SmoothingCoefficients &c) {
    if (n == 0 || x == nullptr || y == nullptr)
        return;
    y[0] = c.n[0] * x[0];
    if (n == 1)
        return;
    y[1] = c.n[0] * x[1] + c.n[1] * x[0] - c.d[0] * y[0];
    for (int i = 2; i < n; i++)
        y[i] = c.n[0] * x[i] + c.n[1] * x[i - 1] - c.d[0] * y[i - 1] - c.d[1] * y[i - 2];
    double y_2 = 0, y_1 = 0, y_0 = c.m[0] * x[n - 1];
    y[n - 2] += y_0;
    for (int i = n - 3; i >= 0; i--) {
        y_2 = y_1;
        y_1 = y_0;
        y_0 = c.m[0] * x[i + 1] + c.m[1] * x[i + 2] - c.d[0] * y_1 - c.d[1] * y_2;
        y[i] += y_0;
    }
}

// Unity-upsampling pole-zero deconvolution. Must match preprocess.cpp's
// deconvolve_unit_upsampling operation-for-operation so the bit-exact comparison holds.
void refDeconvolveUnit(const std::uint16_t *src, int n, float pole_zero, float offset,
                       float scale, float *dst) {
    if (n <= 0)
        return;
    float prev = static_cast<float>(src[0]) - offset;
    dst[0] = scale * prev;
    for (int i = 1; i < n; i++) {
        const float cur = static_cast<float>(src[i]) - offset;
        dst[i] = scale * (cur - pole_zero * prev);
        prev = cur;
    }
}

void refPreprocess(const std::uint16_t *src, int n, int up, float pole_zero,
                   const phepex::SmoothingCoefficients *sm, float offset, float scale,
                   float *out) {
    const int dst_samples = up * n;
    std::vector<float> tmp;
    if (sm) {
        tmp.resize(dst_samples);
        if (up == 1)
            refDeconvolveUnit(src, n, pole_zero, offset, scale, tmp.data());
        else
            refUpsample<std::uint16_t, float>(n, up, src, offset, pole_zero, tmp.data(),
                                              scale);
        refSmooth(tmp.data(), out, dst_samples, *sm);
    } else {
        if (up == 1)
            refDeconvolveUnit(src, n, pole_zero, offset, scale, out);
        else
            refUpsample<std::uint16_t, float>(n, up, src, offset, pole_zero, out, scale);
    }
}

// Expected preprocess_valid_range. The margins are in UPSAMPLED samples, so the range
// indexes the up*num_samples output.
std::pair<int, int> refValidRange(int up, float pole_zero,
                                  const phepex::SmoothingCoefficients *sm,
                                  int num_samples) {
    if (up < 1)
        up = 1;
    int right = 2 * up - 2;
    int left = std::max(right, (pole_zero != 0.0f) ? (3 * up - 2) : 0);
    if (sm) {
        const int fwhm = static_cast<int>(std::floor(sm->fwhm));
        right += fwhm;
        left += fwhm;
    }
    const int n_up = up * num_samples;
    if (left >= n_up - right)
        return {0, 0};
    return {left, n_up - right};
}

}  // namespace

TEST_CASE("preprocess_waveform matches the frozen libdvr oracle bit-for-bit",
          "[preprocess]") {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> adc(0, 4095);
    const int n = 40;
    const float offset = 3.5f, scale = 0.7f;

    for (int up : {1, 2, 4, 8}) {
        for (float pz : {0.0f, 0.9f}) {
            for (double fwhm : {-1.0, 1.5, 4.0}) {  // -1 => no smoothing
                phepex::SmoothingCoefficients sc;
                const phepex::SmoothingCoefficients *sm = nullptr;
                if (fwhm > 0.0) {
                    sc = phepex::calculate_smoothing_coefficients(fwhm);
                    sm = &sc;
                }
                std::vector<std::uint16_t> src(n);
                for (int i = 0; i < n; i++)
                    src[i] = static_cast<std::uint16_t>(adc(rng));

                std::vector<float> got(n * up), ref(n * up);
                phepex::preprocess_waveform(src.data(), n, up, pz, sm, offset, scale,
                                            got.data());
                refPreprocess(src.data(), n, up, pz, sm, offset, scale, ref.data());
                REQUIRE(std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) ==
                        0);
            }
        }
    }
}

TEST_CASE("preprocess_waveform float and uint16 overloads agree", "[preprocess]") {
    const int n = 24, up = 4;
    std::vector<std::uint16_t> src = {10, 12, 40, 90, 60, 30, 20, 15, 11, 10, 10, 10,
                                      10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
    std::vector<float> srcf(src.begin(), src.end());
    std::vector<float> a(n * up), b(n * up);
    phepex::preprocess_waveform(src.data(), n, up, 0.9f, nullptr, 2.0f, 0.5f, a.data());
    phepex::preprocess_waveform(srcf.data(), n, up, 0.9f, nullptr, 2.0f, 0.5f, b.data());
    REQUIRE(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

TEST_CASE("preprocess_valid_range trims the upsampled output", "[preprocess]") {
    for (int up : {1, 2, 4, 8}) {
        for (float pz : {0.0f, 0.9f}) {
            for (double fwhm : {-1.0, 1.5, 4.0}) {
                for (int ns : {3, 8, 40}) {
                    phepex::SmoothingCoefficients sc;
                    const phepex::SmoothingCoefficients *sm = nullptr;
                    if (fwhm > 0.0) {
                        sc = phepex::calculate_smoothing_coefficients(fwhm);
                        sm = &sc;
                    }
                    auto r = phepex::preprocess_valid_range(up, pz, sm, ns);
                    auto e = refValidRange(up, pz, sm, ns);
                    REQUIRE(r.lo == e.first);
                    REQUIRE(r.hi == e.second);
                }
            }
        }
    }
}

TEST_CASE("calculate_smoothing_coefficients has unity DC gain", "[preprocess]") {
    // A constant input must pass through the smoothing unchanged deep in the interior
    // (n large enough that the IIR transient has died out at n/2).
    const int n = 64;
    std::vector<std::uint16_t> src(n, 100);
    for (double fwhm : {1.5, 4.0, 8.0}) {
        auto sc = phepex::calculate_smoothing_coefficients(fwhm);
        std::vector<float> out(n);
        phepex::preprocess_waveform(src.data(), n, 1, 0.0f, &sc, 0.0f, 1.0f, out.data());
        REQUIRE(out[n / 2] == Approx(100.0).margin(1e-2));
    }
}

TEST_CASE("neighbor_peak_indices: neighbor_count + skip-broken (neighbours-only)",
          "[neighbor]") {
    // Line graph 0-1-2-3; local_weight 0 => neighbours only.
    const int n_ch = 1, n_pix = 4, n_up = 5;
    std::vector<float> wf(n_pix * n_up, 0.0f);
    wf[0 * n_up + 1] = 9;  // pixel 0 peak @1
    wf[1 * n_up + 2] = 9;  // pixel 1 peak @2
    wf[2 * n_up + 3] = 9;  // pixel 2 peak @3
    wf[3 * n_up + 4] = 9;  // pixel 3 peak @4
    // CSR line graph: 0-1, 1-{0,2}, 2-{1,3}, 3-2
    std::vector<std::int32_t> indptr = {0, 1, 3, 5, 6};
    std::vector<std::int32_t> indices = {1, 0, 2, 1, 3, 2};
    std::vector<std::uint8_t> broken(n_pix, 0);  // byte mask: nonzero => broken
    std::vector<std::int64_t> peak(n_pix);
    std::vector<std::int32_t> count(n_pix, -1);
    const std::uint8_t *bp = broken.data();

    SECTION("counts and neighbours-only peaks") {
        phepex::neighbor_peak_indices(wf.data(), n_ch, n_pix, n_up, indptr.data(),
                                      indices.data(), 0, bp, 0, 0, peak.data(),
                                      count.data());
        REQUIRE(count[0] == 1);  // nb {1}
        REQUIRE(count[1] == 2);  // nb {0,2}
        REQUIRE(count[2] == 2);  // nb {1,3}
        REQUIRE(count[3] == 1);  // nb {2}
        REQUIRE(peak[0] == 2);   // nb1 peak @2
        REQUIRE(peak[1] == 1);   // nb0@1 + nb2@3 -> first max @1
    }
    SECTION("broken neighbour is skipped (not counted), later valid ones still summed") {
        broken[0] = 1;  // pixel 0 broken; pixel 1's neighbour list is {0,2}
        phepex::neighbor_peak_indices(wf.data(), n_ch, n_pix, n_up, indptr.data(),
                                      indices.data(), 0, bp, 0, 0, peak.data(),
                                      count.data());
        REQUIRE(count[1] == 1);  // only nb2 counted (nb0 skipped, NOT broken off early)
        REQUIRE(peak[1] == 3);   // nb2 peak @3
    }
    SECTION("null neighbor_count is allowed") {
        phepex::neighbor_peak_indices(wf.data(), n_ch, n_pix, n_up, indptr.data(),
                                      indices.data(), 0, bp, 0, 0, peak.data(), nullptr);
        REQUIRE(peak[2] == 2);  // nb1@2 + nb3@4 -> first max @2
    }
    SECTION("external scratch buffer matches internal allocation") {
        std::vector<std::int64_t> peak_ext(n_pix);
        std::vector<float> scratch(n_up);  // n_up floats always suffices
        for (auto range : {std::make_pair(0, 0), std::make_pair(1, 4)}) {
            phepex::neighbor_peak_indices(wf.data(), n_ch, n_pix, n_up, indptr.data(),
                                          indices.data(), 1, bp, range.first,
                                          range.second, peak.data());
            phepex::neighbor_peak_indices(
                wf.data(), n_ch, n_pix, n_up, indptr.data(), indices.data(), 1, bp,
                range.first, range.second, peak_ext.data(), nullptr, scratch.data());
            REQUIRE(std::memcmp(peak.data(), peak_ext.data(),
                                peak.size() * sizeof(std::int64_t)) == 0);
        }
    }
}

TEST_CASE("preprocess_waveform: external scratch buffer matches internal allocation",
          "[preprocess]") {
    const int n = 24, up = 4;
    std::vector<std::uint16_t> src = {10, 12, 40, 90, 60, 30, 20, 15, 11, 10, 10, 10,
                                      10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
    auto sc =
        phepex::calculate_smoothing_coefficients(4.0);  // smoothing path uses scratch
    std::vector<float> got(n * up), ref(n * up);
    std::vector<float> scratch(n * up);  // n_samples*upsampling floats
    phepex::preprocess_waveform(src.data(), n, up, 0.9f, &sc, 2.0f, 0.5f, ref.data());
    phepex::preprocess_waveform(src.data(), n, up, 0.9f, &sc, 2.0f, 0.5f, got.data(),
                                scratch.data());
    REQUIRE(std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) == 0);
}

TEST_CASE("preprocess_waveforms: uint16 input matches the per-row scalar kernel",
          "[preprocess]") {
    // 40 rows spans more than one internal tile plus a remainder for any tile width, so
    // both the tiled body and the scalar remainder loop are covered. Per-row (stride 1)
    // coefficients exercise the per-lane parameter gather.
    const int n_rows = 40, n = 24;
    std::mt19937 rng(4242);
    std::uniform_int_distribution<int> adc(0, 4095);
    std::vector<std::uint16_t> src(static_cast<std::size_t>(n_rows) * n);
    for (auto &v : src)
        v = static_cast<std::uint16_t>(adc(rng));
    std::vector<float> srcf(src.begin(), src.end());

    std::vector<float> pz(n_rows), offset(n_rows), scale(n_rows);
    for (int r = 0; r < n_rows; r++) {
        pz[r] = 0.5f + 0.01f * r;
        offset[r] = 2.0f + 0.5f * r;
        scale[r] = 0.25f + 0.05f * r;
    }

    const auto sc = phepex::calculate_smoothing_coefficients(4.0);
    struct Cfg {
        int up;
        const phepex::SmoothingCoefficients *sm;
    };
    // All four branches: tiled upsampling+smoothing, tiled upsampling, tiled smoothing,
    // and the per-row scalar path (up == 1, no smoothing).
    for (const Cfg cfg : {Cfg{4, &sc}, Cfg{4, nullptr}, Cfg{1, &sc}, Cfg{1, nullptr}}) {
        const std::size_t n_up = static_cast<std::size_t>(cfg.up) * n;
        std::vector<float> ref(static_cast<std::size_t>(n_rows) * n_up);
        std::vector<float> got(static_cast<std::size_t>(n_rows) * n_up);
        std::vector<float> from_float(static_cast<std::size_t>(n_rows) * n_up);

        for (int r = 0; r < n_rows; r++)
            phepex::preprocess_waveform(src.data() + static_cast<std::size_t>(r) * n, n,
                                        cfg.up, pz[r], cfg.sm, offset[r], scale[r],
                                        ref.data() + r * n_up);
        phepex::preprocess_waveforms(src.data(), n_rows, n, cfg.up, pz.data(), 1, cfg.sm,
                                     offset.data(), 1, scale.data(), 1, got.data());
        REQUIRE(std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) == 0);

        // uint16 widens exactly, so the float overload on the converted input must agree
        // bit-for-bit as well.
        phepex::preprocess_waveforms(srcf.data(), n_rows, n, cfg.up, pz.data(), 1, cfg.sm,
                                     offset.data(), 1, scale.data(), 1,
                                     from_float.data());
        REQUIRE(std::memcmp(from_float.data(), ref.data(), got.size() * sizeof(float)) ==
                0);
    }
}

TEST_CASE("preprocess_waveforms: external scratch buffer matches internal allocation",
          "[preprocess]") {
    // 40 rows spans more than one internal tile plus a remainder for any tile width.
    const int n_rows = 40, n = 24;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(0.0f, 4095.0f);
    std::vector<float> src(static_cast<std::size_t>(n_rows) * n);
    for (auto &v : src)
        v = d(rng);

    const float pz = 0.9f, offset = 2.0f, scale = 0.5f;
    const auto sc = phepex::calculate_smoothing_coefficients(4.0);

    // Each tiled branch: upsampling+smoothing, upsampling-only, smoothing-only. Broadcast
    // (stride 0) coefficients keep the focus on scratch vs internal-allocation
    // equivalence.
    struct Cfg {
        int up;
        const phepex::SmoothingCoefficients *sm;
    };
    for (const Cfg cfg : {Cfg{4, &sc}, Cfg{4, nullptr}, Cfg{1, &sc}}) {
        const std::size_t n_up = static_cast<std::size_t>(cfg.up) * n;
        const std::size_t need =
            phepex::preprocess_waveforms_scratch_size(n, cfg.up, cfg.sm);
        REQUIRE(need > 0);  // every tiled branch needs a scratch buffer
        std::vector<float> ref(static_cast<std::size_t>(n_rows) * n_up);
        std::vector<float> got(static_cast<std::size_t>(n_rows) * n_up);
        std::vector<float> scratch(need);
        phepex::preprocess_waveforms(src.data(), n_rows, n, cfg.up, &pz, 0, cfg.sm,
                                     &offset, 0, &scale, 0, ref.data());
        phepex::preprocess_waveforms(src.data(), n_rows, n, cfg.up, &pz, 0, cfg.sm,
                                     &offset, 0, &scale, 0, got.data(), scratch.data());
        REQUIRE(std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) == 0);
    }

    // Scalar path (upsampling == 1, no smoothing): no tiles, so the size query is 0 and a
    // supplied scratch is ignored -- still bit-identical to the internal-allocation call.
    REQUIRE(phepex::preprocess_waveforms_scratch_size(n, 1, nullptr) == 0);
    std::vector<float> ref(static_cast<std::size_t>(n_rows) * n);
    std::vector<float> got(static_cast<std::size_t>(n_rows) * n);
    std::vector<float> unused_scratch(8);
    phepex::preprocess_waveforms(src.data(), n_rows, n, 1, &pz, 0, nullptr, &offset, 0,
                                 &scale, 0, ref.data());
    phepex::preprocess_waveforms(src.data(), n_rows, n, 1, &pz, 0, nullptr, &offset, 0,
                                 &scale, 0, got.data(), unused_scratch.data());
    REQUIRE(std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) == 0);
}
