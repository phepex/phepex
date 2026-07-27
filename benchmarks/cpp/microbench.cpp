/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If
 * a copy of the MPL was not distributed with this file, You can obtain one at
 * https://mozilla.org/MPL/2.0/.
 *
 * SPDX-License-Identifier: MPL-2.0
 *
 * Copyright (c) 2026 Max-Planck-Institut für Kernphysik
 */

/* Per-kernel micro-benchmarks for the phepex signal-processing kernels. Each kernel is
 * timed in isolation on a camera configuration loaded from a text file (see
 * scripts/export-camera-config.py and benchmarks/flashcam-config.txt), so a regression
 * can be attributed to a specific kernel rather than to the pipeline as a whole.
 *
 * This is a normal executable built with the library's optimisation flags (-O3), unlike
 * the Catch2 unit tests which run under coverage/sanitizer instrumentation, so its
 * absolute numbers are meaningful. For machine-tuned figures build with -march=native,
 * e.g. cmake -DPHEPEX_BUILD_BENCHMARKS=ON -DCMAKE_CXX_FLAGS="-march=native" ..
 *
 * The reported statistic is the minimum time over all reps (least contaminated by
 * scheduler preemption), with the median alongside as a spread indicator. One "op" is the
 * per-event cost of that kernel: one full-camera sweep (all pixels, one gain channel).
 *
 * The input is one artificial event rather than random samples, because the peak-search
 * and centroid kernels are data-dependent: a skewed-Gaussian shower image
 * (generate_shower_image) convolved with the camera's reference pulse and superposed on a
 * Poisson NSB process (generate_waveforms), digitised to 12-bit ADC counts on a pedestal.
 * The shower covers ~200 pixels at ~1-100 p.e. with a ~1 ns/pixel time gradient; see the
 * constants in main().
 */

#include "phepex/clip.hpp"        // for pos_soft_clip
#include "phepex/extract.hpp"     // for extract_around_peak, adaptive_centroid
#include "phepex/generate.hpp"    // for generate_waveforms, generate_shower_image
#include "phepex/neighbor.hpp"    // for neighbor_peak_indices
#include "phepex/preprocess.hpp"  // for preprocess_waveform, calculate_smoothing_coeff...

#include <algorithm>  // for min, max, sort
#include <chrono>     // for steady_clock
#include <cmath>      // for lround
#include <cstdint>    // for uint16_t, int32_t, int64_t, uint64_t
#include <cstring>    // for strcmp
#include <fstream>    // for ifstream
#include <iomanip>    // for setw, setprecision
#include <iostream>   // for cout, cerr
#include <sstream>    // for istringstream
#include <stdexcept>  // for runtime_error
#include <string>     // for string
#include <vector>     // for vector

using Clock = std::chrono::steady_clock;

namespace {

// Volatile sink to keep the optimiser from eliding the benchmarked work.
volatile double g_sink = 0;

/// Camera configuration (geometry + readout scalars) loaded from the flat text file
/// written by scripts/export-camera-config.py. No kernel reads the pixel geometry -- the
/// neighbour adjacency already encodes the connectivity derived from it -- but the
/// coordinates, area and pitch are needed to place the artificial shower image the
/// kernels are benchmarked on, in camera-independent units.
struct CameraConfig {
    std::string name;
    int num_pixels = 0;
    int num_samples = 0;
    double sampling_rate_ghz = 0;
    double ref_sample_width_ns = 0;
    double pix_area_m2 = 0;
    double pix_pitch_m = 0;
    std::vector<double> pix_x;            // pixel centres (m), length num_pixels
    std::vector<double> pix_y;            // pixel centres (m), length num_pixels
    std::vector<double> reference_pulse;  // n_ref reference pulse samples
    std::vector<std::int32_t> indptr;     // CSR row pointers, length num_pixels+1
    std::vector<std::int32_t> indices;    // CSR neighbour ids, length neighbor_nnz
};

/// Parse the config text file. Lines beginning with '#' (or content after a '#') are
/// comments. Tokens are whitespace-delimited. Scalars appear as `key value`; the arrays
/// (`pix_x`, `pix_y`, `reference_pulse`, `indptr`, `indices`) appear as a bare `key`
/// followed by a fixed number of tokens, so their length fields (`num_reference_pulse`,
/// `num_pixels`, `neighbor_nnz`) must be read before the arrays they size. Throws
/// std::runtime_error on a malformed file, an unexpected key, or a missing/non-positive
/// field the benchmark's event generation requires.
CameraConfig LoadCameraConfig(const std::string &path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open config file: " + path);

    // Flatten to a token stream, dropping comments.
    std::vector<std::string> tok;
    std::string line;
    while (std::getline(in, line)) {
        const auto hash = line.find('#');
        if (hash != std::string::npos)
            line.erase(hash);
        std::istringstream ls(line);
        std::string t;
        while (ls >> t)
            tok.push_back(t);
    }

    CameraConfig g;
    int num_reference_pulse = -1;
    int neighbor_nnz = -1;
    size_t i = 0;
    auto next = [&](const char *what) -> std::string {
        if (i >= tok.size())
            throw std::runtime_error(std::string("unexpected end of file, expected ") +
                                     what);
        return tok[i++];
    };

    while (i < tok.size()) {
        const std::string key = tok[i++];
        if (key == "name") {
            g.name = next("name value");
        } else if (key == "num_pixels") {
            g.num_pixels = std::stoi(next("num_pixels value"));
        } else if (key == "num_samples") {
            g.num_samples = std::stoi(next("num_samples value"));
        } else if (key == "sampling_rate_ghz") {
            g.sampling_rate_ghz = std::stod(next("sampling_rate_ghz value"));
        } else if (key == "ref_sample_width_ns") {
            g.ref_sample_width_ns = std::stod(next("ref_sample_width_ns value"));
        } else if (key == "pix_area_m2") {
            g.pix_area_m2 = std::stod(next("pix_area_m2 value"));
        } else if (key == "pix_pitch_m") {
            g.pix_pitch_m = std::stod(next("pix_pitch_m value"));
        } else if (key == "pix_x" || key == "pix_y") {
            if (g.num_pixels <= 0)
                throw std::runtime_error(key + " before num_pixels");
            std::vector<double> &dst = (key == "pix_x") ? g.pix_x : g.pix_y;
            dst.reserve(g.num_pixels);
            for (int k = 0; k < g.num_pixels; k++)
                dst.push_back(std::stod(next("pixel coordinate")));
        } else if (key == "num_reference_pulse") {
            num_reference_pulse = std::stoi(next("num_reference_pulse value"));
        } else if (key == "neighbor_nnz") {
            neighbor_nnz = std::stoi(next("neighbor_nnz value"));
        } else if (key == "reference_pulse") {
            if (num_reference_pulse < 0)
                throw std::runtime_error("reference_pulse before num_reference_pulse");
            g.reference_pulse.reserve(num_reference_pulse);
            for (int k = 0; k < num_reference_pulse; k++)
                g.reference_pulse.push_back(std::stod(next("reference_pulse sample")));
        } else if (key == "indptr") {
            if (g.num_pixels <= 0)
                throw std::runtime_error("indptr before num_pixels");
            g.indptr.reserve(g.num_pixels + 1);
            for (int k = 0; k < g.num_pixels + 1; k++)
                g.indptr.push_back(std::stoi(next("indptr entry")));
        } else if (key == "indices") {
            if (neighbor_nnz < 0)
                throw std::runtime_error("indices before neighbor_nnz");
            g.indices.reserve(neighbor_nnz);
            for (int k = 0; k < neighbor_nnz; k++)
                g.indices.push_back(std::stoi(next("indices entry")));
        } else {
            throw std::runtime_error("unknown key in config file: " + key);
        }
    }

    if (g.num_pixels <= 0 || g.num_samples <= 0)
        throw std::runtime_error("config missing num_pixels/num_samples");
    if (static_cast<int>(g.indptr.size()) != g.num_pixels + 1)
        throw std::runtime_error("indptr length != num_pixels+1");
    if (static_cast<int>(g.indices.size()) != neighbor_nnz)
        throw std::runtime_error("indices length != neighbor_nnz");
    if (static_cast<int>(g.pix_x.size()) != g.num_pixels ||
        static_cast<int>(g.pix_y.size()) != g.num_pixels || g.pix_area_m2 <= 0 ||
        g.pix_pitch_m <= 0)
        throw std::runtime_error(
            "config lacks pix_x/pix_y/pix_area_m2/pix_pitch_m; regenerate it with "
            "scripts/export-camera-config.py");
    // The readout scalars below divide into sample widths, and the pulse shape is
    // indexed unconditionally by generate_waveforms; a missing or zero value would
    // otherwise degenerate the kernel bank (plausible but meaningless timings) or read
    // past the end of an empty vector.
    if (g.sampling_rate_ghz <= 0 || g.ref_sample_width_ns <= 0)
        throw std::runtime_error(
            "config needs sampling_rate_ghz > 0 and ref_sample_width_ns > 0");
    if (g.reference_pulse.empty() ||
        static_cast<int>(g.reference_pulse.size()) != num_reference_pulse)
        throw std::runtime_error("config needs a non-empty reference_pulse of length "
                                 "num_reference_pulse");
    return g;
}

/** @brief Runs `body` `reps` times; `body` returns the number of ops it performed and a
 * value folded into g_sink. Reports min and median us/op. */
template <typename Body>
void Run(const std::string &name, const std::string &note, unsigned int reps,
         Body &&body) {
    std::vector<double> us_per_op;
    us_per_op.reserve(reps);
    double sink = 0;

    // Warmup: two untimed passes to fault in pages and warm caches/branch predictors.
    for (int w = 0; w < 2; w++) {
        auto [ops, s] = body();
        sink += s;
        (void)ops;
    }

    for (unsigned int r = 0; r < reps; r++) {
        const auto t0 = Clock::now();
        auto [ops, s] = body();
        const auto t1 = Clock::now();
        sink += s;
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        us_per_op.push_back(ops > 0 ? us / ops : us);
    }
    g_sink += sink;

    std::sort(us_per_op.begin(), us_per_op.end());
    const double min = us_per_op.front();
    const double median = us_per_op[us_per_op.size() / 2];

    std::cout << "  " << std::left << std::setw(33) << name << std::right << std::fixed
              << std::setprecision(1) << std::setw(7) << min << " us  " << std::setw(7)
              << median << " us    " << note << "\n";
}

}  // namespace

int main(int argc, char **argv) {
    std::string config_filename = "benchmarks/flashcam-config.txt";
    unsigned int reps = 200;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc)
            config_filename = argv[++i];
        else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc)
            reps = static_cast<unsigned int>(std::strtoul(argv[++i], nullptr, 10));
        else {
            std::cerr << "Usage: " << argv[0] << " [--config PATH] [--reps N]\n"
                      << "Benchmarks the phepex signal-processing kernels on the camera "
                         "configuration\n"
                      << "from the given text file (default "
                         "benchmarks/flashcam-config.txt).\n"
                      << "One op = one full-camera sweep (per-event kernel cost).\n";
            return std::strcmp(argv[i], "--help") == 0 ? 0 : 2;
        }
    }

    CameraConfig cfg;
    try {
        cfg = LoadCameraConfig(config_filename);
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    const int n_ch = 1;  // FlashCam has a single gain channel
    const int num_pixels = cfg.num_pixels;
    const int num_samples = cfg.num_samples;
    const double sample_width_ns = 1.0 / cfg.sampling_rate_ghz;

    // Shower image: a skewed Gaussian at the camera centre. Its axes and time gradient
    // are expressed in pixel pitches so the same numbers give a comparable event on any
    // camera in the config format: 4 x 1.4 pitches of Gaussian sigma puts ~200 pixels
    // above 1 p.e., and the gradient is 1 ns per pitch along the major axis. Measured on
    // the shipped FlashCam config (pitch 5 cm): 212 pixels at 1..106 p.e., pulse times
    // spanning 31.2..63.8 ns inside the 88 ns readout window. The per-pixel charge scale
    // is pitch-independent: the pdf peak goes as 1/pitch^2 and the pixel area as pitch^2.
    phepex::ShowerModel shower;
    shower.length_m = 4.0 * cfg.pix_pitch_m;
    shower.width_m = 1.4 * cfg.pix_pitch_m;
    shower.psi_rad = 0.3;
    shower.skewness = 0.3;
    shower.intensity_pe = 4000.0;
    shower.time_gradient_ns_per_m = 1.0 / cfg.pix_pitch_m;
    shower.time_intercept_ns = 0.5 * num_samples * sample_width_ns;  // window centre
    shower.time_jitter_ns = 0.5;
    std::vector<double> q(num_pixels), tns(num_pixels);
    phepex::generate_shower_image(shower, cfg.pix_x.data(), cfg.pix_y.data(), num_pixels,
                                  cfg.pix_area_m2, /*seed=*/1, q.data(), tns.data());

    // Raw waveforms of one artificial event: the shower image convolved with the
    // reference pulse plus a Poisson NSB process, digitised to 12-bit ADC counts.
    // Deterministic seeds, so the benchmark is reproducible across runs.
    constexpr int gen_up = 10;            // oversampling factor in the simulation
    constexpr double nsb_rate_ghz = 0.2;  // representative dark-sky NSB
    // generate_waveforms conserves charge in readout-sample units, so the ADC gain below
    // is the raw pulse integral in LSB per p.e. at this sampling rate; it and the
    // pedestal are the knobs to adjust per camera.
    constexpr double pedestal_lsb = 200.0;
    constexpr double spe_integral_lsb = 30.0;
    std::vector<float> amp(static_cast<size_t>(num_pixels) * num_samples);
    phepex::generate_waveforms(
        q.data(), tns.data(), /*n_events=*/1, num_pixels, cfg.reference_pulse.data(),
        static_cast<int>(cfg.reference_pulse.size()), cfg.ref_sample_width_ns,
        sample_width_ns, num_samples, gen_up, nsb_rate_ghz, /*seed=*/42, amp.data());
    std::vector<std::uint16_t> raw(static_cast<size_t>(num_pixels) * num_samples);
    for (size_t i = 0; i < raw.size(); i++) {
        const long adc = std::lround(pedestal_lsb + spe_integral_lsb * amp[i]);
        raw[i] = static_cast<std::uint16_t>(std::min(4095L, std::max(0L, adc)));
    }

    // Baseline to subtract: the pedestal plus the DC level the NSB adds on top of it, as
    // a pedestal calibration from NSB-illuminated events would yield.
    const float preprocess_offset = static_cast<float>(
        pedestal_lsb + nsb_rate_ghz * sample_width_ns * spe_integral_lsb);
    // Scale putting the preprocessed pulse in p.e..
    const float preprocess_scale = static_cast<float>(1.0 / spe_integral_lsb);

    int signal_pixels = 0;
    for (int p = 0; p < num_pixels; p++)
        signal_pixels += (q[p] >= 1.0);
    std::cout << "phepex-microbench\n"
              << "  config  : " << config_filename << " (" << cfg.name << ")\n"
              << "  camera  : " << num_pixels << " pixels x " << num_samples
              << " samples, " << cfg.sampling_rate_ghz << " GHz\n"
              << "  event   : " << shower.intensity_pe << " p.e. shower over "
              << signal_pixels << " pixels (>= 1 p.e.), " << nsb_rate_ghz * 1000.0
              << " MHz NSB, " << pedestal_lsb << " LSB pedestal\n"
              << "  reps    : " << reps << "\n"
              << "  (one op = one full-camera sweep = per-event kernel cost)\n\n"
              << "  kernel                               min/op   median/op    note\n"
              << "  ------                            ---------   ---------    ----\n";

    // preprocess_waveform: offset/scale only (upsampling 1, no pole-zero)
    {
        const int up = 1;
        std::vector<float> dst(static_cast<size_t>(num_pixels) * num_samples * up);
        Run("preprocess", "offset/scale only", reps, [&] {
            for (int p = 0; p < num_pixels; p++)
                phepex::preprocess_waveform(
                    raw.data() + static_cast<size_t>(p) * num_samples, num_samples, up,
                    0.0f, nullptr, preprocess_offset, preprocess_scale,
                    dst.data() + static_cast<size_t>(p) * num_samples * up);
            return std::pair<int, double>{1, dst[num_samples * up / 2]};
        });
    }

    // preprocess_waveform: pole-zero deconvolution only (upsampling 1)
    {
        const int up = 1;
        const float pz = 0.75f;
        std::vector<float> dst(static_cast<size_t>(num_pixels) * num_samples * up);
        Run("preprocess pz", "pole-zero only", reps, [&] {
            for (int p = 0; p < num_pixels; p++)
                phepex::preprocess_waveform(
                    raw.data() + static_cast<size_t>(p) * num_samples, num_samples, up,
                    pz, nullptr, preprocess_offset, preprocess_scale,
                    dst.data() + static_cast<size_t>(p) * num_samples * up);
            return std::pair<int, double>{1, dst[num_samples * up / 2]};
        });
    }

    // preprocess_waveform: 4x upsampling only
    {
        const int up = 4;
        std::vector<float> dst(static_cast<size_t>(num_pixels) * num_samples * up);
        Run("preprocess up", "4x upsampling only", reps, [&] {
            for (int p = 0; p < num_pixels; p++)
                phepex::preprocess_waveform(
                    raw.data() + static_cast<size_t>(p) * num_samples, num_samples, up,
                    0.0f, nullptr, preprocess_offset, preprocess_scale,
                    dst.data() + static_cast<size_t>(p) * num_samples * up);
            return std::pair<int, double>{1, dst[num_samples * up / 2]};
        });
    }

    // preprocess_waveform: 4x upsampling + pole-zero (FlashCam deconvolution)
    {
        const int up = 4;
        const float pz = 0.75f;
        std::vector<float> dst(static_cast<size_t>(num_pixels) * num_samples * up);
        Run("preprocess up/pz", "4x upsampling + pole-zero", reps, [&] {
            for (int p = 0; p < num_pixels; p++)
                phepex::preprocess_waveform(
                    raw.data() + static_cast<size_t>(p) * num_samples, num_samples, up,
                    pz, nullptr, preprocess_offset, preprocess_scale,
                    dst.data() + static_cast<size_t>(p) * num_samples * up);
            return std::pair<int, double>{1, dst[num_samples * up / 2]};
        });
    }

    // preprocess_waveforms: 4x upsampling + pole-zero (no smoothing) through the
    // batched, row-tiled entry point.
    {
        const int up = 4;
        const float pz = 0.75f;
        std::vector<float> dst(static_cast<size_t>(num_pixels) * num_samples * up);
        std::vector<float> scratch(
            phepex::preprocess_waveforms_scratch_size(num_samples, up, nullptr));
        Run("preprocess up/pz batch", "4x upsampling + pole-zero, tiled", reps, [&] {
            phepex::preprocess_waveforms(
                raw.data(), num_pixels, num_samples, up, &pz, 0, nullptr,
                &preprocess_offset, 0, &preprocess_scale, 0, dst.data(), scratch.data());
            return std::pair<int, double>{1, dst[num_samples * up / 2]};
        });
    }

    // preprocess_waveform: smoothing only (upsampling 1, no pole-zero)
    {
        const int up = 1;
        const float pz = 0.0f;
        const auto coeffs = phepex::calculate_smoothing_coefficients(3.0);
        std::vector<float> dst(static_cast<size_t>(num_pixels) * num_samples * up);
        std::vector<float> scratch(static_cast<size_t>(num_samples) * up);
        Run("preprocess smoothing", "Deriche smoothing only", reps, [&] {
            for (int p = 0; p < num_pixels; p++)
                phepex::preprocess_waveform(
                    raw.data() + static_cast<size_t>(p) * num_samples, num_samples, up,
                    pz, &coeffs, preprocess_offset, preprocess_scale,
                    dst.data() + static_cast<size_t>(p) * num_samples * up,
                    scratch.data());
            return std::pair<int, double>{1, dst[num_samples * up / 2]};
        });
    }

    // preprocess_waveform: 4x upsampling + smoothing (no pole-zero)
    {
        const int up = 4;
        const float pz = 0.0f;
        const auto coeffs = phepex::calculate_smoothing_coefficients(3.0);
        std::vector<float> dst(static_cast<size_t>(num_pixels) * num_samples * up);
        std::vector<float> scratch(static_cast<size_t>(num_samples) * up);
        Run("preprocess up/smoothing", "4x upsampling + smoothing", reps, [&] {
            for (int p = 0; p < num_pixels; p++)
                phepex::preprocess_waveform(
                    raw.data() + static_cast<size_t>(p) * num_samples, num_samples, up,
                    pz, &coeffs, preprocess_offset, preprocess_scale,
                    dst.data() + static_cast<size_t>(p) * num_samples * up,
                    scratch.data());
            return std::pair<int, double>{1, dst[num_samples * up / 2]};
        });
    }

    // preprocess_waveform: 4x upsampling + pole-zero + smoothing (all sub-kernels)
    {
        const int up = 4;
        const float pz = 0.75f;
        const auto coeffs = phepex::calculate_smoothing_coefficients(3.0);
        std::vector<float> dst(static_cast<size_t>(num_pixels) * num_samples * up);
        std::vector<float> scratch(static_cast<size_t>(num_samples) * up);
        Run("preprocess up/pz/smoothing", "all sub-kernels", reps, [&] {
            for (int p = 0; p < num_pixels; p++)
                phepex::preprocess_waveform(
                    raw.data() + static_cast<size_t>(p) * num_samples, num_samples, up,
                    pz, &coeffs, preprocess_offset, preprocess_scale,
                    dst.data() + static_cast<size_t>(p) * num_samples * up,
                    scratch.data());
            return std::pair<int, double>{1, dst[num_samples * up / 2]};
        });
    }

    // preprocess_waveforms: the same all-sub-kernels workload through the batched,
    // row-tiled entry point
    {
        const int up = 4;
        const float pz = 0.75f;
        const auto coeffs = phepex::calculate_smoothing_coefficients(3.0);
        std::vector<float> dst(static_cast<size_t>(num_pixels) * num_samples * up);
        std::vector<float> scratch(
            phepex::preprocess_waveforms_scratch_size(num_samples, up, &coeffs));
        Run("preprocess up/pz/smoothing batch", "all sub-kernels, tiled", reps, [&] {
            phepex::preprocess_waveforms(
                raw.data(), num_pixels, num_samples, up, &pz, 0, &coeffs,
                &preprocess_offset, 0, &preprocess_scale, 0, dst.data(), scratch.data());
            return std::pair<int, double>{1, dst[num_samples * up / 2]};
        });
    }

    // A realistic deconvolved buffer for the downstream kernels (clip / neighbour /
    // extract): 4x upsampling + pole-zero, the FlashCam configuration. Computed once,
    // outside timing.
    const int up = 4;
    const float pz = 0.75f;
    const int n_up = num_samples * up;
    std::vector<float> signals(static_cast<size_t>(num_pixels) * n_up);
    for (int p = 0; p < num_pixels; p++)
        phepex::preprocess_waveform(raw.data() + static_cast<size_t>(p) * num_samples,
                                    num_samples, up, pz, nullptr, preprocess_offset,
                                    preprocess_scale,
                                    signals.data() + static_cast<size_t>(p) * n_up);
    // Trustworthy (non-edge) sample range of the deconvolution; the search/extraction
    // kernels are restricted to it, as in the extractor pipeline.
    const phepex::SampleRange vr =
        phepex::preprocess_valid_range(up, pz, nullptr, num_samples);

    // pos_soft_clip over the valid range
    {
        // Representative FlashCam neighbour-sum clipping level, in the p.e. units the
        // preprocess `scale` above puts the deconvolved signal in.
        const float clip_scale = 12.0f;
        std::vector<float> clipped(static_cast<size_t>(num_pixels) * n_up);
        Run("pos_soft_clip", "positive soft clip", reps, [&] {
            phepex::pos_soft_clip(signals.data(), n_ch, num_pixels, n_up, clip_scale,
                                  vr.lo, vr.hi, clipped.data());
            return std::pair<int, double>{
                1, clipped[static_cast<size_t>(num_pixels / 2) * n_up + n_up / 2]};
        });
    }

    // neighbor_peak_indices over the real CSR neighbour graph
    std::vector<std::uint8_t> broken(static_cast<size_t>(n_ch) * num_pixels, 0);
    std::vector<std::int64_t> peak(static_cast<size_t>(n_ch) * num_pixels);
    std::vector<std::int32_t> count(static_cast<size_t>(n_ch) * num_pixels);
    for (int local_weight : {0, 1}) {
        const std::string note =
            local_weight == 0 ? "neighbour sum, local_weight=0" : "local_weight=1";
        Run(local_weight == 0 ? "neighbor_peak_indices" : "neighbor_peak_indices lw1",
            note, reps, [&] {
                phepex::neighbor_peak_indices(signals.data(), n_ch, num_pixels, n_up,
                                              cfg.indptr.data(), cfg.indices.data(),
                                              local_weight, broken.data(), vr.lo, vr.hi,
                                              peak.data(), count.data());
                return std::pair<int, double>{1,
                                              static_cast<double>(peak[num_pixels / 2])};
            });
    }

    // extract_around_peak (window integration + weighted peak time)
    {
        std::vector<float> charge(static_cast<size_t>(n_ch) * num_pixels);
        std::vector<float> peak_time(static_cast<size_t>(n_ch) * num_pixels);
        Run("extract_around_peak", "window integral + peak time", reps, [&] {
            phepex::extract_around_peak(
                signals.data(), n_ch, num_pixels, n_up, peak.data(),
                /*width=*/7, /*shift=*/3, cfg.sampling_rate_ghz * up, charge.data(),
                peak_time.data());
            return std::pair<int, double>{1, static_cast<double>(charge[num_pixels / 2])};
        });
    }

    // adaptive_centroid (leading-edge weighted centroid)
    {
        std::vector<float> centroids(static_cast<size_t>(n_ch) * num_pixels);
        Run("adaptive_centroid", "leading-edge centroid", reps, [&] {
            phepex::adaptive_centroid(signals.data(), n_ch, num_pixels, n_up, peak.data(),
                                      /*rel_descend_limit=*/0.5, centroids.data());
            return std::pair<int, double>{1,
                                          static_cast<double>(centroids[num_pixels / 2])};
        });
    }

    // generate_waveforms: re-synthesise the event that fed the kernels above, one fresh
    // NSB realisation per rep.
    {
        std::vector<float> out(static_cast<size_t>(num_pixels) * num_samples);
        std::uint64_t seed = 1;
        Run("generate_waveforms", "pulse convolution + NSB", reps, [&] {
            phepex::generate_waveforms(
                q.data(), tns.data(), /*n_events=*/1, num_pixels,
                cfg.reference_pulse.data(), static_cast<int>(cfg.reference_pulse.size()),
                cfg.ref_sample_width_ns, sample_width_ns, num_samples, gen_up,
                nsb_rate_ghz, seed++, out.data());
            return std::pair<int, double>{1, static_cast<double>(out[num_samples / 2])};
        });
    }

    if (g_sink == -1.0)  // never true; keeps g_sink observably live
        std::cout << "\n";
    return 0;
}
