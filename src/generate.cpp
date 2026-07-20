// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Max-Planck-Institut für Kernphysik
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

#include "phepex/generate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace phepex {
namespace {

// Linear interpolation matching numpy.interp on a uniform, strictly-increasing grid;
// clamps to the endpoints outside [xp.front(), xp.back()].
double interp(double x, const std::vector<double> &xp, const std::vector<double> &fp) {
    if (x <= xp.front())
        return fp.front();
    if (x >= xp.back())
        return fp.back();
    const double step = xp[1] - xp[0];
    const std::size_t i = static_cast<std::size_t>(x / step);
    const double t = (x - xp[i]) / step;
    return fp[i] + t * (fp[i + 1] - fp[i]);
}

// Bank of `upsampling` phase kernels built from a single-channel reference pulse.
// row(p)[k] is the downsampled response (p.e. units) to a unit charge deposited at
// readout-sample CENTER with sub-sample phase p; each row(p) sums to 1.
struct PhaseKernels {
    int upsampling;
    int klen;
    int center;
    int pad;
    // Flat, row-major kernel bank: upsampling * klen contiguous doubles. row(p)
    // returns the start of phase p's klen taps (see row() below).
    std::vector<double> g;

    const double *row(int p) const {
        return g.data() + static_cast<std::size_t>(p) * klen;
    }
    // Per-phase significant-tap window [nsb_lo, nsb_lo+nsb_len) capturing >= NSB_CAPTURE
    // of each kernel's integral. Used ONLY for NSB deposits (signal keeps the full klen
    // taps), trading a <(1-NSB_CAPTURE) shape approximation for ~2x fewer NSB deposit
    // adds.
    std::vector<int> nsb_lo;
    std::vector<int> nsb_len;

    PhaseKernels(const double *ref_pulse, std::size_t n_ref, double ref_sample_width_ns,
                 double sample_width_ns, int upsampling_)
        : upsampling(upsampling_) {
        const double ref_width = sample_width_ns / upsampling;

        std::vector<double> xp(n_ref), fp(n_ref);
        for (std::size_t i = 0; i < n_ref; ++i) {
            xp[i] = static_cast<double>(i) * ref_sample_width_ns;
            fp[i] = ref_pulse[i];
        }
        const double stop = xp.back();
        const std::size_t n_up = static_cast<std::size_t>(std::ceil(stop / ref_width));
        std::vector<double> ref_interp(n_up);
        double area = 0.0;
        for (std::size_t j = 0; j < n_up; ++j) {
            ref_interp[j] = interp(static_cast<double>(j) * ref_width, xp, fp);
            area += ref_interp[j];
        }
        const double norm = area * ref_width;
        std::size_t argmax = 0;
        double vmax = -1e300;
        for (std::size_t j = 0; j < n_up; ++j) {
            ref_interp[j] /= norm;
            if (ref_interp[j] > vmax) {
                vmax = ref_interp[j];
                argmax = j;
            }
        }

        const int khalf = static_cast<int>(std::ceil(double(n_up) / upsampling)) + 2;
        klen = 2 * khalf + 1;
        center = khalf;
        pad = khalf;
        const int L = static_cast<int>(n_up);
        const int am = static_cast<int>(argmax);

        g.assign(static_cast<std::size_t>(upsampling) * klen, 0.0);
        const int delta_pos = center * upsampling;
        const int n_up_out = klen * upsampling;
        std::vector<double> conv(n_up_out);
        for (int p = 0; p < upsampling; ++p) {
            const int s = delta_pos + p;
            std::fill(conv.begin(), conv.end(), 0.0);
            for (int k = 0; k < L; ++k) {
                const int j = s + k - am;
                if (j >= 0 && j < n_up_out)
                    conv[j] = ref_interp[k];
            }
            for (int nidx = 0; nidx < klen; ++nidx) {
                double acc = 0.0;
                for (int d = 0; d < upsampling; ++d)
                    acc += conv[nidx * upsampling + d];
                g[static_cast<std::size_t>(p) * klen + nidx] = acc * ref_width;
            }
        }

        // Per-phase significant-tap window: smallest contiguous span around the peak
        // capturing >= NSB_CAPTURE of the kernel's (non-negative) integral.
        constexpr double NSB_CAPTURE = 0.9999;
        nsb_lo.assign(upsampling, 0);
        nsb_len.assign(upsampling, klen);
        for (int p = 0; p < upsampling; ++p) {
            const double *gp = row(p);
            double tot = 0.0;
            int pk = 0;
            double gmax = gp[0];
            for (int n = 0; n < klen; ++n) {
                tot += gp[n];
                if (gp[n] > gmax) {
                    gmax = gp[n];
                    pk = n;
                }
            }
            int lo = pk, hi = pk;
            double s = gp[pk];
            while (s < NSB_CAPTURE * tot && (lo > 0 || hi < klen - 1)) {
                const double left = (lo > 0) ? gp[lo - 1] : -1.0;
                const double right = (hi < klen - 1) ? gp[hi + 1] : -1.0;
                if (hi < klen - 1 && right >= left) {
                    s += gp[++hi];
                } else if (lo > 0) {
                    s += gp[--lo];
                } else {
                    s += gp[++hi];
                }
            }
            nsb_lo[p] = lo;
            nsb_len[p] = hi - lo + 1;
        }
    }
};

// Fast PRNG (xoshiro256++ [1], seeded via splitmix64 [2]). Satisfies
// UniformRandomBitGenerator so it can drive std::poisson_distribution, but is ~5x faster
// than std::mt19937_64.
//
// [1]: David Blackman and Sebastiano Vigna. 2021. Scrambled Linear Pseudorandom Number
// Generators. ACM Trans. Math. Softw. 47, 4, Article 36 (September 2021), 32 pages.
// [doi:10.1145/3460772]
// [2]: Guy L. Steele, Jr., Doug Lea, and Christine H. Flood. 2014. Fast splittable
// pseudorandom number generators. In Proceedings of the 2014 ACM International Conference
// on Object Oriented Programming Systems Languages & Applications (OOPSLA’14). ACM, New
// York, NY, 453—472. [doi:10.1145/2660193.2660195]
struct Xoshiro256pp {
    using result_type = std::uint64_t;
    std::uint64_t s[4];
    static std::uint64_t rotl(std::uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
    explicit Xoshiro256pp(std::uint64_t seed) {
        // Use splitmix64's mix64variant13 [2] to fill the state, as suggested at the end
        // of [1] Sec. 5.3.
        std::uint64_t z = seed;
        for (int i = 0; i < 4; ++i) {
            z += 0x9e3779b97f4a7c15ULL;
            std::uint64_t x = z;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            s[i] = x ^ (x >> 31);
        }
    }
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return ~std::uint64_t(0); }
    result_type operator()() {
        const std::uint64_t r = rotl(s[0] + s[3], 23) + s[0];
        const std::uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return r;
    }
};

// Lemire's near-uniform bounded integer in [0, n) using 32 random bits (no rejection;
// bias ~ n/2^32 is negligible here). n must be < 2^32.
inline std::uint32_t bounded_u32(Xoshiro256pp &g, std::uint32_t n) {
    const std::uint32_t x = static_cast<std::uint32_t>(g() >> 32);
    return static_cast<std::uint32_t>((static_cast<std::uint64_t>(x) * n) >> 32);
}

// Add v*g[0..klen) into buf at [base, base+klen). `buf` must have klen-wide guard zones
// on each side so `base` is always valid, making this a branch-free, auto-vectorizable
// loop.
inline void deposit(double *buf, long base, const double *g, int klen, double v) {
    double *dst = buf + base;
    for (int m = 0; m < klen; ++m)
        dst[m] += v * g[m];
}

}  // namespace

void generate_waveforms(const double *charge, const double *time_ns, int n_events,
                        int n_pix, const double *reference_pulse, int n_ref,
                        double ref_sample_width_ns, double sample_width_ns, int n_samples,
                        int upsampling, double nsb_rate_ghz, std::uint64_t seed,
                        float *out) {
    const PhaseKernels pk(reference_pulse, static_cast<std::size_t>(n_ref),
                          ref_sample_width_ns, sample_width_ns, upsampling);
    const int pad = pk.pad;
    const int total = n_samples + 2 * pad;  // padded readout window
    const double ref_width = sample_width_ns / upsampling;
    const double nsb_mean = nsb_rate_ghz * ref_width * upsampling * total;

    const std::uint32_t n_cells = static_cast<std::uint32_t>(total) * upsampling;
    // buf carries klen-wide guard zones on each side so every deposit is in-range
    // (branch-free and vectorizable); the guard captures kernel tails that fall outside
    // the readout window, which are discarded at crop anyway (identical output to per-tap
    // clamping).
    const int guard = pk.klen;
    const int total_g = total + 2 * guard;
    std::vector<double> buf(total_g);
    // The NSB Poisson distribution depends only on the loop-invariant mean; construct it
    // once and reset() its cached state per event (equivalent to a fresh distribution, so
    // the per-event draws stay bit-identical while avoiding repeated setup).
    std::poisson_distribution<int> nsb_pois(nsb_mean);
    for (std::size_t e = 0; e < static_cast<std::size_t>(n_events); ++e) {
        // Deterministic per-event RNG so results are reproducible & event-independent.
        Xoshiro256pp rng(seed + 0x9e3779b97f4a7c15ULL * (e + 1));
        nsb_pois.reset();

        for (std::size_t pix = 0; pix < static_cast<std::size_t>(n_pix); ++pix) {
            std::fill(buf.begin(), buf.end(), 0.0);

            // signal: one deposit per pixel, placed for physical correctness (which
            // deliberately differs from ctapipe's WaveformModel near the window edges).
            // The deposit time is snapped to the upsampled grid with floor() (not
            // truncation toward zero) so negative/early times land in the correct cell
            // and sub-sample phase; the phase is taken with a non-negative modulo. A
            // deposit is rendered whenever its kernel overlaps the padded readout window
            // (ns in [0, total), pad == kernel half-width), so a pulse centred just
            // before the window still contributes its rising tail -- rather than being
            // dropped outright as WaveformModel does for any sample < 0.
            const double q = charge[e * n_pix + pix];
            const double t = time_ns[e * n_pix + pix];
            if (q != 0.0) {
                const long s_up = static_cast<long>(std::floor(t / ref_width));
                const int phase =
                    static_cast<int>(((s_up % upsampling) + upsampling) % upsampling);
                const long ns = (s_up - phase) / upsampling + pad;
                if (ns >= 0 && ns < total)
                    deposit(buf.data(), (ns - pk.center) + guard, pk.row(phase), pk.klen,
                            q);
            }

            // NSB: Poisson p.e. at uniform time & phase over the padded window. One
            // bounded draw per p.e. gives the joint (readout-sample, phase) cell
            // uniformly.
            if (nsb_mean > 0.0) {
                const int n_nsb = nsb_pois(rng);
                for (int h = 0; h < n_nsb; ++h) {
                    const std::uint32_t cell = bounded_u32(rng, n_cells);
                    const int ns = static_cast<int>(cell / upsampling);
                    const int phase = static_cast<int>(cell % upsampling);
                    // NSB deposits use only the kernel's significant taps
                    // (approximation).
                    const int fo = pk.nsb_lo[phase];
                    const long base = (static_cast<long>(ns) - pk.center) + guard + fo;
                    deposit(buf.data(), base, pk.row(phase) + fo, pk.nsb_len[phase], 1.0);
                }
            }

            // crop away the guard + padding into the output
            float *dst = out + (e * n_pix + pix) * n_samples;
            for (int n = 0; n < n_samples; ++n)
                dst[n] = static_cast<float>(buf[guard + pad + n]);
        }
    }
}

}  // namespace phepex
