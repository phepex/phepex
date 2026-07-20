// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Max-Planck-Institut für Kernphysik
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

#ifndef PHEPEX_HPP
#define PHEPEX_HPP

/// phepex -- photo-electron pulse extraction.
///
/// A small, dependency-free C++17 library of the numeric kernels used to extract charge
/// and timing from PMT/SiPM (e.g. Cherenkov telescopes or Water Cherenkov Detectors)
/// waveforms: pole-zero deconvolution + upsampling, neighbour-sum peak finding, soft
/// clipping, window integration and leading-edge timing, plus a fast waveform generator
/// for testing.
///
/// All functions live in namespace `phepex`, operate on caller-owned raw buffers, and
/// depend only on the C++ standard library.

#include "phepex/clip.hpp"
#include "phepex/deconvolve.hpp"
#include "phepex/extract.hpp"
#include "phepex/generate.hpp"
#include "phepex/neighbor.hpp"
#include "phepex/preprocess.hpp"
#include "phepex/version.hpp"

#endif  // PHEPEX_HPP
