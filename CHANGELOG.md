# Changelog

All notable changes to phepex are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Version
numbers are derived from git tags (`vX.Y.Z`) by setuptools-scm (Python) and GitVersion
(C++); there is no version string committed to the tree.

## [Unreleased]

## [0.2.1] - 2026-07-30

### Changed
- Documentation and docstrings revised for accuracy and consistency across the README, the
  Sphinx pages, the C++ header comments and the Python/binding docstrings.

### Added
- Introductory slide deck (`docs/slides/phepex-intro.typ`, [Typst](https://typst.app)): eight
  slides on where phepex sits relative to libdvr and ctapipe, the kernels it exposes, and the
  0.2.0 benchmark figures.

## [0.2.0] - 2026-07-28

### Changed
- The `phepex.preprocess` binding evaluates the whole `(n_channels, n_pix, n_samples)` batch
  through the new `preprocess_waveforms` entry point in one call, instead of a Python-side loop
  over per-row `preprocess_waveform` calls. With smoothing enabled the rows are tiled 24 at a
  time and the Deriche IIR is evaluated one row per SIMD lane; the recurrence is loop-carried
  and latency-bound, so filling it with independent rows raises throughput. On the FlashCam
  micro-benchmark (1764 pixels, upsampling 4, 22 samples, aarch64/NEON) the upsampling +
  pole-zero + smoothing sweep is ~5x faster (1125 -> 222 us/event; `preprocess up/pz/smoothing`
  vs `preprocess up/pz/smoothing batch`). `upsampling > 1` without smoothing is tiled as well
  (`preprocess up/pz batch`); only the single case with no loop-carried recurrence --
  `upsampling == 1` and `smoothing == nullptr` -- stays on the per-row scalar kernel. Results are
  bit-for-bit unchanged: the tiled path uses double-precision smoothing accumulators as before
  and performs the per-lane arithmetic in the same order as the scalar kernels. The gain is from
  breaking the latency bound (independent rows in the reorder window), not SIMD width alone.
- `neighbor_peak_indices` accumulates neighbour waveforms in pairs (`buf[j] += a[j] +
  b[j]`) instead of one at a time, and for `local_weight == 0` seeds the accumulator from
  the first neighbour pair rather than a zeroed `self*0` pass. The per-kernel
  micro-benchmark attributes the kernel's cost to the read-modify-write traffic on the
  accumulator, not the neighbour reads (which are L1/L2-resident); pairing halves that
  traffic and skipping the self pass removes one full sweep over the trace. On the
  benchmarked aarch64 core (Apple M1, gcc -O3; 1764-pixel FlashCam, upsampling 4, 22
  samples, min/op over 400 reps) this is ~22% faster for `local_weight == 0` (227 -> 176
  us/event) and ~18% for `local_weight != 0` (231 -> 189 us/event). Pairwise grouping
  changes the float32 summation order, so the argmax can differ from a strictly sequential
  sum in near-ties within ~1 ULP. The strategy is selectable at compile time via the CMake option
  `PHEPEX_NEIGHBOR_PAIRWISE_SUM` (default `ON`); building with
  `-DPHEPEX_NEIGHBOR_PAIRWISE_SUM=OFF`, or compiling `neighbor.cpp` with the preprocessor
  macro `PHEPEX_NEIGHBOR_PAIRWISE_SUM=0`, restores the sequential sum.
- Deconvolution is unified with preprocessing; the separate deconvolution entry points are
  removed. Pole-zero deconvolution + upsampling is exactly preprocessing without the
  optional Deriche pass, so it is now `preprocess_waveform` / `preprocess_valid_range` with
  `smoothing == nullptr`. Removed in C++: `phepex::deconvolve_upsample` (the 3-D batch) and
  `phepex::deconvolve_valid_range`; the `phepex/deconvolve.hpp` header is deleted and
  `SampleRange` now lives in `phepex/preprocess.hpp`. Removed in the extension: the
  `_core.deconvolve_upsample` and `_core.deconvolve_valid_range` bindings. The public Python
  API is unchanged: `phepex.deconvolve(waveforms, baselines, upsampling, pole_zero)` and
  `phepex.deconvolve_valid_range(upsampling, n_samples, pole_zero)` keep their signatures,
  now implemented as thin wrappers over `phepex.preprocess` / `phepex.preprocess_valid_range`
  (smoothing disabled). As a result `phepex.deconvolve` also accepts per-pixel
  `pole_zero`/`baselines` (each independently a scalar, `(n_pix,)`, or `(n_channels, n_pix)`
  array, matching `preprocess`), where it previously forced `pole_zero` to a scalar. Results
  are bit-for-bit unchanged for scalar arguments and the ctapipe `FastFlashCamExtractor` is
  unaffected. One edge-case change: `deconvolve_valid_range` now returns `(0, 0)` instead of
  an inverted range when `n_samples` is too small to leave any trustworthy samples.
- `neighbor_peak_indices` takes the broken-pixel mask as `const std::uint8_t *` (nonzero =>
  broken) instead of `const bool *`. Callers held byte storage (`std::vector<char>`/`<unsigned
  char>`) and `reinterpret_cast` it to `bool *`. This was defective on two independent grounds:
  (1) reading a `char`/`unsigned char` object through a `bool` glvalue violates the
  strict-aliasing rule regardless of the stored value; and (2) a byte other than 0/1 is not a
  valid `bool` value, so loading it is UB — compilers assume `bool` is in {0,1} and miscompile
  (observed: a mask byte of 2 read via `bool *` tests as false at `-O2` on g++ 15.2, skipping
  the broken-pixel branch). A numpy bool array is genuine 1-byte bool storage holding only 0/1,
  so the Python binding's reinterpret to `uint8_t *` is well-defined. Source-incompatible for
  C++ callers passing `bool *`.
- Documentation is published one subdirectory per version on a `gh-pages` branch (`dev/` from
  `main`, `latest/` tracking the newest release tag, `vX.Y.Z/` per tag) instead of a
  single GitHub Pages artifact that replaced the whole site on each push. The `docs`
  workflow builds one version per run and publishes it with `peaceiris/actions-gh-pages`
  (`force_orphan`, so the branch is kept at a single commit). `latest/` is derived from
  the highest `vX.Y.Z` tag directory present, so pushing a backport tag older than the
  current newest does not move it. Tag builds are restricted to release tags matching
  `v[0-9]+.[0-9]+.[0-9]+`; pre-release tags (e.g. `v1.0.0-rc1`) are not built or
  published.

### Added
- `generate_waveforms` takes an `electronic_noise` parameter (C++ positional, between
  `nsb_rate_ghz` and `seed`; Python keyword `electronic_noise`, default `0.0`): the
  standard deviation of zero-mean Gaussian noise added independently to every output
  sample, in the units of `out` (p.e. of pulse integral per readout sample). It models
  uncorrelated readout/digitisation noise, which the Poisson NSB process cannot represent.
  The draws come last within each pixel from the same per-event RNG stream, so the signal
  and NSB draws are unaffected; `electronic_noise == 0` consumes no random numbers and
  reproduces earlier output bit-for-bit. Negative values raise `std::invalid_argument`/
  `ValueError` rather than silently disabling the noise.
- `phepex::preprocess_waveforms` (C++, `<phepex/preprocess.hpp>`): batched form of
  `preprocess_waveform` that applies the kernel to `n_rows` consecutive waveforms in one call.
  `pole_zero`, `offset` and `scale` are read per row as `array[row * stride]` (stride `0`
  broadcasts one value to every row, `1` selects a distinct per-row value). The rows are
  processed in fixed-width tiles (24 rows by default) held sample-major, one row per SIMD lane, whenever
  the per-row work carries a latency-bound recurrence -- the upsampling running sums
  (`upsampling > 1`) and/or the Deriche IIR (`smoothing != nullptr`) -- filling the recurrence
  with independent rows. The one case without such a recurrence, `upsampling == 1` with no
  smoothing (a pole-zero stencil that already vectorises across samples), uses the per-row
  scalar path, where tiling would only add transpose overhead. Overloaded for `const
  std::uint16_t *` and `const float *` input, matching `preprocess_waveform`; uint16 rows are
  widened during the tile transpose the batched path performs anyway, so raw ADC input costs no
  more than float input and needs no caller-side conversion pass. Input and output rows must be
  contiguous (row `r` at `src[r * n_samples]`, `out[r * n_samples * upsampling]`); strided views
  have to be copied by the caller. The result is
  bit-identical to calling `preprocess_waveform` per row (double-precision smoothing accumulators
  and arithmetic order are unchanged); the `n_rows % tile_width` remainder rows use
  `preprocess_waveform` directly. The tile width (default 24) is a build-time constant set via the
  CMake cache variable `PHEPEX_PREPROCESS_TILE_WIDTH`; it affects performance only (bit-identical
  for any width `>= 1`) and should be tuned to the target — see the README build options. An
  optional trailing `scratch` argument (default `nullptr`) accepts a caller-owned workspace for
  the tile buffers, avoiding the per-call allocation when preprocessing many batches in a loop;
  its required length is returned by the companion `preprocess_waveforms_scratch_size(n_samples,
  upsampling, smoothing)` (0 for the non-tiled `upsampling == 1`, no-smoothing case). Passing
  `nullptr` allocates internally, as before.
- `phepex.preprocess` and `phepex.preprocess_valid_range` Python bindings over the C++
  `preprocess_waveform` / `preprocess_valid_range` kernels. `preprocess` is a batched wrapper
  that applies the single-waveform kernel to every `(channel, pixel)` row of a
  `(n_channels, n_pix, n_samples)` array, returning float32
  `(n_channels, n_pix, n_samples*upsampling)`. `pole_zero`, `baseline` and `scale` are each
  independently a scalar, a per-pixel `(n_pix,)` array, or a `(n_channels, n_pix)` array;
  smoothing is enabled by a positive `smoothing_fwhm` (`0`/`None` disables it) with the
  Deriche coefficients computed once and shared across rows. With `smoothing_fwhm=0` the
  result matches `deconvolve` bit-for-bit (shared upsample kernel). A scalar
  `pole_zero`/`baseline`/`scale` is passed to `_core.preprocess` as a length-1 array and
  applied to every row with row stride 0, so a scalar-argument call does not allocate a
  full `(n_channels*n_pix,)` array per parameter. `waveforms` may be float32 or uint16:
  `_core.preprocess` is bound for both dtypes and the wrapper forwards a uint16 array
  uncopied (the kernel widens it inside the tile transpose it performs anyway), so raw ADC
  input no longer costs a full float32 staging array — on a (1, 1764, 22) uint16 batch with
  upsampling 4 and smoothing, 239 vs 261 us/call and 1.8 KiB instead of ~152 KiB of
  Python-side allocation. Every other dtype is converted to float32; the uint16 overload is
  bound `noconvert`, so it is reachable only by an exact dtype match and a float64 array can
  never be truncated into it. uint16 widens exactly, so both paths agree bit-for-bit.
- Documentation version switcher in the furo sidebar. A `switcher.json` at the site root lists
  the published versions; `docs/_static/version-switcher.js` fetches it at runtime and fills a
  `<select>`. The control is server-rendered with the `hidden` attribute and revealed only
  after a successful load, so a page with JavaScript disabled or an unreachable `switcher.json`
  (e.g. opened over `file://`) shows no control instead of an empty box. `docs/gen_switcher.py`
  regenerates `switcher.json`, the root redirect to `latest/`, and `.nojekyll` from the version
  directories present on the branch, so no entry points at an unpublished version.
- `ci` workflow: on every push and pull request, builds and tests both halves — the C++17
  library (CMake + Catch2 unit tests on Ubuntu and macOS, plus an installed-package
  `find_package`/pkg-config consumer build) and the Python package (from-source build on
  each supported interpreter, full pytest suite with the ctapipe reference on one).
- `phepex::generate_shower_image` and `phepex::ShowerModel` in `<phepex/generate.hpp>`: fills
  per-pixel `(charge, time_ns)` for one artificial shower image as input for
  `generate_waveforms`. Charges are Poisson draws around
  `intensity_pe * pdf(x, y) * pixel_area`, with `pdf` a Gaussian of sigma `width_m`
  transverse to the major axis times a skew-normal derived from (`length_m`, `skewness`)
  along it; times are linear in the longitudinal coordinate plus uniform jitter. C++ only;
  Python callers have `ctapipe.image.toymodel`.
- Example camera config, extracted from ctapipe by `scripts/export-camera-config.py`.
- C++ micro-benchmark (`benchmarks/cpp/microbench.cpp`) which runs on one artificial event:
  a skewed-Gaussian shower image (`generate_shower_image`) convolved with the camera's
  reference pulse over a 200 MHz Poisson NSB (`generate_waveforms`), digitised to 12-bit
  counts at 30 LSB per p.e. pulse integral on a 200 LSB pedestal. On the shipped FlashCam
  config the image covers ~200 pixels at up to ~100 p.e. with pulse times spanning ~32–64
  ns of the 88 ns window.

### Fixed
- `phepex.deconvolve` zeroed the first output sample at `upsampling == 1` even when
  `pole_zero == 0`, where there is no deconvolution and every sample is valid
  (`deconvolve_valid_range(1, n, 0)` returns `(0, n)`) -- destroying trustworthy data. The
  Python wrapper had a separate numpy path for `upsampling <= 1` that reproduced this; it is
  removed, and `deconvolve` now always uses the C++ kernel, which preserves the first sample
  as `wf - baseline`. At `upsampling == 1, pole_zero == 0` only sample 0 changes (`0` -> the
  baseline-subtracted value); the interior is unchanged. `deconvolve` now matches ctapipe
  in the trustworthy region rather than bit-for-bit at `upsampling == 1`, consistent with
  `upsampling > 1`.
- `upsampling < 1` is rejected with `ValueError` at the `_core` binding boundary, so it is
  caught for every caller of `_core.preprocess` / `_core.preprocess_valid_range` (and hence
  `phepex.preprocess`, `phepex.deconvolve`, `phepex.preprocess_valid_range` and
  `phepex.deconvolve_valid_range`), not only the `phepex.preprocess` wrapper. Previously
  `upsampling == 0` reached the C++ kernel via `preprocess`, which divides by `upsampling^2`
  and reads `output[-1]` out of bounds (undefined behavior); the valid-range path silently
  clamped `upsampling` to 1 and returned a range for an `upsampling` that `preprocess`
  itself refuses. Negative values were similarly out of contract (the documented domain is
  `upsampling >= 1`).
- `preprocess_waveform` (and the `phepex.preprocess` binding) skipped the pole-zero
  correction at `upsampling == 1`: the fast path computed `scale*(src - offset)` and ignored
  `pole_zero`. It now applies the single-pole correction, `out[i] = scale*((src[i]-offset) -
  pole_zero*(src[i-1]-offset))` with `out[0] = scale*(src[0]-offset)`, consistent with the
  `upsampling > 1` kernel. The `pole_zero == 0` result is unchanged. Output changes only for
  `upsampling == 1` with `pole_zero != 0`.
- `preprocess_valid_range` computed its upper bound as `num_samples - right`, subtracting an
  upsampled-sample margin from the RAW (pre-upsample) sample count while the lower bound and
  the `right` margin are already in upsampled samples. For `upsampling > 1` this returned a
  range far shorter than the `upsampling*num_samples` output (e.g. `(6, 34)` instead of
  `(6, 154)` for `upsampling=4, num_samples=40`), and produced inverted ranges (`lo > hi`)
  for small `num_samples`. The upper bound is now `upsampling*num_samples - right`, so the
  range indexes the upsampled output and equals `deconvolve_valid_range` when no smoothing is
  applied. Output changes for every `upsampling > 1` call.
- `docs` workflow checks out full git history (`fetch-depth: 0`) so setuptools-scm derives
  the tagged version; published documentation previously reported `0.0.0`.
- `tests/test_generator.py` imports `astropy.units` after the `pytest.importorskip("ctapipe")`
  guard. astropy ships as a ctapipe dependency, so the module now collects when ctapipe is
  absent instead of failing at import.

## [0.1.0] - 2026-07-20

### Added
- Initial release. C++17 kernel library (`libphepex.so`/`.a`, headers under `phepex/`):
  pole-zero deconvolution + upsampling, neighbour-sum peak finding, soft clipping, window
  integration, leading-edge timing, and a waveform generator. Standard-library-only, no
  OpenMP. Installs a CMake package config (`find_package(phepex)`) and a pkg-config file.
- Python bindings (`import phepex`): numpy kernel wrappers over the compiled extension, and
  the ctapipe-integrated `phepex.extractor.FastFlashCamExtractor` (imported on demand).
- Unified Sphinx documentation covering the C++ (Doxygen + Breathe) and Python (autodoc)
  API surfaces, published to GitHub Pages.

[Unreleased]: https://github.com/phepex/phepex/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/phepex/phepex/releases/tag/v0.2.1
[0.2.0]: https://github.com/phepex/phepex/releases/tag/v0.2.0
[0.1.0]: https://github.com/phepex/phepex/releases/tag/v0.1.0
