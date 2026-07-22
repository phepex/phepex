# Changelog

All notable changes to phepex are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Version
numbers are derived from git tags (`vX.Y.Z`) by setuptools-scm (Python) and GitVersion
(C++); there is no version string committed to the tree.

## [Unreleased]

### Changed
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
- `phepex.preprocess` and `phepex.preprocess_valid_range` Python bindings over the C++
  `preprocess_waveform` / `preprocess_valid_range` kernels. `preprocess` is a batched wrapper
  that applies the single-waveform kernel to every `(channel, pixel)` row of a
  `(n_channels, n_pix, n_samples)` array, returning float32
  `(n_channels, n_pix, n_samples*upsampling)`. `pole_zero`, `baseline` and `scale` are each
  independently a scalar, a per-pixel `(n_pix,)` array, or a `(n_channels, n_pix)` array;
  smoothing is enabled by a positive `smoothing_fwhm` (`0`/`None` disables it) with the
  Deriche coefficients computed once and shared across rows. With `smoothing_fwhm=0` the
  result matches `deconvolve` bit-for-bit (shared upsample kernel).
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

### Fixed
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

[Unreleased]: https://github.com/phepex/phepex/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/phepex/phepex/releases/tag/v0.1.0
