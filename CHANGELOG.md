# Changelog

All notable changes to phepex are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Version
numbers are derived from git tags (`vX.Y.Z`) by setuptools-scm (Python) and GitVersion
(C++); there is no version string committed to the tree.

## [Unreleased]

### Added
- `ci` workflow: on every push and pull request, builds and tests both halves — the C++17
  library (CMake + Catch2 unit tests on Ubuntu and macOS, plus an installed-package
  `find_package`/pkg-config consumer build) and the Python package (from-source build on
  each supported interpreter, full pytest suite with the ctapipe reference on one).

### Fixed
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
