# phepex — photo-electron pulse extraction

<!-- intro:start -->
A small, dependency-free **C++17 library** of the numeric kernels used to extract charge and timing from digitised PMT/SiPM (e.g. Cherenkov telescopes or Water Cherenkov Detectors) waveforms — pole-zero deconvolution + upsampling, neighbour-sum peak finding, soft clipping, window integration and leading-edge timing — plus a fast waveform generator for testing.

Ships **`libphepex.so`** and **`libphepex.a`**, public headers under `phepex/`, and optional **Python bindings** (`import phepex`).

The C++ library depends only on the C++ standard library, so it drops into any C++ project. `ctapipe` is used only by the Python benchmark/extractor layer, never by the library itself.
<!-- intro:end -->

## Documentation

Full API documentation covering **both** the C++ and Python surfaces is published to GitHub Pages: **https://phepex.github.io/phepex/** (built from source by the `docs` workflow on every push to `main`). Build it locally with:

```bash
pip install .[bench] && pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html      # open docs/_build/html/index.html
```

## C++ library

<!-- quickstart-cpp:start -->
All kernels are free functions in namespace `phepex`, operating on caller-owned buffers (raw pointers + sizes). See `include/phepex/*.hpp`; umbrella header `phepex/phepex.hpp`.

```cpp
#include <phepex/phepex.hpp>
// phepex::preprocess_waveform(wf, n_samples, upsampling, pole_zero, smoothing, offset, scale, out);
// phepex::pos_soft_clip / neighbor_peak_indices / extract_around_peak / adaptive_centroid / generate_waveforms
```

Build & install:

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/your/prefix
cmake --build build -j
cmake --install build      # headers, libphepex.{so,a}, CMake package, pkg-config
```

Consume it — **CMake** (`find_package`):

```cmake
find_package(phepex REQUIRED)
target_link_libraries(app PRIVATE phepex::phepex)        # or phepex::phepex_static
```

or **pkg-config** (Makefile-based projects):

```bash
g++ -std=c++17 app.cpp $(pkg-config --cflags --libs phepex)
```

A runnable example is in `examples/example.cpp` (build with `-DPHEPEX_BUILD_EXAMPLES=ON`).
<!-- quickstart-cpp:end -->

## Build options

CMake cache options, passed as `-D<name>=<value>`. Defaults are chosen for a plain library
build; the Python wheel (built via scikit-build-core, which defines `SKBUILD`) enables the
bindings automatically.

| Option | Default | Effect |
| --- | --- | --- |
| `PHEPEX_BUILD_PYTHON` | `ON` under scikit-build-core, else `OFF` | Build the nanobind Python module `phepex._core`. |
| `PHEPEX_BUILD_EXAMPLES` | `OFF` | Build the C++ example (`examples/example.cpp`). |
| `PHEPEX_BUILD_TESTS` | `OFF` | Build the standalone C++ unit tests (vendored Catch2). |
| `PHEPEX_BUILD_BENCHMARKS` | `OFF` | Build the C++ per-kernel micro-benchmark. |
| `PHEPEX_NEIGHBOR_PAIRWISE_SUM` | `ON` | Accumulate the neighbour sum in `neighbor_peak_indices` pairwise (up to 25% faster depending on target). `OFF` selects the sequential sum. See the note in `include/phepex/neighbor.hpp`. |
| `PHEPEX_PREPROCESS_TILE_WIDTH` | `24` | Number of waveforms mapped onto SIMD lanes per tile in the batched `preprocess_waveforms` path. Bit-identical for any value `>= 1`; performance only. See the tuning notes below. |

### Tuning `PHEPEX_PREPROCESS_TILE_WIDTH`

`preprocess_waveforms` processes waveforms in tiles of this width, one waveform per SIMD lane,
to fill the loop-carried recurrences (the upsampling running sums and the Deriche smoothing
IIR) with independent rows — these are latency-bound, so per-row throughput rises as more
independent rows are in flight. The width affects only the tiled paths (`upsampling > 1` and/or
smoothing enabled); the `upsampling == 1`, no-smoothing case runs per-row scalar and is
unaffected.

Recommendations:

- Larger widths expose more independent rows to the out-of-order window but grow the
  working set: about `3 * width * upsampling * n_samples` floats are live per tile. Keep
  that inside L1; above it the tiles spill and the gain reverses.
- Build the micro-benchmark (`-DPHEPEX_BUILD_BENCHMARKS=ON
  -DCMAKE_CXX_FLAGS="-march=native" -DPHEPEX_PREPROCESS_TILE_WIDTH=K`) at a few widths and
  compare the `preprocess up/pz(/smoothing) batch` rows
- `24` (the default) works best for Apple Silicon (NEON) and Zen 4 architectures, even though
  the latter supports AVX-512 (float SIMD width of 16).

To obtain an optimised Python wheel, pass it through scikit-build-core, e.g.
`CMAKE_CXX_FLAGS="-march=native" pip install . -C cmake.define.PHEPEX_PREPROCESS_TILE_WIDTH=K`.

## Python bindings

<!-- quickstart-python:start -->
```bash
pip install .            # builds the phepex wheel (scikit-build-core + nanobind)
```

```python
import phepex
# phepex.deconvolve, phepex.pos_soft_clip, phepex.neighbor_peak_indices,
# phepex.extract_around_peak, phepex.adaptive_centroid, phepex.generate_waveforms
```

`import phepex` pulls only numpy + the compiled extension. The ctapipe-integrated `phepex.extractor.FastFlashCamExtractor` (a drop-in accelerating ctapipe's `FlashCamExtractor`) is a separate submodule imported on demand.
<!-- quickstart-python:end -->

## Layout

```
include/phepex/   public headers (phepex:: API)
src/              libphepex sources (+ bindings.cpp for phepex._core)
python/phepex/    Python package: kernels.py (numpy wrappers), extractor.py (ctapipe)
examples/         C++ usage example
benchmarks/       comparative benchmark (stock ctapipe vs FastFlashCamExtractor)
benchmarks/cpp/   standalone C++ per-kernel micro-benchmark (no Python/ctapipe)
benchmarks/flashcam-config.txt  frozen FlashCam configuration for the micro-benchmark
scripts/          maintenance scripts (e.g. export-camera-config.py)
tests/            Python equivalence / bit-exactness tests (pytest)
tests/cpp/        standalone C++ unit tests (Catch2, no Python/ctapipe)
third_party/catch2/  vendored Catch2 v3 (amalgamated) for the C++ tests
```

## C++ unit tests (no Python/ctapipe)

`tests/cpp/test_phepex.cpp` covers every `phepex::` function with self-verifying
hand-computed values / invariants, using the vendored **Catch2 v3** (amalgamated, in
`third_party/catch2/`, so no network or extra dependency). Build and run standalone:

```bash
cmake -S . -B build -DPHEPEX_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure     # or: ./build/phepex_tests
```

## C++ micro-benchmark (no Python/ctapipe)

`benchmarks/cpp/microbench.cpp` times each `phepex::` kernel in isolation on a realistic
camera configuration, so a regression can be attributed to a specific kernel. It links only
`libphepex` and reads the configuration at run time from a text file, so it needs neither
Python nor ctapipe. One "op" is one full-camera sweep (the per-event cost). The input is one
artificial event — a skewed-Gaussian shower image over ~200 pixels at ~1–100 p.e. with a
~1 ns/pixel time gradient, plus 200 MHz NSB on a 200 LSB pedestal, digitised to 12-bit ADC
counts — because the peak-search and centroid kernels are data-dependent. The image is sized
in pixel pitches, so a config for another camera gives a comparable event without code
changes.

```bash
cmake -S . -B build -DPHEPEX_BUILD_BENCHMARKS=ON -DCMAKE_CXX_FLAGS="-march=native"
cmake --build build -j
./build/phepex-microbench                        # default config, 200 reps
./build/phepex-microbench --config PATH --reps N
```

The bundled `benchmarks/flashcam-config.txt` holds the FlashCam configuration (1764 pixels,
neighbour adjacency in CSR form, pixel coordinates and area, reference pulse, readout
scalars) exported from ctapipe.
Regenerate it, or export another camera, with:

```bash
python3 scripts/export-camera-config.py --camera FlashCam \
    --out benchmarks/flashcam-config.txt
```

## Benchmark

<!-- benchmark:start -->
`benchmarks/benchmark-fast-extractor.py` compares stock ctapipe `FlashCamExtractor` against `phepex.extractor.FastFlashCamExtractor` on synthetic gamma-like FlashCam-MST events (needs the `bench` extra: `pip install .[bench]`):

```bash
python3 benchmarks/benchmark-fast-extractor.py --events 5000
```

Typical result: **~5× faster** end-to-end (leading-edge timing on), with bit-exact / ~1e-7 agreement on signal pixels. Tests: `python3 -m pytest tests/ -q`.
<!-- benchmark:end -->

## Notes

- Single-threaded by design: parallelise at the event level (the caller's job). No OpenMP (performance boost is sub-linear).
- ctapipe (and thus numba/scipy/astropy) is only needed for the benchmark and the `FastFlashCamExtractor` Python wrapper, not for `libphepex` or `import phepex`.

## License

phepex is licensed under the **Mozilla Public License 2.0** — see [`LICENSE`](LICENSE).
Copyright © 2026 Max-Planck-Institut für Kernphysik.

Vendored third-party code under `third_party/catch2/` (used only by the C++ unit tests) is
Catch2, distributed under the Boost Software License 1.0.
