# Comparative benchmark: stock ctapipe vs phepex

Compares ctapipe's `FlashCamExtractor` against `phepex.extractor.FastFlashCamExtractor`
(same algorithm, numeric kernels in C++) on a realistic synthetic workload:

- unique gamma-like events for a **1764-pixel FlashCam-MST** camera
- **200 MHz night-sky background** (temporally-distributed Poisson p.e.)
- **22-sample** waveforms, single gain channel, all held in memory

Requires the `bench` extra (ctapipe): `pip install .[bench]`.

## Files

- `generate_events.py` — builds the FlashCam-MST subarray and generates the events. The
  gamma-like charge/time images come from ctapipe's toy models (`SkewedGaussian` +
  `obtain_time_image`); the perf-critical waveform synthesis (pulse convolution + NSB) is
  done by the C++ `phepex.generate_waveforms`.
- `benchmark-fast-extractor.py` — the comparison itself: the isolated deconvolution step
  (ctapipe/scipy vs C++) and the full extractor (stock vs Fast), for leading-edge timing
  on and off.

## Run

```bash
python3 benchmarks/benchmark-fast-extractor.py --events 5000
```

Typical result: **~5× faster** end-to-end with leading-edge timing on (~2700 →
~600 µs/event), **~4.5×** off. The extracted charge/peak_time match stock ctapipe
bit-exactly with no NSB and to ~1e-7 on signal pixels (validated in `tests/`).

## Notes

- The deconvolution (two pole-zero passes/event) dominates the extractor runtime (~50%);
  the phepex kernel is ~7–9× faster than ctapipe's scipy `deconvolve`/`repeat`/`filtfilt`.
