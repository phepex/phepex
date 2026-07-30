# Comparative benchmark: stock ctapipe vs phepex

Compares ctapipe's `FlashCamExtractor` against `phepex.extractor.FastFlashCamExtractor`
(same algorithm bar one documented divergence, numeric kernels in C++) on a synthetic
workload:

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
~600 µs/event), **~4.5×** off.

Charge agreement with stock ctapipe, measured over 60 events (1764 pixels, upsampling 4,
22 samples); relative deviations are over signal pixels (true charge > 1 p.e.):

| | no NSB | 200 MHz NSB |
| --- | --- | --- |
| signal pixels bit-exact | 40% | 8% |
| relative deviation, median | 7.4e-8 | 1.0e-7 |
| relative deviation, 99th percentile | 2.7e-7 | 2.4e-5 |
| relative deviation, max | 3.2e-7 | O(1) |

Without NSB the deviation is pure float32 rounding, bounded at ~3e-7. With NSB, a small
tail of pixels differs significantly: the neighbour-sum peak search resolves a near-tie
between two local maxima differently, or the true peak falls in an edge margin this
extractor excludes (see the notes below), and the integration window then lands elsewhere.
`tests/test_extractor_equivalence.py` asserts bounds that admit this tail.

## Notes

- The deconvolution (two pole-zero passes/event) dominates the extractor runtime (~50%);
  the phepex kernel is ~7–9× faster than ctapipe's scipy `deconvolve`/`repeat`/`filtfilt`.
- `FastFlashCamExtractor` restricts the clip and neighbour peak search to the non-edge
  deconvolution samples, so a peak inside an edge margin is found by stock ctapipe but not
  here. This is the one intentional algorithmic difference.
