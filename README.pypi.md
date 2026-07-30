# phepex — photo-electron pulse extraction

Compiled numeric kernels for extracting charge and timing from digitised PMT/SiPM
waveforms (e.g. from Cherenkov telescopes or Water Cherenkov Detectors): pole-zero
deconvolution + upsampling, neighbour-sum peak finding, soft clipping, window
integration and leading-edge timing — plus a waveform generator for tests and
benchmarks.

The heavy lifting runs in a compiled extension; `import phepex` needs only numpy.

## Install

```bash
pip install phepex
```

## Usage

```python
import phepex
# phepex.preprocess, phepex.deconvolve, phepex.preprocess_valid_range,
# phepex.deconvolve_valid_range, phepex.pos_soft_clip, phepex.neighbor_peak_indices,
# phepex.extract_around_peak, phepex.adaptive_centroid, phepex.generate_waveforms
```

The ctapipe-integrated `phepex.extractor.FastFlashCamExtractor` — a drop-in
accelerating ctapipe's `FlashCamExtractor` (typically ~5× faster end-to-end, agreeing
on signal-pixel charge to a median relative deviation of ~1e-7) — lives in a separate
submodule imported on demand. It needs the `bench` extra:

```bash
pip install phepex[bench]      # pulls in ctapipe
```

## Documentation

Full API documentation is published to **https://phepex.github.io/phepex/**.

## License

phepex is licensed under the **Mozilla Public License 2.0**.
Copyright © 2026 Max-Planck-Institut für Kernphysik.
