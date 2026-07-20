# C++ API

The public `phepex::` kernels, grouped by header. Include everything via the umbrella header
`<phepex/phepex.hpp>`, or a single header for one kernel. Each function's description below is
rendered from its own header documentation.

## Deconvolution & upsampling

Declared in `<phepex/deconvolve.hpp>`.

```{doxygenfunction} phepex::deconvolve_upsample
```

```{doxygenstruct} phepex::SampleRange
:members:
```

```{doxygenfunction} phepex::deconvolve_valid_range
```

## Waveform preprocessing

Declared in `<phepex/preprocess.hpp>`. Single-waveform upsampling + pole-zero deconvolution
with optional Deriche (1992) Gaussian smoothing, plus its DVR-convention valid range.

```{doxygenstruct} phepex::SmoothingCoefficients
:members:
```

```{doxygenfunction} phepex::calculate_smoothing_coefficients
```

```{doxygenfunction} phepex::preprocess_waveform(const std::uint16_t *src, int n_samples, int upsampling, float pole_zero, const SmoothingCoefficients *smoothing, float offset, float scale, float *out, float *scratch = nullptr);
```

```{doxygenfunction} phepex::preprocess_waveform(const float *src, int n_samples, int upsampling, float pole_zero, const SmoothingCoefficients *smoothing, float offset, float scale, float *out, float *scratch = nullptr);
```

```{doxygenfunction} phepex::preprocess_valid_range
```

## Clipping

Declared in `<phepex/clip.hpp>`.

```{doxygenfunction} phepex::pos_soft_clip
```

## Neighbour peak finding

Declared in `<phepex/neighbor.hpp>`.

```{doxygenfunction} phepex::neighbor_peak_indices
```

## Charge & timing extraction

Declared in `<phepex/extract.hpp>`.

```{doxygenfunction} phepex::extract_around_peak
```

```{doxygenfunction} phepex::adaptive_centroid
```

## Waveform generation

Declared in `<phepex/generate.hpp>`.

```{doxygenfunction} phepex::generate_waveforms
```
