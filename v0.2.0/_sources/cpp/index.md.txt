# C++ API

The public `phepex::` kernels, grouped by header. Include everything via the umbrella header
`<phepex/phepex.hpp>`, or a single header for one kernel. Each function's description below is
rendered from its own header documentation.

## Waveform preprocessing

Declared in `<phepex/preprocess.hpp>`. Single-waveform upsampling + pole-zero deconvolution
with optional Deriche (1992) Gaussian smoothing, plus its DVR-convention valid range.
Pole-zero deconvolution without smoothing is just `preprocess_waveform` with
`smoothing == nullptr`. `preprocess_waveforms` applies the same operation to a batch of rows
in one call; it tiles the rows (one row per SIMD lane) to fill the loop-carried recurrence
whenever one is present (the upsampling running sums and/or the Deriche IIR), and its output
is bit-identical to the per-row form.

```{doxygenstruct} phepex::SampleRange
:members:
```

```{doxygenstruct} phepex::SmoothingCoefficients
:members:
```

```{doxygenfunction} phepex::calculate_smoothing_coefficients
```

```{doxygenfunction} phepex::preprocess_waveform(const std::uint16_t *src, int n_samples, int upsampling, float pole_zero, const SmoothingCoefficients *smoothing, float offset, float scale, float *out, float *scratch = nullptr);
```

```{doxygenfunction} phepex::preprocess_waveform(const float *src, int n_samples, int upsampling, float pole_zero, const SmoothingCoefficients *smoothing, float offset, float scale, float *out, float *scratch = nullptr);
```

```{doxygenfunction} phepex::preprocess_waveforms(const std::uint16_t *src, int n_rows, int n_samples, int upsampling, const float *pole_zero, std::ptrdiff_t pole_zero_stride, const SmoothingCoefficients *smoothing, const float *offset, std::ptrdiff_t offset_stride, const float *scale, std::ptrdiff_t scale_stride, float *out, float *scratch = nullptr);
```

```{doxygenfunction} phepex::preprocess_waveforms(const float *src, int n_rows, int n_samples, int upsampling, const float *pole_zero, std::ptrdiff_t pole_zero_stride, const SmoothingCoefficients *smoothing, const float *offset, std::ptrdiff_t offset_stride, const float *scale, std::ptrdiff_t scale_stride, float *out, float *scratch = nullptr);
```

```{doxygenfunction} phepex::preprocess_waveforms_scratch_size
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

```{doxygenfunction} phepex::generate_shower_image
```

```{doxygenstruct} phepex::ShowerModel
:members:
```
