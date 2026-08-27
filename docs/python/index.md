# Python API

The `phepex` package: numpy kernel wrappers, the C++ waveform generator, and the
ctapipe-integrated `FastFlashCamExtractor`.

## Kernels (numpy wrappers)

Thin numpy-friendly wrappers over the compiled `phepex._core` kernels; these depend only on
numpy (no ctapipe).

```{eval-rst}
.. automodule:: phepex.kernels
   :members:
```

## Waveform generator

Implemented in C++ and re-exported at the package top level from `phepex._core`.

```{eval-rst}
.. autofunction:: phepex.generate_waveforms
```

## ctapipe extractor

`FastFlashCamExtractor` is a drop-in subclass of ctapipe's `FlashCamExtractor` that runs the
numeric pipeline (deconvolution, clip, peak search, integration, leading-edge timing) through
the phepex C++ kernels. Importing it requires ctapipe (`pip install .[bench]`).

```{eval-rst}
.. autoclass:: phepex.extractor.FastFlashCamExtractor
   :members:
   :special-members: __call__
   :show-inheritance:
```
