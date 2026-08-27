# phepex Library Documentation

```{include} ../README.md
:start-after: <!-- intro:start -->
:end-before: <!-- intro:end -->
```

This site documents the installation process and **both** API surfaces:

- [Getting started](getting-started) — how to build, install and consume each side.
- [C++ API](cpp/index) — the `phepex::` kernels (from the public headers).
- [Python API](python/index) — the `phepex` package: numpy kernel wrappers and the
  ctapipe-integrated `FastFlashCamExtractor`.
- [Changelog](changelog) — notable changes per version.

Use the search box (top-left) to jump to any symbol across both languages.

Use the version switcher (bottom-left) to jump to the documentation of a specific
version.

```{toctree}
:hidden:
:maxdepth: 2

getting-started
cpp/index
python/index
changelog
```
