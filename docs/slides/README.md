# Introductory slide deck

`phepex-intro.typ` is an 8-slide deck introducing phepex: its relationship to libdvr and
ctapipe, the kernels it exposes, and the v0.2 benchmark figures. Written in
[Typst](https://typst.app); the page is 16×9 in, so the PDF projects at any 16:9 resolution.

Assets are resolved relative to the source file — currently `phepex-logo-light.svg`, used on
the title slide.

## Compile

```bash
typst compile docs/slides/phepex-intro.typ  # -> phepex-intro.pdf
```

Install the compiler from <https://github.com/typst/typst/releases>, or via a package
manager (`brew install typst`, `cargo install --locked typst-cli`, `snap install typst`).

## Fonts

The deck sets **Fira Sans** (body) and **Fira Mono** (code). Neither is bundled with Typst.
If they are not installed, Typst substitutes Libertinus Serif **silently** — no error, no
warning — and the deck renders in a serif face throughout. Verify before distributing:

```bash
typst fonts | grep -i fira        # expect "Fira Sans" and "Fira Mono"
```

Fetch them from [Mozilla's Fira release](https://github.com/mozilla/Fira) and
either install them system-wide or point Typst at the directory:

```bash
typst compile --font-path /path/to/fira/otf phepex-intro.typ
```

The SVG export traces glyphs to outlines, so fonts affect the build only, not a reader of
the published deck.

## In the documentation site

`docs/conf.py` compiles the deck on every Sphinx build — the PDF beside this file, and one
SVG per slide into `docs/_static/slides/` for the viewer on `docs/slides.md` (both
git-ignored). It uses the [`typst` PyPI package](https://pypi.org/project/typst/), the same
compiler as a library, so the docs need no system Typst; that wheel ships no `typst`
executable, so install the CLI separately for the commands above. A missing package or a
failed compile is reported and skipped rather than fatal.

**Adding or removing a slide requires editing the outline list in `docs/slides.md`.** It is
the deck's text alternative — SVG carries no readable text — and supplies each slide's alt
text and the viewer's slide count. `conf.py` fails the build when the list and the deck
disagree on length.

| Variable | Effect |
| --- | --- |
| `PHEPEX_SKIP_SLIDES=1` | Skip compiling; use whatever artefacts are already present. |
| `TYPST_FONT_PATHS` | `os.pathsep`-separated font directories, as for `--font-path`. The docs workflow sets this so the published deck uses Fira. |
