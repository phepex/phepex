# Configuration file for the Sphinx documentation builder.
#
# phepex docs: one Sphinx build covering both the C++ API (via Doxygen XML + Breathe) and
# the Python API (via autodoc). Doxygen is invoked from here so a single `sphinx-build`
# regenerates everything.

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from sphinx.util import logging as sphinx_logging

HERE = Path(__file__).parent.resolve()
REPO = HERE.parent
logger = sphinx_logging.getLogger(__name__)

# Single-source the published-docs base URL: gen_switcher.py owns the one copy
# (it also stamps it into switcher.json entries), so a site move is a one-file edit.
sys.path.insert(0, str(HERE))
from gen_switcher import DEFAULT_BASE_URL  # noqa: E402

# Project information
project = "phepex"
author = "Max-Planck-Institut für Kernphysik"
copyright = "2026, Max-Planck-Institut für Kernphysik"

# Single-source the version from the installed package (also required for autodoc).
try:
    import phepex

    release = phepex.__version__
except Exception:  # pragma: no cover - fallback when the package is not importable
    from importlib.metadata import PackageNotFoundError, version

    try:
        release = version("phepex")
    except PackageNotFoundError:
        release = "0.0.0"
version = ".".join(release.split(".")[:2])

# Run Doxygen (XML only) so Breathe has fresh input
DOXY_INPUT = REPO / "include" / "phepex"
DOXY_XML = HERE / "_doxygen" / "xml"


def _run_doxygen() -> None:
    """Render Doxyfile.in -> Doxyfile with resolved paths, then run doxygen."""
    template = (HERE / "Doxyfile.in").read_text()
    rendered = (
        template.replace("@DOXYGEN_INPUT@", str(DOXY_INPUT))
        .replace("@DOXYGEN_XML_OUT@", str(DOXY_XML))
        .replace("@DOXYGEN_VERSION@", release)
    )
    (HERE / "Doxyfile").write_text(rendered)
    DOXY_XML.mkdir(parents=True, exist_ok=True)
    subprocess.run(["doxygen", "Doxyfile"], cwd=HERE, check=True)


# Skip only if explicitly asked (e.g. a fast Python-only local rebuild).
if os.environ.get("PHEPEX_SKIP_DOXYGEN") != "1":
    _run_doxygen()

# Compile the Typst slide deck. The output sits next to its source, under the name a plain
# `typst compile phepex-intro.typ` would produce, so the docs build and a manual build
# share one artefact. docs/slides.md links it; Sphinx resolves that link to a download and
# copies the file into the site, so the PDF is published from here, not via _static.
# See docs/slides/README.md for the deck and its font requirements.
SLIDES_SRC = HERE / "slides" / "phepex-intro.typ"
SLIDES_PDF = SLIDES_SRC.with_suffix(".pdf")

# One SVG per slide, for the in-page viewer on docs/slides.md. These go to _static (copied
# verbatim, unlike the download-resolved PDF) because the viewer references them from an
# <img src> that Sphinx does not rewrite.
#
# One file per slide rather than all eight inlined into the page: Typst emits every glyph
# as a <symbol id="...">, and those ids repeat across slides (45 of slide 2's 118 also
# occur in slide 3), so inlining would put duplicate ids in one document and leave each
# <use> resolving unpredictably. Separate files keep each in its own id scope, and let the
# browser fetch and cache them one at a time.
SLIDES_SVG_DIR = HERE / "_static" / "slides"
SLIDES_SVG_STEM = "slide"


def _count_svg() -> int:
    return len(list(SLIDES_SVG_DIR.glob(f"{SLIDES_SVG_STEM}-*.svg")))


def _build_slides() -> tuple[bool, int]:
    """Compile the Typst deck to PDF + per-slide SVG; (PDF available, number of slides).

    Uses the `typst` PyPI package (the compiler as a library) so the docs build needs no
    system Typst install. A missing package or a compile error is reported and skipped
    rather than fatal -- the deck is supplementary and the API pages must still build --
    but artefacts already present (a manual or earlier build) are still used.

    Fira Sans/Fira Mono are not bundled with Typst, and Typst substitutes a serif face
    silently when they are absent, so TYPST_FONT_PATHS (os.pathsep-separated, as for the
    Typst CLI) is forwarded for callers that supply the fonts out of tree. The SVG export
    traces glyphs to outlines, so the fonts matter only here, never to a reader.
    """
    if os.environ.get("PHEPEX_SKIP_SLIDES") == "1":
        return SLIDES_PDF.is_file(), _count_svg()
    try:
        import typst
    except ImportError:
        print("conf.py: typst not installed, skipping the slide deck")
        return SLIDES_PDF.is_file(), _count_svg()

    env_paths = os.environ.get("TYPST_FONT_PATHS", "").split(os.pathsep)
    font_paths = [p for p in env_paths if p]
    # Stale SVGs would otherwise outlive a slide being deleted from the deck.
    for old in SLIDES_SVG_DIR.glob(f"{SLIDES_SVG_STEM}-*.svg"):
        old.unlink()
    SLIDES_SVG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        typst.compile(SLIDES_SRC, output=SLIDES_PDF, font_paths=font_paths)
        typst.compile(
            SLIDES_SRC,
            output=SLIDES_SVG_DIR / f"{SLIDES_SVG_STEM}-{{n}}.svg",
            format="svg",
            font_paths=font_paths,
        )
    except Exception as exc:  # a broken deck must not fail the API documentation
        print(f"conf.py: slide deck compile failed, skipping: {exc}")
        return SLIDES_PDF.is_file(), _count_svg()
    return True, _count_svg()


slides_available, slides_count = _build_slides()


def _check_slide_outline(app, env, docnames) -> None:
    """Warn when docs/slides.md's outline list and the deck disagree on slide count.

    The outline is the deck's text alternative and the source of its per-slide alt text
    (the SVG export leaves no readable text), so a slide added to the deck without a
    matching entry would silently go undescribed. Only checked when the deck was actually
    exported; with -W in CI a mismatch fails the build.
    """
    if not slides_count:
        return
    page = (HERE / "slides.md").read_text()
    outline = re.search(r'class="phepex-deck-outline"(.*?)</ol>', page, re.DOTALL)
    entries = len(re.findall(r"<li>", outline.group(1))) if outline else 0
    if entries != slides_count:
        logger.warning(
            "docs/slides.md: the slide outline has %d entries but the deck has %d "
            "slides; the viewer takes its alt text from that list",
            entries,
            slides_count,
        )


def setup(app):
    app.connect("env-before-read-docs", _check_slide_outline)


# General configuration
extensions = [
    "myst_parser",
    "breathe",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
myst_enable_extensions = ["colon_fence", "deflist", "substitution"]
# slides/README.md documents building the deck for its authors; it is not a site page
# (docs/slides.md is), so keep it out of the toctree to avoid an "not included" warning.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "slides/README.md"]

# Resolves to the deck download link, or to a note when the deck was not compiled, so a
# build without Typst produces no dead link (the CI build runs with -W). The viewer needs
# no such substitution: MyST does not expand these inside a raw HTML block, so
# slides-viewer.js takes the slide count from the page's outline list and detects a
# missing export from the first image failing to load.
myst_substitutions = {
    "slides_download": (
        "[Download the introduction slides (PDF)](slides/phepex-intro.pdf)"
        if slides_available
        else "The PDF was not built (Typst unavailable); see `docs/slides/README.md`."
    ),
}

# Breathe (C++)
breathe_projects = {"phepex": str(DOXY_XML)}
breathe_default_project = "phepex"
breathe_default_members = ()  # list members explicitly on the pages

# autodoc / napoleon (Python)
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# HTML output
html_theme = "furo"
html_title = f"phepex {release} Documentation"
html_static_path = ["_static"]
templates_path = ["_templates"]
# Loaded site-wide (as the switcher is); slides-viewer.js returns immediately on any page
# without a deck, and the pair is under 2 KB.
html_css_files = ["version-switcher.css", "slides-viewer.css"]
html_js_files = ["version-switcher.js", "slides-viewer.js"]
html_theme_options = {
    "light_logo": "phepex-icon-light.svg",
    "dark_logo": "phepex-icon-dark.svg",
    "light_css_variables": {
        "color-brand-primary": "#7C4DFF",
        "color-brand-content": "#7C4DFF",
    },
    "dark_css_variables": {
        "color-brand-primary": "#9C86DE",
        "color-brand-content": "#9C86DE",
    },
}

# Version switcher (furo has no built-in switcher).
#
# The site is published one subdirectory per version on the gh-pages branch
# (/dev/, /latest/, /vX.Y.Z/). A single switcher.json at the site root lists the
# published versions; docs/_static/version-switcher.js fetches it at runtime and
# fills the <select> injected by docs/_templates/sidebar/version-switcher.html.
#
# phepex_current_version identifies which entry of switcher.json this build is.
# CI sets PHEPEX_DOC_VERSION to the slug it publishes under ("dev" or the tag);
# it must equal the "version" field of the matching switcher.json entry. Without
# the env var (local builds) it falls back to the setuptools_scm release, which
# will not match a switcher entry — the switcher then shows no current selection,
# which is the intended local-build behaviour.
html_context = {
    "phepex_current_version": os.environ.get("PHEPEX_DOC_VERSION", release),
    "phepex_switcher_url": f"{DEFAULT_BASE_URL}/switcher.json",
}

# Place the switcher last, after scroll-end, so it renders at the bottom of the
# sidebar (outside furo's flex-grow .sidebar-scroll region, pinned below the nav
# tree). The remaining entries are furo's defaults (omitting an entry removes that
# sidebar component). The switcher is server-rendered hidden and revealed by JS
# only after switcher.json loads; placing it at the bottom keeps that late reveal
# from moving the nav tree or the main content column.
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
        "sidebar/version-switcher.html",
    ]
}
