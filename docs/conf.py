# Configuration file for the Sphinx documentation builder.
#
# phepex docs: one Sphinx build covering both the C++ API (via Doxygen XML + Breathe) and
# the Python API (via autodoc). Doxygen is invoked from here so a single `sphinx-build`
# regenerates everything.

from __future__ import annotations

import os
import subprocess
from pathlib import Path

HERE = Path(__file__).parent.resolve()
REPO = HERE.parent

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
myst_enable_extensions = ["colon_fence", "deflist"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

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
