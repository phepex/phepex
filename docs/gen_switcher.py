#!/usr/bin/env python3
"""Generate the root files for the multi-version documentation site.

The docs are published one subdirectory per version on the gh-pages branch:
``dev/`` (main), ``latest/`` (copy of the newest tag), and ``vX.Y.Z/`` per tag.
This script scans a target directory for those version subdirectories, refreshes
``latest/`` to match the newest tag, and writes, into that same directory:

- ``switcher.json`` -- the list consumed by docs/_static/version-switcher.js,
  newest-first: ``latest``, ``dev``, then each ``vX.Y.Z`` tag in descending
  semantic-version order.
- ``index.html`` -- a meta-refresh redirect to ``latest/`` (or ``dev/``, or the
  newest tag, in that order of preference; a non-redirecting placeholder if the
  site is empty).
- ``.nojekyll`` -- empty marker so GitHub Pages serves ``_static`` and other
  underscore-prefixed paths.

``latest/`` is derived here (not written by CI) as a copy of the highest
``vX.Y.Z`` tag directory present, so it tracks the newest release regardless of
the order tags were pushed. Entries are derived from the directories actually
present, so no entry can point at an unpublished version. ``switcher.json``
"version" fields equal the subdirectory names, which equal the
``PHEPEX_DOC_VERSION`` slug each build was produced with; version-switcher.js
matches on that field.

Usage: ``python docs/gen_switcher.py <site-dir> [--base-url URL]``
The default base URL is the phepex GitHub Pages site.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

DEFAULT_BASE_URL = "https://phepex.github.io/phepex"

# Match version subdirectories of the form v<major>.<minor>.<patch>.
_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def _discover_versions(site: Path) -> tuple[list[str], bool, bool]:
    """Return (sorted tag dir names, has_dev, has_latest).

    Tags are sorted by (major, minor, patch) descending. Only immediate
    subdirectories are considered; files and unrelated directories are ignored.
    """
    tags: list[tuple[tuple[int, int, int], str]] = []
    has_dev = (site / "dev").is_dir()
    has_latest = (site / "latest").is_dir()
    for child in site.iterdir():
        if not child.is_dir():
            continue
        m = _TAG_RE.match(child.name)
        if m:
            tags.append(((int(m[1]), int(m[2]), int(m[3])), child.name))
    tags.sort(key=lambda item: item[0], reverse=True)
    return [name for _, name in tags], has_dev, has_latest


def _newest_tag(site: Path) -> str | None:
    """Return the highest vX.Y.Z tag directory name, or None if none exist."""
    tag_names, _, _ = _discover_versions(site)
    return tag_names[0] if tag_names else None


def refresh_latest(site: Path) -> str | None:
    """Rebuild ``site/latest`` as a copy of the newest vX.Y.Z tag directory.

    ``latest/`` is derived here rather than written by CI, so it always tracks the
    highest semantic-version tag regardless of the order tags were pushed (a
    backport tag older than the current newest does not move it). Returns the tag
    name copied, or None when no vX.Y.Z tag directory exists (any stale
    ``latest/`` is removed in that case).

    The copied pages are byte-identical to the tag build, which was produced with
    ``PHEPEX_DOC_VERSION=<tag>``; their ``data-current-version`` attribute is
    rewritten from the tag slug to ``"latest"`` so version-switcher.js preselects
    the ``latest`` entry (not the frozen tag entry) when serving ``/latest/``.
    Only the exact attribute string ``data-current-version="<tag>"`` is replaced;
    other occurrences of the slug (e.g. in the page title) are left intact.
    """
    latest = site / "latest"
    if latest.is_dir():
        shutil.rmtree(latest)
    elif latest.exists():
        latest.unlink()

    newest = _newest_tag(site)
    if newest is None:
        return None

    shutil.copytree(site / newest, latest)

    old = f'data-current-version="{newest}"'
    new = 'data-current-version="latest"'
    for html in latest.rglob("*.html"):
        text = html.read_text(encoding="utf-8")
        if old in text:
            html.write_text(text.replace(old, new), encoding="utf-8")
    return newest


def build_switcher(site: Path, base_url: str) -> list[dict[str, object]]:
    """Build the switcher.json entry list from the directories present in ``site``."""
    base = base_url.rstrip("/")
    tag_names, has_dev, has_latest = _discover_versions(site)

    entries: list[dict[str, object]] = []
    if has_latest:
        entries.append(
            {
                "name": "latest",
                "version": "latest",
                "url": f"{base}/latest/",
            }
        )
    if has_dev:
        entries.append({"name": "dev", "version": "dev", "url": f"{base}/dev/"})
    for name in tag_names:
        entries.append({"name": name, "version": name, "url": f"{base}/{name}/"})
    return entries


def _redirect_target(site: Path) -> str | None:
    """Choose the landing subdirectory: latest, else dev, else newest tag, else None.

    Returns None only when no version directory exists. It never returns ``"."``
    (the site root), which would make the root index.html meta-refresh to itself
    and loop.
    """
    if (site / "latest").is_dir():
        return "latest/"
    if (site / "dev").is_dir():
        return "dev/"
    newest = _newest_tag(site)
    if newest is not None:
        return f"{newest}/"
    return None


def _write_index(site: Path, target: str | None) -> None:
    if target is None:
        # No version directory present. Write a non-redirecting placeholder rather
        # than a refresh to the root, which would reload this page indefinitely.
        html = (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            '  <meta charset="utf-8">\n'
            "  <title>phepex documentation</title>\n"
            "</head>\n"
            "<body>\n"
            "  <p>No documentation has been published yet.</p>\n"
            "</body>\n"
            "</html>\n"
        )
    else:
        html = (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            '  <meta charset="utf-8">\n'
            f'  <meta http-equiv="refresh" content="0; url={target}">\n'
            f'  <link rel="canonical" href="{target}">\n'
            "  <title>phepex documentation</title>\n"
            "</head>\n"
            "<body>\n"
            f'  <p>Redirecting to <a href="{target}">the documentation</a>.</p>\n'
            "</body>\n"
            "</html>\n"
        )
    (site / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path, help="directory holding the version subdirs")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"public base URL of the docs site (default: {DEFAULT_BASE_URL})",
    )
    args = parser.parse_args()

    site: Path = args.site
    if not site.is_dir():
        parser.error(f"{site} is not a directory")

    refresh_latest(site)
    entries = build_switcher(site, args.base_url)
    (site / "switcher.json").write_text(
        json.dumps(entries, indent=2) + "\n", encoding="utf-8"
    )
    _write_index(site, _redirect_target(site))
    (site / ".nojekyll").write_text("", encoding="utf-8")

    print(
        f"wrote switcher.json ({len(entries)} entries), index.html, .nojekyll to {site}"
    )


if __name__ == "__main__":
    main()
