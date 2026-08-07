"""Flet entry point for FileFinder v3.

Run directly with ``python -m filefinder.main`` or via the console script
which calls :func:`run`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import flet as ft

from filefinder.ui.app import FileFinderApp

logger = logging.getLogger(__name__)


def _assets_dir() -> str:
    """Resolve the bundled ``assets/`` folder relative to this file.

    Returns:
        Absolute path to ``<package>/assets`` when it exists, otherwise
        the relative ``"assets"`` fallback expected by ``ft.run``.
    """
    candidate = Path(__file__).resolve().parent / "assets"
    return str(candidate) if candidate.is_dir() else "assets"


async def main(page: ft.Page) -> None:
    """Flet app target: build the FileFinder application on ``page``."""
    app = FileFinderApp()
    await app.init(page)


def run() -> None:
    """Launch the desktop application (console-script entry point)."""
    if hasattr(ft, "run"):  # flet>=0.80 renamed app(target=...)→run(main, ...)
        ft.run(main, assets_dir=_assets_dir())
    else:
        ft.app(target=main, assets_dir=_assets_dir())


if __name__ == "__main__":
    run()
