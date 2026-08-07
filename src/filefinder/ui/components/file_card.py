"""File result card component.

Displays a single :class:`~filefinder.core.search.SearchResult` with an
extension-based icon, filename, truncated path, snippet, score badge and an
"Open" button that opens the file with the OS default application.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from typing import Any, Callable, Optional

import flet as ft

logger = logging.getLogger(__name__)

# Extension -> icon mapping (Material icons via ft.Icons).
_ICON_BY_EXT: dict[str, str] = {
    ".pdf": ft.Icons.PICTURE_AS_PDF,
    ".docx": ft.Icons.DESCRIPTION,
    ".doc": ft.Icons.DESCRIPTION,
    ".xlsx": ft.Icons.TABLE_CHART,
    ".xls": ft.Icons.TABLE_CHART,
    ".pptx": ft.Icons.SLIDESHOW,
    ".ppt": ft.Icons.SLIDESHOW,
    ".txt": ft.Icons.TEXT_SNIPPET,
    ".md": ft.Icons.TEXT_SNIPPET,
    ".jpg": ft.Icons.IMAGE,
    ".jpeg": ft.Icons.IMAGE,
    ".png": ft.Icons.IMAGE,
    ".bmp": ft.Icons.IMAGE,
}
_DEFAULT_ICON = ft.Icons.INSERT_DRIVE_FILE

_MAX_PATH_LEN = 60


def truncate_middle(text: str, max_len: int = _MAX_PATH_LEN) -> str:
    """Truncate ``text`` in the middle with an ellipsis.

    Args:
        text: the string to shorten.
        max_len: maximum resulting length.

    Returns:
        The original string if short enough, otherwise ``head…tail``.
    """
    if len(text) <= max_len:
        return text
    keep = max_len - 1
    head = keep // 2
    tail = keep - head
    return f"{text[:head]}…{text[-tail:]}"


def open_file(path: str) -> None:
    """Open ``path`` with the platform default application.

    Uses ``os.startfile`` on Windows, ``open`` on macOS and ``xdg-open`` on
    Linux. Runs blocking; call from a worker thread.

    Args:
        path: absolute file path.

    Raises:
        OSError: if the file cannot be opened.
    """
    if sys.platform.startswith("win") and hasattr(os, "startfile"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class FileCard(ft.Card):
    """Card representing one search result.

    Args:
        result: a ``SearchResult``-like object with ``path``, ``filename``,
            ``extension``, ``snippet`` and ``score`` attributes (dataclass or
            dict).
        open_label: localized label for the open button.
        on_error: optional callback invoked with an error message when the
            file cannot be opened.
    """

    def __init__(
        self,
        result: Any,
        open_label: str = "Open",
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._on_error = on_error
        if isinstance(result, dict):
            self._path = str(result.get("path", ""))
            filename = str(result.get("filename", ""))
            extension = str(result.get("extension", ""))
            snippet = str(result.get("snippet", "") or "")
            score = float(result.get("score", 0.0) or 0.0)
        else:
            self._path = str(getattr(result, "path", ""))
            filename = str(getattr(result, "filename", ""))
            extension = str(getattr(result, "extension", ""))
            snippet = str(getattr(result, "snippet", "") or "")
            score = float(getattr(result, "score", 0.0) or 0.0)

        icon = ft.Icon(
            _ICON_BY_EXT.get(extension.lower(), _DEFAULT_ICON),
            size=32,
            color=ft.Colors.PRIMARY,
        )
        score_badge = ft.Container(
            content=ft.Text(
                f"{max(0.0, min(score, 1.0)) * 100:.0f}%",
                size=12,
                weight=ft.FontWeight.BOLD,
            ),
            bgcolor=ft.Colors.SECONDARY_CONTAINER,
            border_radius=ft.BorderRadius.all(12),
            padding=ft.Padding.symmetric(horizontal=10, vertical=4),
        )
        open_button = ft.TextButton(
            content=ft.Row(
                [ft.Icon(ft.Icons.OPEN_IN_NEW, size=16), ft.Text(open_label)],
                spacing=4,
                tight=True,
            ),
            on_click=self._on_open_click,
        )

        super().__init__(
            content=ft.Container(
                content=ft.Row(
                    [
                        icon,
                        ft.Column(
                            [
                                ft.Row(
                                    [
                                        ft.Text(
                                            filename,
                                            weight=ft.FontWeight.BOLD,
                                            size=15,
                                            expand=True,
                                            max_lines=1,
                                            overflow=ft.TextOverflow.ELLIPSIS,
                                        ),
                                        score_badge,
                                    ],
                                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                                ),
                                ft.Text(
                                    truncate_middle(self._path),
                                    size=12,
                                    color=ft.Colors.OUTLINE,
                                    max_lines=1,
                                    overflow=ft.TextOverflow.ELLIPSIS,
                                ),
                                ft.Text(
                                    snippet,
                                    size=13,
                                    max_lines=3,
                                    overflow=ft.TextOverflow.ELLIPSIS,
                                ),
                            ],
                            spacing=4,
                            expand=True,
                        ),
                        open_button,
                    ],
                    spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                padding=12,
            ),
            elevation=1,
        )

    async def _on_open_click(self, e: ft.ControlEvent) -> None:
        """Open the file in a worker thread and report failures."""
        try:
            await asyncio.to_thread(open_file, self._path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to open %s: %s", self._path, exc)
            if self._on_error is not None:
                self._on_error(str(exc))
