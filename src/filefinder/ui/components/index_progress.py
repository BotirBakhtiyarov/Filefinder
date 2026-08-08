"""Indexing progress component.

A linear progress bar plus a status line and a Cancel button, with
``show()``/``hide()`` helpers. Progress updates are pushed from the indexing
worker thread (marshalled via ``page.run_task`` by the caller).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

import flet as ft

logger = logging.getLogger(__name__)


class IndexProgress(ft.Column):
    """Progress bar + status text + cancel button for indexing runs.

    Args:
        on_cancel: optional async callback invoked when Cancel is pressed.
        cancel_label: localized label for the cancel button.
        status_template: format string with ``{current}``, ``{total}`` and
            ``{filename}`` placeholders used for the status line.
    """

    def __init__(
        self,
        on_cancel: Callable[[ft.ControlEvent], Awaitable[None]] | None = None,
        cancel_label: str = "Cancel",
        status_template: str = "{current}/{total} — {filename}",
    ) -> None:
        self._status_template = status_template
        self.bar = ft.ProgressBar(value=0, expand=True)
        self.status = ft.Text("", size=12, color=ft.Colors.OUTLINE, expand=True)
        self.cancel_button = ft.OutlinedButton(
            content=ft.Row(
                [ft.Icon(ft.Icons.CANCEL, size=16), ft.Text(cancel_label)],
                spacing=4,
                tight=True,
            ),
            on_click=on_cancel,
        )
        super().__init__(
            controls=[
                ft.Row([self.bar], vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Row(
                    [self.status, self.cancel_button],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ],
            spacing=4,
            visible=False,
        )

    def show(self) -> None:
        """Make the progress controls visible and reset state."""
        self.visible = True
        self.bar.value = 0
        self.status.value = ""
        self._safe_update()

    def hide(self) -> None:
        """Hide the progress controls."""
        self.visible = False
        self._safe_update()

    def set_progress(self, current: int, total: int, filename: str) -> None:
        """Update bar value and status line.

        Args:
            current: number of files processed so far.
            total: total number of files (0/negative shows indeterminate bar).
            filename: name of the file currently being processed.
        """
        self.bar.value = (current / total) if total > 0 else None
        self.status.value = self._status_template.format(
            current=current, total=total, filename=filename
        )
        self._safe_update()

    def set_status(self, text: str) -> None:
        """Set a free-form status message (keeps the current bar value)."""
        self.status.value = text
        self._safe_update()

    def _safe_update(self) -> None:
        try:
            self.update()
        except Exception as exc:  # noqa: BLE001 - control may not be mounted yet
            logger.debug("IndexProgress.update() skipped: %s", exc)
