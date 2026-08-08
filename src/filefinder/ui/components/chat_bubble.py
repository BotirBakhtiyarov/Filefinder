"""Chat message bubble component.

User messages are right-aligned on an accent container; assistant messages
are left-aligned on a surface-variant container. Text is selectable.
Assistant bubbles support an optional collapsible "thinking" section and
incremental (streaming) text updates.
"""

from __future__ import annotations

import logging

import flet as ft

logger = logging.getLogger(__name__)

_MAX_BUBBLE_WIDTH = 560


def _estimate_width(text: str) -> int:
    """Rough pixel width estimate so short messages get narrow bubbles."""
    longest_line = max((len(line) for line in text.splitlines()), default=1)
    return max(80, min(_MAX_BUBBLE_WIDTH, 32 + longest_line * 8))


def split_thinking(text: str) -> tuple[str, str]:
    """Split stored assistant text into ``(thinking, answer)``.

    Persisted streaming replies embed the reasoning as a leading
    ``<think>…</think>`` block; returns ``("", text)`` when absent.
    """
    text = text or ""
    if text.startswith("<think>"):
        end = text.find("</think>")
        if end >= 0:
            thinking = text[len("<think>"):end].strip()
            answer = text[end + len("</think>"):].strip()
            return thinking, answer
    return "", text


class ChatBubble(ft.Row):
    """A single chat message bubble.

    Args:
        role: ``"user"`` or ``"assistant"``.
        text: message content (rendered as selectable text).
        thinking: optional reasoning text shown in a collapsed dropdown
            above the answer (assistant messages only).
        think_label: localized label for the thinking dropdown.
    """

    def __init__(
        self,
        role: str,
        text: str = "",
        thinking: str = "",
        think_label: str = "Thinking",
    ) -> None:
        is_user = role == "user"
        self._think_label = think_label
        self._think_tile: ft.ExpansionTile | None = None
        self._think_text: ft.Text | None = None

        self._text_control = ft.Text(text, selectable=True, size=14)
        controls: list[ft.Control] = []
        if thinking:
            controls.append(self._build_think_tile(thinking))
        controls.append(self._text_control)

        bubble = ft.Container(
            content=ft.Column(controls, spacing=4, tight=True),
            bgcolor=(
                ft.Colors.PRIMARY_CONTAINER if is_user else ft.Colors.SECONDARY_CONTAINER
            ),
            border_radius=ft.BorderRadius.only(
                top_left=14,
                top_right=14,
                bottom_left=14 if not is_user else 2,
                bottom_right=2 if not is_user else 14,
            ),
            padding=ft.Padding.symmetric(horizontal=14, vertical=10),
            width=_estimate_width(text) if text else None,
        )
        super().__init__(
            controls=[bubble],
            alignment=ft.MainAxisAlignment.END if is_user else ft.MainAxisAlignment.START,
        )

    # ------------------------------------------------------------------
    # Thinking dropdown
    # ------------------------------------------------------------------

    def _build_think_tile(self, thinking: str) -> ft.ExpansionTile:
        """Create (once) the collapsible thinking section."""
        self._think_text = ft.Text(
            thinking, selectable=True, size=12, color=ft.Colors.OUTLINE
        )
        self._think_tile = ft.ExpansionTile(
            title=ft.Text(self._think_label, size=12, color=ft.Colors.OUTLINE),
            controls=[self._think_text],
            expanded=False,
            dense=True,
            tile_padding=ft.Padding.all(0),
            controls_padding=ft.Padding.only(left=4, bottom=4),
        )
        return self._think_tile

    # ------------------------------------------------------------------
    # Streaming updates
    # ------------------------------------------------------------------

    def _safe_update(self) -> None:
        try:
            self.update()
        except Exception:  # noqa: BLE001 - not mounted yet
            pass

    def append_answer(self, chunk: str) -> None:
        """Append a streamed answer chunk to the bubble text."""
        self._text_control.value = (self._text_control.value or "") + chunk
        self._safe_update()

    def append_thinking(self, chunk: str) -> None:
        """Append a streamed thinking chunk (creates the tile lazily)."""
        if self._think_tile is None:
            column = self.controls[0].content  # the bubble's ft.Column
            column.controls.insert(0, self._build_think_tile(""))
        self._think_text.value = (self._think_text.value or "") + chunk
        self._safe_update()

    def set_text(self, text: str) -> None:
        """Replace the bubble text (used to clear the typing indicator)."""
        self._text_control.value = text
        self._safe_update()
