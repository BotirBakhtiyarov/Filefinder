"""Chat message bubble component.

User messages are right-aligned accent bubbles with plain selectable text.
Assistant messages render **Markdown** full-width (ChatGPT-style) and
support an optional collapsible "thinking" section plus incremental
(streaming) updates for both channels.
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
    """A single chat message.

    Args:
        role: ``"user"`` or ``"assistant"``.
        text: message content. Assistant content is rendered as Markdown.
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
        self.is_user = role == "user"
        self._think_label = think_label
        self._think_tile: ft.ExpansionTile | None = None
        self._think_text: ft.Text | None = None

        if self.is_user:
            self._text_control: ft.Control = ft.Text(text, selectable=True, size=14)
            bubble_content: ft.Control = ft.Column(
                [self._text_control], spacing=4, tight=True
            )
            container = ft.Container(
                content=bubble_content,
                bgcolor=ft.Colors.PRIMARY_CONTAINER,
                border_radius=ft.BorderRadius.only(
                    top_left=14, top_right=14, bottom_left=14, bottom_right=2
                ),
                padding=ft.Padding.symmetric(horizontal=14, vertical=10),
                width=_estimate_width(text) if text else None,
            )
            super().__init__(
                controls=[container], alignment=ft.MainAxisAlignment.END
            )
            return

        # --- assistant: full-width Markdown body ----------------------
        self._md = ft.Markdown(
            text,
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            soft_line_break=True,
        )
        controls: list[ft.Control] = []
        if thinking:
            controls.append(self._build_think_tile(thinking))
        controls.append(self._md)

        container = ft.Container(
            content=ft.Column(controls, spacing=4, tight=True),
            padding=ft.Padding.symmetric(horizontal=6, vertical=4),
            expand=True,
        )
        super().__init__(controls=[container], alignment=ft.MainAxisAlignment.START)

    # ------------------------------------------------------------------
    # Thinking dropdown
    # ------------------------------------------------------------------

    def _build_think_tile(self, thinking: str) -> ft.ExpansionTile:
        """Create (once) the collapsible thinking section."""
        self._think_text = ft.Text(
            thinking, selectable=True, size=12, color=ft.Colors.OUTLINE
        )
        self._think_tile = ft.ExpansionTile(
            title=ft.Row(
                [
                    ft.Icon(ft.Icons.PSYCHOLOGY, size=14, color=ft.Colors.OUTLINE),
                    ft.Text(self._think_label, size=12, color=ft.Colors.OUTLINE),
                ],
                spacing=6,
            ),
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
        """Append a streamed answer chunk (Markdown re-renders in place)."""
        if self.is_user:
            self._text_control.value = (self._text_control.value or "") + chunk
        else:
            self._md.value = (self._md.value or "") + chunk
        self._safe_update()

    def append_thinking(self, chunk: str) -> None:
        """Append a streamed thinking chunk (creates the tile lazily)."""
        if self._think_tile is None:
            column = self.controls[0].content  # the container's ft.Column
            column.controls.insert(0, self._build_think_tile(""))
        self._think_text.value = (self._think_text.value or "") + chunk
        self._safe_update()

    def set_text(self, text: str) -> None:
        """Replace the message text (e.g. to clear the typing indicator)."""
        if self.is_user:
            self._text_control.value = text
        else:
            self._md.value = text
        self._safe_update()

    @property
    def current_text(self) -> str:
        """Return the current message text."""
        if self.is_user:
            return str(self._text_control.value or "")
        return str(self._md.value or "")
