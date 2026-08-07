"""Search bar component: text field + search button."""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

import flet as ft

logger = logging.getLogger(__name__)


class SearchBar(ft.Row):
    """Search input row with a submit-on-enter text field and search button.

    Args:
        on_search: async callback invoked with the current query string when
            the user presses Enter or clicks the search icon.
        hint_text: localized placeholder for the text field.
        button_tooltip: localized tooltip for the search button.
    """

    def __init__(
        self,
        on_search: Callable[[str], Awaitable[None]],
        hint_text: str = "",
        button_tooltip: str = "",
    ) -> None:
        self._on_search = on_search
        self.field = ft.TextField(
            hint_text=hint_text,
            expand=True,
            on_submit=self._on_submit,
            prefix_icon=ft.Icons.SEARCH,
            dense=True,
        )
        self.button = ft.IconButton(
            icon=ft.Icons.SEARCH,
            tooltip=button_tooltip,
            on_click=self._on_click,
        )
        super().__init__(
            controls=[self.field, self.button],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

    @property
    def query(self) -> str:
        """Current text in the search field."""
        return self.field.value or ""

    async def _on_submit(self, e: ft.ControlEvent) -> None:
        await self._on_search(self.query)

    async def _on_click(self, e: ft.ControlEvent) -> None:
        await self._on_search(self.query)
