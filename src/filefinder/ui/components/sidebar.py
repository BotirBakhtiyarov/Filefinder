"""Navigation sidebar/bottom-bar builder.

Returns a :class:`ft.NavigationRail` for desktop layouts or a
:class:`ft.NavigationBar` for mobile-width layouts.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Sequence

import flet as ft

logger = logging.getLogger(__name__)

_ICONS = [
    ft.Icons.SEARCH,
    ft.Icons.CHAT_BUBBLE_OUTLINE,
    ft.Icons.FOLDER,
    ft.Icons.SETTINGS,
]


def build_navigation(
    is_mobile: bool,
    labels: Sequence[str],
    on_change: Callable[[ft.ControlEvent], Awaitable[None]],
    selected_index: int = 0,
) -> ft.Control:
    """Build the main navigation control.

    Args:
        is_mobile: when True, build a bottom :class:`ft.NavigationBar`;
            otherwise a side :class:`ft.NavigationRail`.
        labels: localized labels for the 4 destinations
            (search, chat, index, settings).
        on_change: async handler receiving the change event; the selected
            index is available as ``e.control.selected_index``.
        selected_index: initially selected destination.

    Returns:
        The configured navigation control.
    """
    if is_mobile:
        return ft.NavigationBar(
            destinations=[
                ft.NavigationBarDestination(icon=icon, label=label)
                for icon, label in zip(_ICONS, labels)
            ],
            selected_index=selected_index,
            on_change=on_change,
        )
    return ft.NavigationRail(
        destinations=[
            ft.NavigationRailDestination(icon=icon, label=ft.Text(label))
            for icon, label in zip(_ICONS, labels)
        ],
        selected_index=selected_index,
        on_change=on_change,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=92,
        group_alignment=-0.9,
    )
