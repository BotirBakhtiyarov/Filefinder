"""Reusable Flet UI components for FileFinder v3."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ChatBubble",
    "FileCard",
    "IndexProgress",
    "SearchBar",
    "build_navigation",
]


def __getattr__(name: str) -> Any:
    if name == "FileCard":
        from filefinder.ui.components.file_card import FileCard

        return FileCard
    if name == "ChatBubble":
        from filefinder.ui.components.chat_bubble import ChatBubble

        return ChatBubble
    if name == "SearchBar":
        from filefinder.ui.components.search_bar import SearchBar

        return SearchBar
    if name == "IndexProgress":
        from filefinder.ui.components.index_progress import IndexProgress

        return IndexProgress
    if name == "build_navigation":
        from filefinder.ui.components.sidebar import build_navigation

        return build_navigation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
