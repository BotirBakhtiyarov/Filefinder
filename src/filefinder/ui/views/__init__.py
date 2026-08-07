"""App views for FileFinder v3 (search, chat, index, settings)."""

from __future__ import annotations

from typing import Any

__all__ = ["SearchView", "ChatView", "IndexView", "SettingsView"]


def __getattr__(name: str) -> Any:
    if name == "SearchView":
        from filefinder.ui.views.search_view import SearchView

        return SearchView
    if name == "ChatView":
        from filefinder.ui.views.chat_view import ChatView

        return ChatView
    if name == "IndexView":
        from filefinder.ui.views.index_view import IndexView

        return IndexView
    if name == "SettingsView":
        from filefinder.ui.views.settings_view import SettingsView

        return SettingsView
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
