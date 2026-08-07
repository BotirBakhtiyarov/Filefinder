"""Flet UI layer for FileFinder v3.

Exposes :class:`FileFinderApp` (lazily, so importing this package does not
require ``flet`` to be installed) plus small shared helpers used across the
UI modules.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = ["FileFinderApp", "is_pro", "kv_get", "kv_set", "maybe_await"]


def is_pro() -> bool:
    """Return True when a Pro license is active.

    Fail-open: when the license module (or the check itself) is
    unavailable, the app behaves as before the freemium feature existed
    (everything unlocked).

    Returns:
        True for pro users or when the license state cannot be determined.
    """
    try:
        from filefinder.core import license as license_core

        return bool(license_core.is_pro())
    except Exception as exc:  # noqa: BLE001
        logger.debug("license check unavailable: %s", exc)
        return True


def kv_get(db: Any, key: str, default: Optional[str] = None) -> Optional[str]:
    """Read a value from the database ``kv`` table.

    The exact accessor name on the ``Database`` facade is not fixed by the
    design spec, so a few conventional names are tried.

    Args:
        db: the ``Database`` facade instance.
        key: kv key to look up.
        default: value returned when the key (or accessor) is missing.

    Returns:
        The stored string value, or ``default``.
    """
    for name in ("kv_get", "get_kv", "get_setting", "get_meta"):
        fn = getattr(db, name, None)
        if callable(fn):
            try:
                value = fn(key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("kv_get(%r) via %s failed: %s", key, name, exc)
                return default
            return default if value is None else value
    logger.debug("No kv accessor found on %r", type(db).__name__)
    return default


def kv_set(db: Any, key: str, value: str) -> None:
    """Write a value to the database ``kv`` table (see :func:`kv_get`)."""
    for name in ("kv_set", "set_kv", "set_setting", "set_meta"):
        fn = getattr(db, name, None)
        if callable(fn):
            try:
                fn(key, value)
            except Exception as exc:  # noqa: BLE001
                logger.warning("kv_set(%r) via %s failed: %s", key, name, exc)
            return
    logger.debug("No kv accessor found on %r", type(db).__name__)


async def maybe_await(value: Any) -> Any:
    """Await ``value`` if it is awaitable, otherwise return it as-is.

    Useful for Flet APIs that changed from sync to async between releases.
    """
    if inspect.isawaitable(value):
        return await value
    return value


def __getattr__(name: str) -> Any:
    if name == "FileFinderApp":
        from filefinder.ui.app import FileFinderApp

        return FileFinderApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
