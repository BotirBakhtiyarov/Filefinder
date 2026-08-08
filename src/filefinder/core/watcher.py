"""Filesystem watcher based on watchdog (desktop only).

Watches monitored directories and reports debounced change events to a
callback of the form ``on_change(kind, path)`` where ``kind`` is
``"upsert"`` (created/modified/moved) or ``"delete"``. The heavy watchdog
dependency is imported lazily so this module imports cleanly without it
and :func:`is_available` reports False (also on Android).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from filefinder.core import constants
from filefinder.core.config import is_android

logger = logging.getLogger(__name__)

ChangeCallback = Callable[[str, str], None]


def _load_watchdog() -> tuple | None:
    """Import watchdog lazily; return (Observer, handler base) or None."""
    try:
        from watchdog.events import FileSystemEventHandler  # type: ignore
        from watchdog.observers import Observer  # type: ignore

        return Observer, FileSystemEventHandler
    except Exception as exc:  # noqa: BLE001 - optional dependency
        logger.debug("watchdog not available: %s", exc)
        return None


def is_available() -> bool:
    """Return True if filesystem watching is supported on this platform."""
    if is_android():
        return False
    return _load_watchdog() is not None


class FileWatcher:
    """Debounced recursive watcher over a set of directories.

    Events per path are debounced by :data:`constants.DEBOUNCE_SECONDS`;
    only the last event within the window is dispatched. The callback is
    invoked from timer threads, so it must be thread-safe.
    """

    def __init__(self) -> None:
        self._observer: Any = None
        self._handler: object = None
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self._running = False

    # --------------------------------------------------------------
    # Debounce
    # --------------------------------------------------------------

    def _schedule(self, kind: str, path: str, callback: ChangeCallback) -> None:
        """(Re)schedule a debounced dispatch for ``path``."""
        with self._lock:
            if not self._running:
                return
            existing = self._timers.pop(path, None)
            if existing is not None:
                existing.cancel()

            def _fire() -> None:
                with self._lock:
                    self._timers.pop(path, None)
                try:
                    callback(kind, path)
                except Exception as exc:  # noqa: BLE001 - never kill watcher
                    logger.error("on_change callback failed for %s: %s", path, exc)

            timer = threading.Timer(constants.DEBOUNCE_SECONDS, _fire)
            timer.daemon = True
            self._timers[path] = timer
            timer.start()

    # --------------------------------------------------------------
    # Event handling
    # --------------------------------------------------------------

    def _should_handle(self, path: str) -> bool:
        """Filter events: supported, non-hidden files only."""
        name = Path(path).name
        if name.startswith("."):
            return False
        return Path(path).suffix.lower() in constants.SUPPORTED_EXTENSIONS

    def _make_handler(self, base: type, callback: ChangeCallback) -> object:
        """Build a watchdog event handler bound to ``callback``."""
        watcher = self

        class _Handler(base):  # type: ignore[misc, valid-type]
            def on_created(self, event):
                if not event.is_directory and watcher._should_handle(event.src_path):
                    watcher._schedule("upsert", event.src_path, callback)

            def on_modified(self, event):
                if not event.is_directory and watcher._should_handle(event.src_path):
                    watcher._schedule("upsert", event.src_path, callback)

            def on_deleted(self, event):
                if not event.is_directory and watcher._should_handle(event.src_path):
                    watcher._schedule("delete", event.src_path, callback)

            def on_moved(self, event):
                if event.is_directory:
                    return
                if watcher._should_handle(event.src_path):
                    watcher._schedule("delete", event.src_path, callback)
                if watcher._should_handle(event.dest_path):
                    watcher._schedule("upsert", event.dest_path, callback)

        return _Handler()

    # --------------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------------

    def start(self, dirs: list[str], on_change_cb: ChangeCallback) -> bool:
        """Start watching the given directories.

        Args:
            dirs: directories to watch recursively.
            on_change_cb: ``(kind, path)`` callback, kind ``"upsert"`` or
                ``"delete"``.

        Returns:
            True if the watcher started, False if watchdog is unavailable
            or watching is unsupported on this platform.
        """
        loaded = _load_watchdog()
        if is_android() or loaded is None:
            logger.warning("FileWatcher unavailable on this platform")
            return False
        observer_cls, handler_base = loaded

        self.stop()
        with self._lock:
            self._running = True
        observer = observer_cls()
        handler = self._make_handler(handler_base, on_change_cb)
        watched = 0
        for directory in dirs:
            if not Path(directory).is_dir():
                logger.warning("Watcher skipping missing directory: %s", directory)
                continue
            observer.schedule(handler, directory, recursive=True)
            watched += 1
        if watched == 0:
            with self._lock:
                self._running = False
            logger.warning("FileWatcher: no valid directories to watch")
            return False
        observer.daemon = True
        observer.start()
        self._observer = observer
        logger.info("FileWatcher started on %d directories", watched)
        return True

    def stop(self) -> None:
        """Stop watching and cancel all pending debounce timers."""
        with self._lock:
            self._running = False
            timers = list(self._timers.values())
            self._timers.clear()
        for timer in timers:
            timer.cancel()
        observer = self._observer
        self._observer = None
        if observer is not None:
            try:
                observer.stop()
                observer.join(timeout=5)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to stop observer cleanly: %s", exc)
        logger.debug("FileWatcher stopped")

    @property
    def is_running(self) -> bool:
        """Return True while the watcher is active."""
        return self._running


# ----------------------------------------------------------------------
# Module-level singleton API (spec section 4.10)
# ----------------------------------------------------------------------

_watcher: FileWatcher | None = None
_watcher_lock = threading.Lock()


def start(dirs: list[str], on_change_cb: ChangeCallback) -> bool:
    """Start the shared :class:`FileWatcher` on ``dirs`` (spec 4.10).

    Args:
        dirs: directories to watch recursively.
        on_change_cb: ``(kind, path)`` callback, kind ``"upsert"`` or
            ``"delete"``.

    Returns:
        True if the watcher started, False otherwise.
    """
    global _watcher
    from filefinder.core import license as license_mod  # lazy

    # Free tier may only watch a single directory (None = unlimited).
    max_dirs = license_mod.get_limit("max_watch_dirs")
    if max_dirs is not None and len(dirs) > max_dirs:
        logger.info(
            "Free-tier watch limit: watching %d of %d directories",
            max_dirs, len(dirs),
        )
        dirs = list(dirs)[:max_dirs]
    with _watcher_lock:
        if _watcher is None:
            _watcher = FileWatcher()
        return _watcher.start(dirs, on_change_cb)


def stop() -> None:
    """Stop the shared watcher, if running (spec 4.10)."""
    global _watcher
    with _watcher_lock:
        if _watcher is not None:
            _watcher.stop()
            _watcher = None
