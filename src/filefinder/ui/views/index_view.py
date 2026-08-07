"""Index view: monitored folders, stats, indexing with progress and cancel."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import flet as ft

from filefinder.core.config import get_config, save_config, t
from filefinder.ui import is_pro
from filefinder.ui.components.index_progress import IndexProgress

if TYPE_CHECKING:  # pragma: no cover
    from filefinder.ui.app import FileFinderApp

logger = logging.getLogger(__name__)


def _stat(stats: dict, *names: str, default: Any = 0) -> Any:
    """Return the first present key among ``names`` from a stats dict."""
    for name in names:
        if name in stats and stats[name] is not None:
            return stats[name]
    return default


def _format_last_indexed(value: Any) -> str:
    """Format a last-indexed timestamp (epoch seconds or string)."""
    if value in (None, "", 0):
        return t("index.last_indexed_never")
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M")
        except (OverflowError, OSError, ValueError):
            return str(value)
    return str(value)


def _watch_limit() -> int:
    """Return the monitored-folder limit for the free tier (default 1)."""
    try:
        from filefinder.core import license as license_core

        limit = license_core.get_limit("max_watch_dirs")
        if isinstance(limit, int) and limit > 0:
            return limit
    except Exception as exc:  # noqa: BLE001
        logger.debug("watch limit unavailable: %s", exc)
    return 1


class IndexView(ft.Column):
    """Index management view.

    Args:
        app: owning :class:`FileFinderApp` (used for snackbars and dialogs).
    """

    def __init__(self, app: "FileFinderApp") -> None:
        self._app = app
        self._cancel_event = threading.Event()
        self._indexing = False

        self.dirs_list = ft.Column(spacing=4, scroll=ft.ScrollMode.AUTO, height=180)
        self.add_folder_button = ft.ElevatedButton(
            content=ft.Row(
                [ft.Icon(ft.Icons.CREATE_NEW_FOLDER, size=16), ft.Text(t("index.add_folder"))],
                spacing=6,
                tight=True,
            ),
            on_click=self._on_add_folder,
        )
        self._picker = ft.FilePicker()

        self.stats_total = ft.Text("", size=13)
        self.stats_db_size = ft.Text("", size=13)
        self.stats_last = ft.Text("", size=13)

        self.start_button = ft.ElevatedButton(
            content=ft.Row(
                [ft.Icon(ft.Icons.PLAY_ARROW, size=16), ft.Text(t("index.start"))],
                spacing=6,
                tight=True,
            ),
            on_click=self._on_start_indexing,
        )
        self.reindex_button = ft.OutlinedButton(
            content=ft.Row(
                [ft.Icon(ft.Icons.REFRESH, size=16), ft.Text(t("index.reindex"))],
                spacing=6,
                tight=True,
            ),
            on_click=self._on_reindex_all,
        )
        self.clear_button = ft.TextButton(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.DELETE_FOREVER, size=16, color=ft.Colors.ERROR),
                    ft.Text(t("index.clear"), color=ft.Colors.ERROR),
                ],
                spacing=6,
                tight=True,
            ),
            on_click=self._on_clear_index,
        )
        self.progress = IndexProgress(
            on_cancel=self._on_cancel,
            cancel_label=t("index.cancel"),
            status_template=t("index.status", current="{current}", total="{total}", filename="{filename}"),
        )

        super().__init__(
            controls=[
                ft.Text(t("index.folders"), weight=ft.FontWeight.BOLD, size=15),
                self.dirs_list,
                self.add_folder_button,
                ft.Divider(height=1),
                ft.Row(
                    [self.stats_total, self.stats_db_size, self.stats_last],
                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                ),
                ft.Divider(height=1),
                ft.Row(
                    [self.start_button, self.reindex_button, self.clear_button],
                    spacing=10,
                    wrap=True,
                ),
                self.progress,
            ],
            expand=True,
            spacing=10,
            scroll=ft.ScrollMode.AUTO,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def did_mount(self) -> None:
        """Attach the FilePicker service and load current state."""
        services = getattr(self.page, "services", None)
        if services is not None:
            if self._picker not in services:
                services.append(self._picker)
        elif self._picker not in self.page.overlay:
            self.page.overlay.append(self._picker)
        self.page.update()
        self.page.run_task(self._refresh)

    # ------------------------------------------------------------------
    # State refresh
    # ------------------------------------------------------------------

    async def _refresh(self) -> None:
        """Reload the monitored dir list and stats row."""
        cfg = get_config()
        self.dirs_list.controls = [
            ft.Row(
                [
                    ft.Icon(ft.Icons.FOLDER, size=18),
                    ft.Text(path, expand=True, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                    ft.IconButton(
                        icon=ft.Icons.DELETE_OUTLINE,
                        icon_size=18,
                        tooltip=t("index.remove_folder"),
                        data=path,
                        on_click=self._on_remove_dir,
                    ),
                ],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
            for path in cfg.monitored_dirs
        ]
        if not cfg.monitored_dirs:
            self.dirs_list.controls.append(
                ft.Text(t("index.no_dirs"), color=ft.Colors.OUTLINE, size=13)
            )

        try:
            from filefinder.core.db import get_db

            stats = await asyncio.to_thread(get_db().get_stats)
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_stats failed: %s", exc)
            stats = {}
        total = _stat(stats, "total_files", "files", "file_count")
        db_size = _stat(stats, "db_size_mb", default=None)
        if db_size is None:
            size_bytes = _stat(stats, "db_size_bytes", "db_size", default=0)
            try:
                db_size = round(float(size_bytes) / (1024 * 1024), 2)
            except (TypeError, ValueError):
                db_size = 0
        last = _format_last_indexed(_stat(stats, "last_indexed", default=None))

        self.stats_total.value = t("index.stats_total", count=total)
        self.stats_db_size.value = t("index.stats_db_size", size=db_size)
        self.stats_last.value = t("index.stats_last_indexed", time=last)
        self.update()

    # ------------------------------------------------------------------
    # Folder management
    # ------------------------------------------------------------------

    async def _on_add_folder(self, e: ft.ControlEvent) -> None:
        """Pick a directory (flet >=0.24 service API returns the path)."""
        try:
            path = await self._picker.get_directory_path(
                dialog_title=t("index.pick_folder")
            )
        except Exception as exc:  # noqa: BLE001
            self._app.show_snackbar(t("error.generic", error=exc), error=True)
            return
        if not path:
            return
        cfg = get_config()
        if path not in cfg.monitored_dirs:
            # Free tier: a single monitored folder; the 2nd one is Pro.
            if not is_pro() and len(cfg.monitored_dirs) >= _watch_limit():
                if not self._app.require_pro("watch"):
                    return
            cfg.monitored_dirs.append(path)
            try:
                save_config(cfg)
            except Exception as exc:  # noqa: BLE001
                self._app.show_snackbar(t("error.generic", error=exc), error=True)
                return
        await self._refresh()

    async def _on_remove_dir(self, e: ft.ControlEvent) -> None:
        path = str(e.control.data)
        cfg = get_config()
        if path in cfg.monitored_dirs:
            cfg.monitored_dirs.remove(path)
            try:
                save_config(cfg)
            except Exception as exc:  # noqa: BLE001
                self._app.show_snackbar(t("error.generic", error=exc), error=True)
                return
        await self._refresh()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def _on_start_indexing(self, e: ft.ControlEvent) -> None:
        if self._indexing:
            return
        cfg = get_config()
        if not cfg.monitored_dirs:
            self._app.show_snackbar(t("index.no_dirs"), error=True)
            return
        await self._run_indexing(reindex=False)

    async def _on_reindex_all(self, e: ft.ControlEvent) -> None:
        if self._indexing:
            return
        await self._run_indexing(reindex=True)

    async def _run_indexing(self, reindex: bool) -> None:
        """Run (re)indexing in a worker thread with progress + cancel."""
        from filefinder.core import indexer

        self._indexing = True
        self._cancel_event.clear()
        self._set_buttons_disabled(True)
        self.progress.show()

        def progress_cb(current: int, total: int, filename: str) -> None:
            self.page.run_task(self._on_progress, current, total, filename)

        try:
            if reindex:
                stats = await asyncio.to_thread(indexer.reindex_all, progress_cb)
            else:
                stats = await asyncio.to_thread(
                    indexer.index_directories,
                    list(get_config().monitored_dirs),
                    progress_cb,
                    self._cancel_event,
                )
            if self._cancel_event.is_set():
                self._app.show_snackbar(t("index.cancelled"))
            else:
                failed = _stat(stats or {}, "failed", "errors", default=0)
                if failed:
                    self._app.show_snackbar(
                        t("index.done_with_errors", failed=failed), error=True
                    )
                else:
                    self._app.show_snackbar(t("index.done"))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Indexing failed")
            self._app.show_snackbar(t("index.failed", error=exc), error=True)
        finally:
            self._indexing = False
            self._set_buttons_disabled(False)
            self.progress.hide()

        await self._refresh()
        try:
            self._app.refresh_tier_ui()
        except Exception as exc:  # noqa: BLE001
            logger.debug("refresh_tier_ui failed: %s", exc)

    async def _on_progress(self, current: int, total: int, filename: str) -> None:
        """Progress callback marshalled back onto the UI task."""
        self.progress.set_progress(current, total, filename)

    async def _on_cancel(self, e: ft.ControlEvent) -> None:
        self._cancel_event.set()
        self.progress.set_status(t("index.cancelling"))

    def _set_buttons_disabled(self, disabled: bool) -> None:
        self.start_button.disabled = disabled
        self.reindex_button.disabled = disabled
        self.clear_button.disabled = disabled
        self.add_folder_button.disabled = disabled
        self.update()

    # ------------------------------------------------------------------
    # Clear index
    # ------------------------------------------------------------------

    async def _on_clear_index(self, e: ft.ControlEvent) -> None:
        page = self.page

        async def on_confirm(ev: ft.ControlEvent) -> None:
            page.pop_dialog()
            try:
                from filefinder.core.db import get_db

                await asyncio.to_thread(get_db().clear_index)
                self._app.show_snackbar(t("index.cleared"))
            except Exception as exc:  # noqa: BLE001
                self._app.show_snackbar(t("error.generic", error=exc), error=True)
            await self._refresh()

        async def on_cancel(ev: ft.ControlEvent) -> None:
            page.pop_dialog()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(t("index.clear_confirm_title")),
            content=ft.Text(t("index.confirm_clear")),
            actions=[
                ft.TextButton(t("common.cancel"), on_click=on_cancel),
                ft.ElevatedButton(
                    t("common.confirm"),
                    color=ft.Colors.ERROR,
                    on_click=on_confirm,
                ),
            ],
        )
        page.show_dialog(dialog)
