"""FileFinder v3 application shell.

Owns the Flet page: window/theme setup, adaptive navigation (NavigationRail
on desktop widths, NavigationBar on mobile widths), view switching with
keep-alive caching, snackbars, system tray + global hotkey integration and
first-run dialogs (embedding model download, welcome slides).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

import flet as ft

from filefinder.core import config as config_core
from filefinder.core.config import (
    get_config,
    is_android,
    is_windows,
    save_config,
    t,
)
from filefinder.core.constants import APP_ID, APP_NAME
from filefinder.ui import is_pro as ui_is_pro
from filefinder.ui import kv_get, kv_set, maybe_await
from filefinder.ui.components.sidebar import build_navigation

logger = logging.getLogger(__name__)

_DESKTOP_MIN_WIDTH = 700
_WINDOW_WIDTH = 1100
_WINDOW_HEIGHT = 720
_VIEW_COUNT = 4  # search, chat, index, settings
_WELCOME_SLIDES = 5


def _icon_path(name: str = "app_icon.png") -> Optional[str]:
    """Return the absolute path of an app icon, or None when missing."""
    try:
        from filefinder.core.config import get_assets_dir

        candidate = get_assets_dir() / "icons" / name
        return str(candidate) if candidate.is_file() else None
    except Exception:  # noqa: BLE001
        return None


class FileFinderApp:
    """Root application controller for the Flet UI."""

    def __init__(self) -> None:
        self.page: Optional[ft.Page] = None
        self._views: dict[int, ft.Control] = {}
        self._current_index = 0
        self._is_mobile = False
        self._content: Optional[ft.Container] = None
        self._nav: Optional[ft.Control] = None
        self._tray: object = None
        self._hotkey: object = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def init(self, page: ft.Page) -> None:
        """Initialize the application on the given Flet page.

        Args:
            page: the Flet page provided by ``ft.app``.
        """
        self.page = page
        cfg = get_config()

        page.title = APP_NAME
        page.window.width = _WINDOW_WIDTH
        page.window.height = _WINDOW_HEIGHT
        # Window icon: must be an .ico on Windows; resolved from assets_dir.
        if _icon_path("app_icon.ico"):
            page.window.icon = "icons/app_icon.ico"
        page.theme_mode = (
            ft.ThemeMode.DARK if cfg.theme != "light" else ft.ThemeMode.LIGHT
        )
        page.padding = 0

        self._is_mobile = (page.window.width or _WINDOW_WIDTH) < _DESKTOP_MIN_WIDTH
        page.on_resize = self._on_resize

        page.add(self._build_layout())
        page.update()

        self._start_tray()
        self._start_hotkey()
        self.refresh_tier_ui()
        await self._first_run()

        # Automatic background indexing of the monitored folders (seeded
        # with the user's standard folders on first run) + live watcher.
        self.page.run_task(self._auto_index)

    # ------------------------------------------------------------------
    # Automatic indexing
    # ------------------------------------------------------------------

    async def _auto_index(self) -> None:
        """Index monitored folders in the background on every startup.

        Incremental: unchanged files are skipped cheaply (size+mtime), so
        repeated launches are fast. Starts the filesystem watcher afterwards
        so new/changed/deleted files stay in sync automatically.
        """
        cfg = get_config()
        if not cfg.monitored_dirs:
            self._start_auto_watcher()
            return
        try:
            from filefinder.core import indexer

            self._migrate_image_index(indexer)
            self._migrate_fallback_vectors(indexer)
            stats = await asyncio.to_thread(
                indexer.index_directories, list(cfg.monitored_dirs)
            )
            stats = stats or {}
            indexed = int(stats.get("indexed", 0) or 0)
            failed = int(stats.get("failed", 0) or 0)
            if indexed:
                self.show_snackbar(t("index.auto_done", count=indexed))
            if failed:
                logger.info("Auto-index: %d file(s) failed", failed)
            logger.info("Auto-index stats: %s", stats)
            self.refresh_tier_ui()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auto-indexing failed: %s", exc)
        self._start_auto_watcher()

    def _migrate_image_index(self, indexer: object) -> None:
        """One-time reindex of images so they gain CLIP content vectors.

        Older versions indexed images by filename only, and the size+mtime
        fast-path would keep skipping them. Their rows are removed once;
        the subsequent auto-index then re-adds them as "new" files with
        CLIP embeddings. Safe to interrupt: removed rows are re-scanned as
        new files on the next startup.
        """
        try:
            from filefinder.core.constants import IMAGE_EXTENSIONS
            from filefinder.core.db import get_db
            from filefinder.core.image_embeddings import get_image_embedder

            db = get_db()
            if kv_get(db, "clip_reindex_v1") == "1":
                return
            if not get_image_embedder().is_available():
                return
            rows = db.get_files_by_extensions(sorted(IMAGE_EXTENSIONS))
            for row in rows:
                indexer.remove_file(row["path"])
            if rows:
                logger.info(
                    "CLIP migration: %d image(s) scheduled for reindex", len(rows)
                )
            kv_set(db, "clip_reindex_v1", "1")
        except Exception as exc:  # noqa: BLE001
            logger.warning("CLIP image migration skipped: %s", exc)

    def _migrate_fallback_vectors(self, indexer: object) -> None:
        """One-time purge of semantic vectors built from bare filenames.

        Older versions embedded the filename-fallback text (e.g. "😂",
        "IMG_0001"); such single-token chunks produce degenerate embeddings
        that match almost any query with a high score. The rows are removed
        once and re-indexed by the following auto-index WITHOUT semantic
        vectors (they stay keyword-searchable by filename via FTS).
        """
        try:
            from filefinder.core.db import get_db

            db = get_db()
            if kv_get(db, "no_fallback_vectors_v1") == "1":
                return
            removed = 0
            for row in db.get_all_files():
                filename = str(row.get("filename") or "")
                stem = filename.rsplit(".", 1)[0]
                cleaned = stem.replace("_", " ").replace("-", " ").strip()
                preview = str(row.get("text_preview") or "").strip()
                if cleaned and preview == cleaned:
                    indexer.remove_file(str(row.get("path") or ""))
                    removed += 1
            if removed:
                logger.info(
                    "Fallback-vector migration: %d file(s) scheduled for reindex",
                    removed,
                )
            kv_set(db, "no_fallback_vectors_v1", "1")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fallback-vector migration skipped: %s", exc)

    def _start_auto_watcher(self) -> None:
        """Watch monitored folders and keep the index in sync automatically."""
        try:
            from filefinder.core import indexer, watcher

            if not watcher.is_available():
                return
            dirs = list(get_config().monitored_dirs)
            if not dirs:
                return

            def on_change(kind: str, path: str) -> None:
                # Called from watchdog timer threads — index directly there.
                try:
                    if kind == "delete":
                        indexer.remove_file(path)
                    else:
                        indexer.index_file(path)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Watcher update failed for %s: %s", path, exc)

            watcher.start(dirs, on_change)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Auto watcher not started: %s", exc)

    # ------------------------------------------------------------------
    # Layout / navigation
    # ------------------------------------------------------------------

    def _nav_labels(self) -> list[str]:
        return [t("nav.search"), t("nav.chat"), t("nav.index"), t("nav.settings")]

    def _create_view(self, index: int) -> ft.Control:
        """Instantiate the view for a navigation index."""
        if index == 0:
            from filefinder.ui.views.search_view import SearchView

            return SearchView(self)
        if index == 1:
            from filefinder.ui.views.chat_view import ChatView

            return ChatView(self)
        if index == 2:
            from filefinder.ui.views.index_view import IndexView

            return IndexView(self)
        if index == 3:
            from filefinder.ui.views.settings_view import SettingsView

            return SettingsView(self)
        raise ValueError(f"Unknown view index: {index}")

    def _get_view(self, index: int) -> ft.Control:
        """Return the cached (keep-alive) view for ``index``."""
        if index not in self._views:
            self._views[index] = self._create_view(index)
        return self._views[index]

    def _build_layout(self) -> ft.Control:
        """(Re)build the root layout for the current form factor."""
        assert self.page is not None
        self._nav = build_navigation(
            self._is_mobile,
            self._nav_labels(),
            self._on_nav_change,
            self._current_index,
        )
        if self._content is None:
            self._content = ft.Container(expand=True, padding=16)
        self._content.content = self._get_view(self._current_index)

        if self._is_mobile:
            return ft.Column(
                [self._content, self._nav],
                expand=True,
                spacing=0,
            )
        return ft.Row(
            [self._nav, ft.VerticalDivider(width=1), self._content],
            expand=True,
            spacing=0,
        )

    def _rebuild_layout(self) -> None:
        """Tear down and rebuild the whole page layout in place."""
        assert self.page is not None
        self.page.controls.clear()
        self._content = None  # a control cannot live in two trees at once
        self.page.add(self._build_layout())
        self.page.update()

    async def _on_nav_change(self, e: ft.ControlEvent) -> None:
        await self.navigate_to(int(e.control.selected_index))

    async def _on_resize(self, e: ft.ControlEvent) -> None:
        width = getattr(e, "width", None) or (
            self.page.window.width if self.page else _WINDOW_WIDTH
        )
        is_mobile = width < _DESKTOP_MIN_WIDTH
        if is_mobile != self._is_mobile:
            self._is_mobile = is_mobile
            self._rebuild_layout()

    async def navigate_to(self, index: int) -> None:
        """Switch the content area to the view at ``index`` (0..3)."""
        assert self.page is not None
        if not 0 <= index < _VIEW_COUNT:
            return
        self._current_index = index
        if self._content is not None:
            self._content.content = self._get_view(index)
        if self._nav is not None:
            self._nav.selected_index = index
        self.page.update()

    # ------------------------------------------------------------------
    # Freemium / licensing
    # ------------------------------------------------------------------

    def is_pro(self) -> bool:
        """Return True when a Pro license is active (fail-open)."""
        return ui_is_pro()

    def require_pro(self, feature: str) -> bool:
        """Gate a Pro feature behind the upgrade dialog.

        Args:
            feature: gated feature key (``files``/``rag``/``ocr``/
                ``watch``/``export``/``updates``), used to highlight the
                matching benefit in the dialog.

        Returns:
            True when the feature may be used (pro tier); False after
            showing the upgrade dialog to a free user.
        """
        if ui_is_pro():
            return True
        if self.page is not None:
            from filefinder.ui.components.upgrade_dialog import show_upgrade_dialog

            self.page.run_task(
                show_upgrade_dialog,
                self.page,
                feature,
                on_activated=self.notify_tier_changed,
            )
        return False

    def notify_tier_changed(self) -> None:
        """Refresh all tier-dependent UI after activation/deactivation.

        Updates the window title and drops the cached chat/settings views
        so they re-check the license state on next render.
        """
        self.refresh_tier_ui()
        self._views.pop(1, None)  # chat view (RAG switch state)
        self._views.pop(3, None)  # settings view (license section)
        if self.page is not None and self._content is not None:
            if self._current_index in (1, 3):
                self._content.content = self._get_view(self._current_index)
            try:
                self.page.update()
            except Exception as exc:  # noqa: BLE001
                logger.debug("notify_tier_changed update failed: %s", exc)

    def refresh_tier_ui(self) -> None:
        """Refresh the window title to reflect the current license tier.

        Pro: ``FileFinder — 💎 Pro``. Free: ``FileFinder — Free (n/limit)``
        using the indexed-file count from the database stats.
        """
        if self.page is None:
            return
        try:
            from filefinder.core import license as license_core
        except Exception as exc:  # noqa: BLE001
            logger.debug("license module unavailable: %s", exc)
            self.page.title = APP_NAME
            return
        try:
            if license_core.is_pro():
                self.page.title = f"{APP_NAME} — 💎 Pro"
            else:
                limit = license_core.get_limit("max_files")
                files = 0
                try:
                    from filefinder.core.db import get_db

                    stats = get_db().get_stats() or {}
                    files = int(stats.get("total_files", 0) or 0)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("could not read db stats: %s", exc)
                limit_text = str(limit) if limit is not None else "∞"
                self.page.title = f"{APP_NAME} — Free ({files}/{limit_text})"
            self.page.update()
        except Exception as exc:  # noqa: BLE001
            logger.debug("refresh_tier_ui failed: %s", exc)

    # ------------------------------------------------------------------
    # Snackbar helper
    # ------------------------------------------------------------------

    def show_snackbar(self, msg: str, error: bool = False) -> None:
        """Show a transient message at the bottom of the page.

        Args:
            msg: the (already localized) message to show.
            error: when True, use the theme error color.
        """
        if self.page is None:
            logger.info("snackbar (no page): %s", msg)
            return
        self.page.show_dialog(
            ft.SnackBar(
                ft.Text(msg),
                bgcolor=ft.Colors.ERROR if error else None,
            )
        )

    # Spec-named alias for show_snackbar.
    def show_message(self, msg: str, error: bool = False) -> None:
        """Alias of :meth:`show_snackbar` (design-spec name)."""
        self.show_snackbar(msg, error=error)

    # ------------------------------------------------------------------
    # Language switching
    # ------------------------------------------------------------------

    async def set_language(self, lang: str) -> None:
        """Switch the UI language, persisting the choice.

        Rebuilds the navigation labels and recreates all views so every
        user-facing string is re-rendered via :func:`t`.

        Args:
            lang: language code (``zh`` / ``uz`` / ``en``).
        """
        cfg = get_config()
        cfg.language = lang
        try:
            save_config(cfg)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save language: %s", exc)
        try:
            config_core.set_language(lang)
        except Exception as exc:  # noqa: BLE001
            logger.warning("config.set_language failed: %s", exc)

        self._views.clear()
        self._rebuild_layout()

    # ------------------------------------------------------------------
    # System tray (desktop only, optional dependency)
    # ------------------------------------------------------------------

    def _start_tray(self) -> None:
        """Start the pystray system-tray icon in a daemon thread."""
        if is_android():
            return
        try:
            import pystray
            from PIL import Image
        except ImportError:
            logger.info("pystray/Pillow not installed; system tray disabled")
            return

        tray_png = _icon_path("tray_icon.png")
        if tray_png:
            image = Image.open(tray_png)
        else:
            image = Image.new("RGB", (64, 64), color=(63, 81, 181))
        menu = pystray.Menu(
            pystray.MenuItem(
                t("tray.show"), lambda *args: self.page.run_task(self._show_window)
            ),
            pystray.MenuItem(
                t("tray.quit"), lambda *args: self.page.run_task(self._quit_app)
            ),
        )
        try:
            self._tray = pystray.Icon(APP_ID, image, APP_NAME, menu)
            threading.Thread(target=self._tray.run, daemon=True).start()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not start system tray: %s", exc)
            self._tray = None
            return

        # Minimize to tray instead of exiting on window close (Windows).
        if is_windows():
            self.page.window.prevent_close = True
            self.page.window.on_event = self._on_window_event

    async def _on_window_event(self, e: ft.ControlEvent) -> None:
        event_type = getattr(e, "type", None) or getattr(e, "data", None)
        if str(event_type) in ("close", "WindowEventType.CLOSE") and self._tray is not None:
            self.page.window.visible = False
            self.page.update()

    async def _show_window(self) -> None:
        """Restore the window from the tray."""
        self.page.window.visible = True
        self.page.update()

    async def _quit_app(self) -> None:
        """Quit the application from the tray menu."""
        if self._tray is not None:
            try:
                self._tray.stop()
            except Exception:  # noqa: BLE001
                pass
            self._tray = None
        self._stop_hotkey()
        self.page.window.prevent_close = False
        window = self.page.window
        destroy = getattr(window, "destroy", None) or getattr(window, "close", None)
        if callable(destroy):
            await maybe_await(destroy())

    # ------------------------------------------------------------------
    # Global hotkey (Alt+Q toggles window visibility; optional dependency)
    # ------------------------------------------------------------------

    def _start_hotkey(self) -> None:
        """Register the Alt+Q global hotkey via pynput, if available."""
        if is_android():
            return
        try:
            from pynput import keyboard
        except ImportError:
            logger.info("pynput not installed; global hotkey disabled")
            return
        try:
            listener = keyboard.GlobalHotKeys(
                {"<alt>+q": lambda: self.page.run_task(self._toggle_visibility)}
            )
            listener.daemon = True
            listener.start()
            self._hotkey = listener
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not register global hotkey: %s", exc)

    def _stop_hotkey(self) -> None:
        if self._hotkey is not None:
            try:
                self._hotkey.stop()
            except Exception:  # noqa: BLE001
                pass
            self._hotkey = None

    async def _toggle_visibility(self) -> None:
        window = self.page.window
        window.visible = not getattr(window, "visible", True)
        self.page.update()

    # ------------------------------------------------------------------
    # First-run experience
    # ------------------------------------------------------------------

    async def _first_run(self) -> None:
        """Show the welcome slides (once) and the model-download prompt."""
        try:
            from filefinder.core.db import get_db

            db = get_db()
            if not kv_get(db, "welcome_seen"):
                await self._show_welcome()
                kv_set(db, "welcome_seen", "1")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Welcome flow skipped: %s", exc)

        try:
            from filefinder.core.embeddings import get_embedder

            embedder = get_embedder()
            if not embedder.is_model_available():
                await self._show_model_dialog(embedder)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Model availability check failed: %s", exc)

        try:
            await self._maybe_install_ocr()
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR install flow skipped: %s", exc)

    async def _maybe_install_ocr(self) -> None:
        """Offer to auto-install OCR components when missing (Windows only).

        Covers Tesseract (image OCR) and poppler (PDF rendering). Skipped
        entirely when OCR is already available, when the user has
        previously declined (kv flag ``ocr_install_declined``), or on
        non-Windows platforms (auto-install is Windows-only).
        """
        page = self.page
        if page is None or not is_windows():
            return
        try:
            from filefinder.core import deps, ocr

            if ocr.is_ocr_available() and deps.get_poppler_bin_dir() is not None:
                return
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR availability check failed: %s", exc)
            return
        try:
            from filefinder.core.db import get_db

            db = get_db()
            if kv_get(db, "ocr_install_declined"):
                return
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR install flag check failed: %s", exc)
            return

        finished = asyncio.Event()

        progress = ft.ProgressBar(visible=False, expand=True)
        status = ft.Text("", size=12, visible=False, color=ft.Colors.OUTLINE)
        install_button = ft.ElevatedButton(
            t("deps.install"), icon=ft.Icons.DOWNLOAD
        )
        skip_button = ft.TextButton(t("deps.skip"))

        async def on_skip(e: ft.ControlEvent) -> None:
            try:
                kv_set(db, "ocr_install_declined", "1")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not persist OCR decline flag: %s", exc)
            page.pop_dialog()
            finished.set()

        async def on_progress(downloaded: int, total: int) -> None:
            progress.value = (downloaded / total) if total > 0 else None
            try:
                dialog.update()
            except Exception:  # noqa: BLE001
                pass

        async def on_status(msg_key: str) -> None:
            status.value = t(msg_key)
            try:
                dialog.update()
            except Exception:  # noqa: BLE001
                pass

        async def on_install(e: ft.ControlEvent) -> None:
            install_button.disabled = True
            skip_button.disabled = True
            progress.visible = True
            progress.value = None
            status.visible = True
            status.value = t("deps.status_installing")
            dialog.update()

            def progress_cb(downloaded: int, total: int) -> None:
                page.run_task(on_progress, downloaded, total)

            def status_cb(msg_key: str) -> None:
                page.run_task(on_status, msg_key)

            try:
                ok = await asyncio.to_thread(
                    deps.ensure_pdf_ocr, progress_cb, status_cb
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("OCR component install failed: %s", exc)
                ok = False
            if ok:
                self.show_snackbar(t("deps.ocr_installed"))
                page.pop_dialog()
                finished.set()
            else:
                status.value = t("deps.ocr_failed")
                install_button.disabled = False
                skip_button.disabled = False
                dialog.update()

        install_button.on_click = on_install
        skip_button.on_click = on_skip

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(t("deps.ocr_title")),
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text(t("deps.ocr_prompt"), size=14),
                        ft.Row([progress]),
                        status,
                    ],
                    spacing=10,
                    tight=True,
                ),
                width=460,
            ),
            actions=[skip_button, install_button],
        )
        page.show_dialog(dialog)
        await finished.wait()

    async def _show_welcome(self) -> None:
        """Show the 5-slide welcome dialog (skippable)."""
        page = self.page
        finished = asyncio.Event()
        state = {"index": 0}

        slide_title = ft.Text(size=18, weight=ft.FontWeight.BOLD)
        slide_body = ft.Text(size=14, selectable=True)
        dots = ft.Text("", size=12, color=ft.Colors.OUTLINE)
        next_button = ft.ElevatedButton(t("welcome.next"))
        skip_button = ft.TextButton(t("welcome.skip"))

        def render() -> None:
            i = state["index"]
            slide_title.value = t(f"welcome.slide{i + 1}_title")
            slide_body.value = t(f"welcome.slide{i + 1}_body")
            dots.value = f"{i + 1} / {_WELCOME_SLIDES}"
            next_button.content = (
                t("welcome.done") if i == _WELCOME_SLIDES - 1 else t("welcome.next")
            )

        async def on_next(e: ft.ControlEvent) -> None:
            if state["index"] >= _WELCOME_SLIDES - 1:
                page.pop_dialog()
                finished.set()
                return
            state["index"] += 1
            render()
            dialog.update()

        async def on_skip(e: ft.ControlEvent) -> None:
            page.pop_dialog()
            finished.set()

        next_button.on_click = on_next
        skip_button.on_click = on_skip
        render()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(t("firstrun.welcome_title")),
            content=ft.Container(
                content=ft.Column(
                    [slide_title, slide_body, dots],
                    spacing=10,
                    tight=True,
                ),
                width=440,
            ),
            actions=[skip_button, next_button],
        )
        page.show_dialog(dialog)
        await finished.wait()

    async def _show_model_dialog(self, embedder: object) -> None:
        """Offer to download the embedding model or continue keyword-only."""
        page = self.page
        finished = asyncio.Event()

        progress = ft.ProgressBar(visible=False, expand=True)
        status = ft.Text("", size=12, visible=False, color=ft.Colors.OUTLINE)
        download_button = ft.ElevatedButton(
            t("firstrun.download"), icon=ft.Icons.DOWNLOAD
        )
        skip_button = ft.TextButton(t("firstrun.keyword_only"))

        async def on_skip(e: ft.ControlEvent) -> None:
            page.pop_dialog()
            finished.set()

        async def on_progress(downloaded: int, total: int) -> None:
            progress.value = (downloaded / total) if total > 0 else None
            status.value = t("firstrun.downloading")
            try:
                dialog.update()
            except Exception:  # noqa: BLE001
                pass

        async def on_download(e: ft.ControlEvent) -> None:
            download_button.disabled = True
            skip_button.disabled = True
            progress.visible = True
            progress.value = None
            status.visible = True
            status.value = t("firstrun.downloading")
            dialog.update()

            def progress_cb(downloaded: int, total: int) -> None:
                page.run_task(on_progress, downloaded, total)

            try:
                await asyncio.to_thread(embedder.download_model, progress_cb)
            except Exception as exc:  # noqa: BLE001
                logger.error("Model download failed: %s", exc)
                status.value = t("firstrun.download_failed")
                download_button.disabled = False
                skip_button.disabled = False
                dialog.update()
                return
            self.show_snackbar(t("firstrun.download_done"))
            page.pop_dialog()
            finished.set()

        download_button.on_click = on_download
        skip_button.on_click = on_skip

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(t("firstrun.model_title")),
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text(t("welcome.model_download_prompt"), size=14),
                        ft.Row([progress]),
                        status,
                    ],
                    spacing=10,
                    tight=True,
                ),
                width=460,
            ),
            actions=[skip_button, download_button],
        )
        page.show_dialog(dialog)
        await finished.wait()
