"""Search view: query bar, file-type filter chips and result cards."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import flet as ft

from filefinder.core.config import t
from filefinder.ui import is_pro, kv_get, kv_set
from filefinder.ui.components.file_card import FileCard
from filefinder.ui.components.search_bar import SearchBar

if TYPE_CHECKING:  # pragma: no cover
    from filefinder.ui.app import FileFinderApp

logger = logging.getLogger(__name__)

# Filter chip label -> covered extensions.
_CHIP_EXTENSIONS: dict[str, set[str]] = {
    "docx": {".docx"},
    "pdf": {".pdf"},
    "xlsx": {".xlsx"},
    "pptx": {".pptx"},
    "txt": {".txt", ".md"},
    "image": {".jpg", ".jpeg", ".png", ".bmp"},
}

# ft.FilterChip was added in flet 0.24; fall back to ft.Chip if unavailable.
_Chip = getattr(ft, "FilterChip", None) or ft.Chip


class SearchView(ft.Column):
    """Semantic/keyword search view.

    Args:
        app: owning :class:`FileFinderApp` (used for snackbars and routing
            to the index view from the empty state).
    """

    def __init__(self, app: "FileFinderApp") -> None:
        self._app = app
        self._selected_types: set[str] = set()
        self._searching = False
        self._last_results: list[Any] = []
        self._export_picker = ft.FilePicker()

        self.search_bar = SearchBar(
            on_search=self._run_search,
            hint_text=t("search.placeholder"),
            button_tooltip=t("search.button"),
        )
        self._chips = [
            _Chip(
                label=ft.Text(label),
                selected=False,
                on_select=self._on_chip_toggle,
                data=label,
            )
            for label in _CHIP_EXTENSIONS
        ]
        self._spinner = ft.ProgressRing(visible=False, width=22, height=22)
        self.export_button = ft.IconButton(
            icon=ft.Icons.DOWNLOAD,
            tooltip=t("license.export"),
            on_click=self._on_export,
            visible=False,
        )
        self.results = ft.ListView(expand=True, spacing=8)
        self._hint = ft.Container(
            content=ft.Column(
                [
                    ft.Icon(ft.Icons.MANAGE_SEARCH, size=56, color=ft.Colors.OUTLINE),
                    ft.Text(t("search.hint"), color=ft.Colors.OUTLINE),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=8,
            ),
            alignment=ft.Alignment.CENTER,
            expand=True,
            visible=True,
        )
        self._empty = ft.Container(
            content=ft.Column(
                [
                    ft.Icon(ft.Icons.SEARCH_OFF, size=56, color=ft.Colors.OUTLINE),
                    ft.Text(t("search.empty"), color=ft.Colors.OUTLINE),
                    ft.ElevatedButton(
                        content=ft.Row(
                            [
                                ft.Icon(ft.Icons.FOLDER, size=16),
                                ft.Text(t("search.go_to_index")),
                            ],
                            spacing=6,
                            tight=True,
                        ),
                        on_click=self._on_go_to_index,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=10,
            ),
            alignment=ft.Alignment.CENTER,
            expand=True,
            visible=False,
        )

        super().__init__(
            controls=[
                self.search_bar,
                ft.Row([*self._chips, self._spinner, self.export_button], spacing=8, wrap=True),
                ft.Stack([self.results, self._hint, self._empty], expand=True),
            ],
            expand=True,
            spacing=10,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def did_mount(self) -> None:
        """Attach the export FilePicker service once mounted."""
        services = getattr(self.page, "services", None)
        if services is not None:
            if self._export_picker not in services:
                services.append(self._export_picker)
        elif self._export_picker not in self.page.overlay:
            self.page.overlay.append(self._export_picker)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_chip_toggle(self, e: ft.ControlEvent) -> None:
        label = str(e.control.data)
        if e.control.selected:
            self._selected_types.add(label)
        else:
            self._selected_types.discard(label)

    async def _on_go_to_index(self, e: ft.ControlEvent) -> None:
        await self._app.navigate_to(2)

    async def _run_search(self, query: str) -> None:
        """Execute a search off the UI thread and render the results."""
        query = (query or "").strip()
        if not query:
            self._hint.visible = True
            self._empty.visible = False
            self.export_button.visible = False
            self._last_results = []
            self.results.controls.clear()
            self.update()
            return
        if self._searching:
            return
        self._searching = True
        self._spinner.visible = True
        self._hint.visible = False
        self._empty.visible = False
        self.export_button.visible = False
        self.update()
        try:
            upsell, results = await asyncio.gather(
                asyncio.to_thread(self._maybe_upsell),
                asyncio.to_thread(self._do_search, query),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Search failed")
            self._app.show_snackbar(t("search.error", error=exc), error=True)
            upsell, results = None, []
        finally:
            self._searching = False
            self._spinner.visible = False

        if upsell:
            self._app.show_snackbar(upsell)
        self._last_results = list(results)
        self.results.controls = [
            FileCard(
                r,
                open_label=t("filecard.open"),
                on_error=lambda msg: self._app.show_snackbar(msg, error=True),
            )
            for r in results
        ]
        self._empty.visible = not results
        self.export_button.visible = bool(results)
        self.update()

    # ------------------------------------------------------------------
    # Freemium: soft upsell + export gate
    # ------------------------------------------------------------------

    def _maybe_upsell(self) -> Optional[str]:
        """Bump the free-tier search counter (runs in a worker thread).

        Returns:
            The localized upsell message every 5th search on the free
            tier, otherwise None.
        """
        if is_pro():
            return None
        try:
            from filefinder.core.db import get_db

            db = get_db()
            count = int(kv_get(db, "search_count", "0") or 0) + 1
            kv_set(db, "search_count", str(count))
        except Exception as exc:  # noqa: BLE001
            logger.debug("search counter unavailable: %s", exc)
            return None
        if count % 5 == 0:
            return t("license.upsell_search")
        return None

    async def _on_export(self, e: ft.ControlEvent) -> None:
        """Export the current results to CSV or XLSX (Pro feature)."""
        if not self._last_results:
            return
        if not self._app.require_pro("export"):
            return
        try:
            path = await self._export_picker.save_file(
                dialog_title=t("license.export"),
                file_name="filefinder_results.xlsx",
                allowed_extensions=["xlsx", "csv"],
            )
        except Exception as exc:  # noqa: BLE001
            self._app.show_snackbar(t("error.generic", error=exc), error=True)
            return
        if not path:
            return
        try:
            await asyncio.to_thread(self._write_export, path, self._last_results)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Export failed")
            self._app.show_snackbar(t("error.generic", error=exc), error=True)
            return
        self._app.show_snackbar(t("license.export_done", path=path))

    @staticmethod
    def _export_rows(results: list[Any]) -> list[list[Any]]:
        """Normalize result objects/dicts into export row tuples."""

        def field(r: Any, name: str) -> Any:
            if isinstance(r, dict):
                return r.get(name)
            return getattr(r, name, None)

        rows: list[list[Any]] = []
        for r in results:
            modified = field(r, "modified_time")
            try:
                modified_text = datetime.fromtimestamp(float(modified)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except (TypeError, ValueError, OverflowError, OSError):
                modified_text = "" if modified is None else str(modified)
            rows.append(
                [
                    str(field(r, "path") or ""),
                    str(field(r, "filename") or ""),
                    str(field(r, "extension") or ""),
                    float(field(r, "score") or 0.0),
                    modified_text,
                ]
            )
        return rows

    def _write_export(self, path: str, results: list[Any]) -> None:
        """Write ``results`` to ``path`` as CSV or XLSX (by extension)."""
        header = ["path", "filename", "extension", "score", "modified_time"]
        rows = self._export_rows(results)
        if path.lower().endswith(".csv"):
            import csv

            with open(path, "w", newline="", encoding="utf-8-sig") as fh:
                writer = csv.writer(fh)
                writer.writerow(header)
                writer.writerows(rows)
            return
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = "results"
        ws.append(header)
        for row in rows:
            ws.append(row)
        wb.save(path)

    # ------------------------------------------------------------------
    # Core bridge (runs in a worker thread)
    # ------------------------------------------------------------------

    def _do_search(self, query: str) -> list[Any]:
        """Call the search engine with the currently selected filters."""
        from filefinder.core import search as search_core

        factory = getattr(search_core, "get_search_engine", None)
        engine = factory() if callable(factory) else search_core.SearchEngine()

        filters: Optional[Any] = None
        if self._selected_types:
            extensions = sorted(
                {ext for label in self._selected_types for ext in _CHIP_EXTENSIONS[label]}
            )
            filters_cls = getattr(search_core, "Filters", None)
            if filters_cls is not None:
                filters = filters_cls(file_types=extensions)
        return engine.search(query, filters)
