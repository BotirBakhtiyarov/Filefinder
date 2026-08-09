"""Settings view: language, theme, LLM, model management, updates, about."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import flet as ft

from filefinder.core.config import get_config, get_model_dir, is_android, save_config, t
from filefinder.ui import is_pro

if TYPE_CHECKING:  # pragma: no cover
    from filefinder.ui.app import FileFinderApp

logger = logging.getLogger(__name__)

_LANGUAGES = {"zh": "简体中文", "uz": "Oʻzbek", "en": "English"}

from filefinder.core import chat as chat_core

_PROVIDERS = list(chat_core.PROVIDER_CHOICES)
_PROVIDER_LABELS = chat_core.PROVIDER_LABELS


def _app_version() -> str:
    """Return the installed package version (fallback when unavailable)."""
    try:
        from filefinder import __version__

        return __version__
    except Exception:  # noqa: BLE001
        return "0.0.0"


def _model_display_name() -> str:
    """Return the configured/platform-default embedding model name."""
    cfg = get_config()
    if getattr(cfg, "embedding_model", None):
        return str(cfg.embedding_model)
    return "bge-small-zh-v1.5-onnx" if is_android() else "bge-m3-onnx"


def _dir_size_mb(path: Path) -> float:
    """Return the total size of ``path`` in MB (0 when missing)."""
    try:
        total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except OSError:
        return 0.0
    return round(total / (1024 * 1024), 1)


def _section(title: str, controls: list[ft.Control]) -> ft.Card:
    """Build a titled settings section card."""
    return ft.Card(
        content=ft.Container(
            content=ft.Column(
                [ft.Text(title, weight=ft.FontWeight.BOLD, size=15), *controls],
                spacing=10,
            ),
            padding=14,
        ),
        elevation=1,
    )


class SettingsView(ft.Column):
    """Application settings view.

    Args:
        app: owning :class:`FileFinderApp` (language switching, snackbars).
    """

    def __init__(self, app: FileFinderApp) -> None:
        self._app = app
        cfg = get_config()

        # --- License / Pro ------------------------------------------
        self._license_section: ft.Card | None = self._build_license_section()

        # --- General -------------------------------------------------
        self.language_dropdown = ft.Dropdown(
            label=t("settings.language"),
            value=cfg.language if cfg.language in _LANGUAGES else "zh",
            options=[ft.dropdown.Option(code, name) for code, name in _LANGUAGES.items()],
            on_select=self._on_language_change,
            width=220,
            dense=True,
        )
        self.theme_switch = ft.Switch(
            label=t("settings.theme_dark"),
            value=cfg.theme == "dark",
            on_change=self._on_theme_change,
        )

        # --- LLM -----------------------------------------------------
        self.provider_dropdown = ft.Dropdown(
            label=t("settings.provider"),
            value=cfg.llm_provider if cfg.llm_provider in _PROVIDERS else "deepseek",
            options=[
                ft.dropdown.Option(p, _PROVIDER_LABELS.get(p, p)) for p in _PROVIDERS
            ],
            on_select=self._on_provider_change,
            width=220,
            dense=True,
        )
        self.api_url_field = ft.TextField(
            label=t("settings.api_url"),
            value=cfg.llm_api_url or "",
            dense=True,
        )
        self.api_key_field = ft.TextField(
            label=t("settings.api_key"),
            value=cfg.llm_api_key or "",
            password=True,
            can_reveal_password=True,
            dense=True,
        )
        self.llm_model_field = ft.TextField(
            label=t("settings.api_model"),
            value=cfg.llm_model or "",
            dense=True,
        )
        save_llm_button = ft.ElevatedButton(
            t("settings.save"), icon=ft.Icons.SAVE, on_click=self._on_save_llm
        )
        self._apply_provider_state(self.provider_dropdown.value)

        # --- Embedding model ------------------------------------------
        self.model_info_text = ft.Text("", size=13)
        self.model_progress = ft.ProgressBar(visible=False, expand=True)
        self.download_model_button = ft.ElevatedButton(
            t("settings.download"), icon=ft.Icons.DOWNLOAD, on_click=self._on_download_model
        )
        self.delete_model_button = ft.OutlinedButton(
            t("settings.model_delete"),
            icon=ft.Icons.DELETE_OUTLINE,
            on_click=self._on_delete_model,
        )

        # --- CLIP image-search model ------------------------------------
        self.clip_info_text = ft.Text("", size=13)
        self.clip_progress = ft.ProgressBar(visible=False, expand=True)
        self.download_clip_button = ft.ElevatedButton(
            t("settings.download"), icon=ft.Icons.DOWNLOAD, on_click=self._on_download_clip
        )
        self.delete_clip_button = ft.OutlinedButton(
            t("settings.model_delete"),
            icon=ft.Icons.DELETE_OUTLINE,
            on_click=self._on_delete_clip,
        )

        # --- Updates ---------------------------------------------------
        self.update_status = ft.Text(
            t("settings.current_version", version=_app_version()), size=13
        )
        self.update_progress = ft.ProgressBar(visible=False, expand=True)
        check_updates_button = ft.OutlinedButton(
            t("settings.check_updates"),
            icon=ft.Icons.SYSTEM_UPDATE_ALT,
            on_click=self._on_check_updates,
        )

        # --- About -----------------------------------------------------
        from filefinder.core.constants import GITHUB_REPO

        about_controls: list[ft.Control] = [
            ft.Text(t("settings.about_version", version=_app_version()), size=13),
            ft.Text(t("settings.about_author", author="Botir Bakhtiyarov"), size=13),
            ft.TextButton(
                content=ft.Row(
                    [ft.Icon(ft.Icons.LINK, size=16), ft.Text(GITHUB_REPO)],
                    spacing=6,
                    tight=True,
                ),
                url=f"https://github.com/{GITHUB_REPO}",
            ),
        ]

        sections: list[ft.Control] = []
        if self._license_section is not None:
            sections.append(self._license_section)
        super().__init__(
            controls=[
                *sections,
                _section(
                    t("settings.section_general"),
                    [self.language_dropdown, self.theme_switch],
                ),
                _section(
                    t("settings.llm_config"),
                    [
                        self.provider_dropdown,
                        self.api_url_field,
                        self.api_key_field,
                        self.llm_model_field,
                        save_llm_button,
                    ],
                ),
                _section(
                    t("settings.model_management"),
                    [
                        self.model_info_text,
                        ft.Row([self.model_progress]),
                        ft.Row(
                            [self.download_model_button, self.delete_model_button],
                            spacing=10,
                        ),
                        ft.Divider(),
                        self.clip_info_text,
                        ft.Row([self.clip_progress]),
                        ft.Row(
                            [self.download_clip_button, self.delete_clip_button],
                            spacing=10,
                        ),
                    ],
                ),
                _section(
                    t("settings.updates"),
                    [
                        self.update_status,
                        ft.Row([self.update_progress]),
                        check_updates_button,
                    ],
                ),
                _section(t("settings.about"), about_controls),
            ],
            expand=True,
            spacing=12,
            scroll=ft.ScrollMode.AUTO,
        )

    def did_mount(self) -> None:
        """Refresh the model info once mounted."""
        self.page.run_task(self._refresh_model_info)
        self.page.run_task(self._refresh_clip_info)

    # ------------------------------------------------------------------
    # License / Pro
    # ------------------------------------------------------------------

    def _build_license_section(self) -> ft.Card | None:
        """Build the license section card, or None when licensing is off.

        Returns None when ``filefinder.core.license`` is unavailable so
        builds without the freemium feature look exactly as before.
        """
        try:
            from filefinder.core import license as license_core  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            logger.debug("license module unavailable: %s", exc)
            return None
        return _section(t("license.upgrade_title"), self._license_controls())

    def _license_controls(self) -> list[ft.Control]:
        """Return the controls for the current license tier."""
        try:
            from filefinder.core import license as license_core

            status = license_core.get_status() or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_status failed: %s", exc)
            status = {}
        if not is_pro():
            upgrade_button = ft.ElevatedButton(
                t("license.upgrade_title"),
                icon=ft.Icons.DIAMOND_OUTLINED,
                on_click=self._on_upgrade,
            )
            controls = [ft.Text(t("license.current_free"), size=13)]
            days_remaining = status.get("days_remaining")
            if days_remaining is not None and days_remaining < 0:
                controls.append(
                    ft.Text(
                        t("license.expired_note"), size=13, color=ft.Colors.ERROR
                    )
                )
            controls.append(upgrade_button)
            return list(controls)
        activated_at = status.get("activated_at") or "-"
        device_id = status.get("device_id") or "-"
        expires_at = status.get("expires_at") or "-"
        days = status.get("days_remaining")
        deactivate_button = ft.OutlinedButton(
            t("license.deactivate"),
            icon=ft.Icons.LOGOUT,
            on_click=self._on_deactivate,
        )
        return [
            ft.Text(t("license.current_pro"), size=13),
            ft.Text(str(status.get("key_masked") or "-"), size=13, selectable=True),
            ft.Text(t("license.activated_at", date=activated_at), size=13),
            ft.Text(t("license.expires_at", date=expires_at, days=days), size=13),
            ft.Text(t("license.device_id", id=device_id), size=13, selectable=True),
            deactivate_button,
        ]

    async def _on_upgrade(self, e: ft.ControlEvent) -> None:
        from filefinder.ui.components.upgrade_dialog import show_upgrade_dialog

        await show_upgrade_dialog(
            self.page, on_activated=self._app.notify_tier_changed
        )

    async def _on_deactivate(self, e: ft.ControlEvent) -> None:
        page = self.page

        async def on_confirm(ev: ft.ControlEvent) -> None:
            page.pop_dialog()
            try:
                from filefinder.core import license as license_core

                await asyncio.to_thread(license_core.deactivate)
                self._app.show_snackbar(t("license.deactivated"))
            except Exception as exc:  # noqa: BLE001
                self._app.show_snackbar(t("error.generic", error=exc), error=True)
            self._app.notify_tier_changed()

        async def on_cancel(ev: ft.ControlEvent) -> None:
            page.pop_dialog()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(t("license.deactivate")),
            content=ft.Text(t("license.deactivate_confirm")),
            actions=[
                ft.TextButton(t("common.cancel"), on_click=on_cancel),
                ft.ElevatedButton(
                    t("common.confirm"), color=ft.Colors.ERROR, on_click=on_confirm
                ),
            ],
        )
        page.show_dialog(dialog)

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------

    async def _on_language_change(self, e: ft.ControlEvent) -> None:
        await self._app.set_language(e.control.value)

    async def _on_theme_change(self, e: ft.ControlEvent) -> None:
        cfg = get_config()
        cfg.theme = "dark" if e.control.value else "light"
        try:
            save_config(cfg)
        except Exception as exc:  # noqa: BLE001
            self._app.show_snackbar(t("error.generic", error=exc), error=True)
            return
        self.page.theme_mode = (
            ft.ThemeMode.DARK if cfg.theme == "dark" else ft.ThemeMode.LIGHT
        )
        self.page.update()

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    def _apply_provider_state(self, provider: str) -> None:
        """Adjust LLM fields for the selected provider.

        ``deepseek`` (built-in key): URL and key fields are hidden, the
        model field shows the default and stays editable. ``ollama``:
        URL shown disabled, no key. ``custom``: URL + key + model editable.
        ``off``: everything hidden.
        """
        preset = chat_core.LLM_PROVIDERS.get(provider)
        is_custom = provider == "custom"
        if preset is not None and not is_custom:
            self.api_url_field.value = preset["url"]
        self.api_url_field.visible = is_custom or provider == "ollama"
        self.api_url_field.disabled = not is_custom
        self.api_key_field.visible = is_custom
        known_models = {
            p["model"] for p in chat_core.LLM_PROVIDERS.values() if p["model"]
        }
        current = (self.llm_model_field.value or "").strip()
        if preset is not None and (not current or current in known_models):
            self.llm_model_field.value = preset["model"]
        self.llm_model_field.visible = provider != "off"

    async def _on_provider_change(self, e: ft.ControlEvent) -> None:
        self._apply_provider_state(self.provider_dropdown.value or "off")
        try:
            self.update()
        except Exception:  # noqa: BLE001 - not mounted yet
            pass

    async def _on_save_llm(self, e: ft.ControlEvent) -> None:
        cfg = get_config()
        cfg.llm_provider = self.provider_dropdown.value
        cfg.llm_api_url = self.api_url_field.value or ""
        cfg.llm_api_key = self.api_key_field.value or ""
        cfg.llm_model = self.llm_model_field.value or ""
        try:
            save_config(cfg)
            self._app.show_snackbar(t("settings.saved"))
        except Exception as exc:  # noqa: BLE001
            self._app.show_snackbar(t("error.generic", error=exc), error=True)

    # ------------------------------------------------------------------
    # Embedding model management
    # ------------------------------------------------------------------

    async def _refresh_model_info(self) -> None:
        try:
            from filefinder.core.embeddings import get_embedder

            available = await asyncio.to_thread(get_embedder().is_model_available)
        except Exception:  # noqa: BLE001
            available = False
        size_mb = await asyncio.to_thread(_dir_size_mb, Path(get_model_dir()))
        state_key = "settings.model_available" if available else "settings.model_missing"
        self.model_info_text.value = (
            f"{_model_display_name()} — {t(state_key)} — "
            f"{t('settings.model_size', size=size_mb)}"
        )
        self.update()

    async def _on_download_model(self, e: ft.ControlEvent) -> None:
        from filefinder.core.embeddings import get_embedder

        self.download_model_button.disabled = True
        self.delete_model_button.disabled = True
        self.model_progress.visible = True
        self.model_progress.value = None  # indeterminate while total is unknown
        self.update()

        def progress_cb(downloaded: int, total: int) -> None:
            self.page.run_task(self._on_model_progress, downloaded, total)

        try:
            await asyncio.to_thread(get_embedder().download_model, progress_cb)
            self._app.show_snackbar(t("firstrun.download_done"))
        except Exception as exc:  # noqa: BLE001
            logger.error("Model download failed: %s", exc)
            self._app.show_snackbar(t("firstrun.download_failed"), error=True)
        finally:
            self.download_model_button.disabled = False
            self.delete_model_button.disabled = False
            self.model_progress.visible = False
        await self._refresh_model_info()

    async def _on_model_progress(self, downloaded: int, total: int) -> None:
        if total > 0:
            self.model_progress.value = downloaded / total
            self.model_progress.update()

    async def _on_delete_model(self, e: ft.ControlEvent) -> None:
        page = self.page

        async def on_confirm(ev: ft.ControlEvent) -> None:
            page.pop_dialog()
            try:
                await asyncio.to_thread(
                    shutil.rmtree, str(get_model_dir()), True  # ignore_errors
                )
                self._app.show_snackbar(t("settings.model_deleted"))
            except Exception as exc:  # noqa: BLE001
                self._app.show_snackbar(t("error.generic", error=exc), error=True)
            await self._refresh_model_info()

        async def on_cancel(ev: ft.ControlEvent) -> None:
            page.pop_dialog()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(t("settings.model_delete")),
            content=ft.Text(t("settings.delete_model_confirm")),
            actions=[
                ft.TextButton(t("common.cancel"), on_click=on_cancel),
                ft.ElevatedButton(
                    t("common.confirm"), color=ft.Colors.ERROR, on_click=on_confirm
                ),
            ],
        )
        page.show_dialog(dialog)

    # ------------------------------------------------------------------
    # CLIP image-search model management
    # ------------------------------------------------------------------

    async def _refresh_clip_info(self) -> None:
        try:
            from filefinder.core.image_embeddings import (
                get_clip_model_dir,
                get_image_embedder,
            )

            available = await asyncio.to_thread(get_image_embedder().is_available)
            clip_dir = get_clip_model_dir()
        except Exception:  # noqa: BLE001
            available = False
            clip_dir = Path(get_model_dir()).parent / "clip"
        size_mb = await asyncio.to_thread(_dir_size_mb, Path(clip_dir))
        state_key = "settings.model_available" if available else "settings.model_missing"
        self.clip_info_text.value = (
            f"{t('settings.clip_model')} — {t(state_key)} — "
            f"{t('settings.model_size', size=size_mb)}"
        )
        try:
            self.update()
        except Exception:  # noqa: BLE001 - view may be unmounted
            pass

    async def _on_download_clip(self, e: ft.ControlEvent) -> None:
        from filefinder.core.image_embeddings import get_image_embedder

        self.download_clip_button.disabled = True
        self.delete_clip_button.disabled = True
        self.clip_progress.visible = True
        self.clip_progress.value = None  # indeterminate while total is unknown
        self.update()

        def progress_cb(downloaded: int, total: int) -> None:
            self.page.run_task(self._on_clip_progress, downloaded, total)

        try:
            await asyncio.to_thread(get_image_embedder().download_model, progress_cb)
            self._app.show_snackbar(t("firstrun.download_done"))
        except Exception as exc:  # noqa: BLE001
            logger.error("CLIP model download failed: %s", exc)
            self._app.show_snackbar(t("firstrun.download_failed"), error=True)
        finally:
            self.download_clip_button.disabled = False
            self.delete_clip_button.disabled = False
            self.clip_progress.visible = False
        await self._refresh_clip_info()

    async def _on_clip_progress(self, downloaded: int, total: int) -> None:
        if total > 0:
            self.clip_progress.value = downloaded / total
            self.clip_progress.update()

    async def _on_delete_clip(self, e: ft.ControlEvent) -> None:
        page = self.page

        async def on_confirm(ev: ft.ControlEvent) -> None:
            page.pop_dialog()
            try:
                from filefinder.core.image_embeddings import get_clip_model_dir

                await asyncio.to_thread(
                    shutil.rmtree, str(get_clip_model_dir()), True  # ignore_errors
                )
                self._app.show_snackbar(t("settings.model_deleted"))
            except Exception as exc:  # noqa: BLE001
                self._app.show_snackbar(t("error.generic", error=exc), error=True)
            await self._refresh_clip_info()

        async def on_cancel(ev: ft.ControlEvent) -> None:
            page.pop_dialog()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(t("settings.model_delete")),
            content=ft.Text(t("settings.delete_clip_confirm")),
            actions=[
                ft.TextButton(t("common.cancel"), on_click=on_cancel),
                ft.ElevatedButton(
                    t("common.confirm"), color=ft.Colors.ERROR, on_click=on_confirm
                ),
            ],
        )
        page.show_dialog(dialog)

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    async def _on_check_updates(self, e: ft.ControlEvent) -> None:
        try:
            from filefinder.updater import github_checker
        except Exception as exc:  # noqa: BLE001
            self._app.show_snackbar(t("settings.update_check_failed"), error=True)
            logger.warning("updater unavailable: %s", exc)
            return

        self.update_status.value = t("settings.checking_updates")
        self.update()
        try:
            info = await asyncio.to_thread(github_checker.check_update, _app_version())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Update check failed: %s", exc)
            self.update_status.value = t("settings.update_check_failed")
            self.update()
            return

        if not info:
            self.update_status.value = t("settings.no_update")
            self.update()
            return
        self.update_status.value = t(
            "settings.current_version", version=_app_version()
        )
        self.update()
        self._show_update_dialog(info)

    def _show_update_dialog(self, info: dict) -> None:
        page = self.page
        version = str(info.get("version", ""))
        notes = str(info.get("notes", "") or "")
        progress = ft.ProgressBar(visible=False, expand=True)
        status = ft.Text("", size=12, visible=False)
        content_column = ft.Column(
            [
                ft.Text(notes, selectable=True, size=13),
                ft.Row([progress]),
                status,
            ],
            tight=True,
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
        )
        download_button = ft.ElevatedButton(t("settings.download"))
        close_button = ft.TextButton(t("common.close"))
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(t("settings.update_available", version=version)),
            content=ft.Container(content_column, width=460, height=240),
            actions=[close_button, download_button],
        )

        async def on_close(ev: ft.ControlEvent) -> None:
            page.pop_dialog()

        async def on_download(ev: ft.ControlEvent) -> None:
            download_button.disabled = True
            close_button.disabled = True
            progress.visible = True
            status.visible = True
            status.value = t("settings.downloading_update")
            dialog.update()

            from filefinder.updater import downloader, installer

            dest = Path(tempfile.gettempdir()) / f"filefinder-update-{version}.zip"

            def progress_cb(downloaded: int, total: int) -> None:
                page.run_task(self._on_update_progress, progress, downloaded, total)

            try:
                await asyncio.to_thread(
                    downloader.download, str(info.get("url", "")), str(dest), progress_cb
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Update download failed: %s", exc)
                status.value = t("settings.update_download_failed")
                download_button.disabled = False
                close_button.disabled = False
                dialog.update()
                return

            status.value = t("settings.installing_update")
            dialog.update()
            try:
                await asyncio.to_thread(installer.install, str(dest))
                # install() relaunches the app; if it returns, report success.
                status.value = t("settings.update_installed")
            except Exception as exc:  # noqa: BLE001
                logger.error("Update install failed: %s", exc)
                status.value = t("settings.update_install_failed")
            close_button.disabled = False
            dialog.update()

        close_button.on_click = on_close
        download_button.on_click = on_download
        page.show_dialog(dialog)

    async def _on_update_progress(
        self, bar: ft.ProgressBar, downloaded: int, total: int
    ) -> None:
        bar.value = (downloaded / total) if total > 0 else None
        try:
            bar.update()
        except Exception:  # noqa: BLE001
            pass
