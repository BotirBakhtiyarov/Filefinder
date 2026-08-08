"""Pro upgrade dialog: feature pitch, payment QR flow and license activation.

The dialog is a small state machine rendered inside a single modal
:class:`flet.AlertDialog`:

- ``main``: feature checklist, price and payment-method buttons.
- ``pay``: QR code for the chosen method + payment steps + "I paid".
- ``contact``: author WeChat ID with a copy button (key delivery channel).
- ``activate``: license-key entry with auto-formatting and inline errors.
- ``success``: activation confirmation; invokes ``on_activated``.

All license operations go through ``filefinder.core.license`` (imported
lazily) and run in worker threads so the UI stays responsive.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from typing import Any

import flet as ft

from filefinder.core.config import t
from filefinder.ui import maybe_await

logger = logging.getLogger(__name__)

_WECHAT_ID = "nick_2620"
_FEATURES = ("files", "rag", "ocr", "watch", "export", "updates")
_KEY_MAX_LEN = 19  # "FF3" + dash-separated 4-char groups
_DIALOG_WIDTH = 440

# license.activate() error code -> i18n key suffix.
_ERROR_CODES = (
    "invalid_format",
    "invalid_checksum",
    "blacklisted",
    "device_mismatch",
    "expired",
)


def format_license_key(raw: str) -> str:
    """Auto-format a license key as ``FF3-XXXX-XXXX-XXXX``.

    Uppercases the input, strips non-alphanumerics, forces the ``FF3``
    prefix, inserts a dash every 4 characters after the prefix and caps
    the result at 19 characters.

    Args:
        raw: the current raw text-field value.

    Returns:
        The formatted key (may be empty or a partial prefix while typing).
    """
    alnum = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if not alnum:
        return ""
    if not alnum.startswith("FF3"):
        if "FF3".startswith(alnum):  # still typing the prefix itself
            return alnum
        alnum = "FF3" + alnum
    body = alnum[3:]
    if not body:
        return "FF3"
    groups = [body[i : i + 4] for i in range(0, len(body), 4)]
    return ("FF3-" + "-".join(groups))[:_KEY_MAX_LEN].rstrip("-")


async def show_upgrade_dialog(
    page: ft.Page,
    feature_key: str | None = None,
    on_activated: Callable[[], Any] | None = None,
) -> None:
    """Show the Pro upgrade / license activation dialog.

    Args:
        page: the Flet page to host the dialog.
        feature_key: optional gated feature (``files``/``rag``/``ocr``/
            ``watch``/``export``/``updates``); the matching checklist item
            is highlighted.
        on_activated: optional callback invoked (awaited when awaitable)
            after a successful activation so the caller can refresh the UI.
    """
    if page is None:
        return

    state = {"step": "main", "method": "wechat"}
    content = ft.Column(spacing=10, tight=True, scroll=ft.ScrollMode.AUTO)
    close_button = ft.TextButton(t("common.close"))
    dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text(t("license.upgrade_title")),
        content=ft.Container(content, width=_DIALOG_WIDTH),
        actions=[close_button],
    )

    def refresh() -> None:
        """Re-render the current step and update the dialog if mounted."""
        render()
        try:
            dialog.update()
        except Exception:  # noqa: BLE001 - not on the page yet
            pass

    # ------------------------------------------------------------------
    # Step renderers
    # ------------------------------------------------------------------

    def _feature_row(key: str) -> ft.Control:
        highlighted = feature_key == key
        return ft.Row(
            [
                ft.Icon(
                    ft.Icons.STAR if highlighted else ft.Icons.CHECK_CIRCLE_OUTLINE,
                    size=18,
                    color=ft.Colors.PRIMARY if highlighted else ft.Colors.GREEN,
                ),
                ft.Text(
                    t(f"license.feat_{key}"),
                    size=14,
                    weight=ft.FontWeight.BOLD if highlighted else None,
                ),
            ],
            spacing=8,
        )

    def render_main() -> None:
        async def on_wechat(e: ft.ControlEvent) -> None:
            state["method"] = "wechat"
            state["step"] = "pay"
            refresh()

        async def on_alipay(e: ft.ControlEvent) -> None:
            state["method"] = "alipay"
            state["step"] = "pay"
            refresh()

        async def on_enter_key(e: ft.ControlEvent) -> None:
            state["step"] = "activate"
            refresh()

        content.controls = [
            ft.Column([_feature_row(k) for k in _FEATURES], spacing=6, tight=True),
            ft.Divider(height=1),
            ft.Row(
                [
                    ft.Text(
                        t("license.price_monthly"),
                        size=20,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.PRIMARY,
                    ),
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            ft.Row(
                [
                    ft.ElevatedButton(
                        t("license.wechat_pay"),
                        icon=ft.Icons.PAYMENT,
                        on_click=on_wechat,
                    ),
                    ft.ElevatedButton(
                        t("license.alipay"),
                        icon=ft.Icons.PAYMENT,
                        on_click=on_alipay,
                    ),
                ],
                spacing=10,
            ),
            ft.TextButton(t("license.enter_key"), on_click=on_enter_key),
        ]

    def render_pay() -> None:
        method = state["method"]
        qr_src = "icons/wechat_qr.png" if method == "wechat" else "icons/alipay_qr.png"
        method_label = (
            t("license.wechat_pay") if method == "wechat" else t("license.alipay")
        )

        async def on_paid(e: ft.ControlEvent) -> None:
            state["step"] = "contact"
            refresh()

        async def on_back(e: ft.ControlEvent) -> None:
            state["step"] = "main"
            refresh()

        content.controls = [
            ft.Row(
                [
                    ft.Image(
                        src=qr_src,
                        width=220,
                        height=220,
                        fit=ft.BoxFit.CONTAIN,
                        border_radius=ft.BorderRadius.all(8),
                        error_content=ft.Icon(
                            ft.Icons.QR_CODE_2, size=120, color=ft.Colors.OUTLINE
                        ),
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            ft.Text(
                t("license.pay_steps", method=method_label),
                size=13,
                selectable=True,
            ),
            ft.Row(
                [
                    ft.TextButton(t("license.back"), on_click=on_back),
                    ft.ElevatedButton(t("license.i_paid"), on_click=on_paid),
                ],
                alignment=ft.MainAxisAlignment.END,
                spacing=10,
            ),
        ]

    def render_contact() -> None:
        async def on_copy(e: ft.ControlEvent) -> None:
            try:
                await ft.Clipboard().set(_WECHAT_ID)
                page.show_dialog(ft.SnackBar(ft.Text(t("license.copied"))))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Clipboard unavailable: %s", exc)

        async def on_activate_next(e: ft.ControlEvent) -> None:
            state["step"] = "activate"
            refresh()

        content.controls = [
            ft.Text(t("license.contact_title"), weight=ft.FontWeight.BOLD, size=15),
            ft.Text(t("license.contact_steps"), size=13, selectable=True),
            ft.Row(
                [
                    ft.Text(t("license.wechat_id"), size=14, selectable=True),
                    ft.IconButton(
                        icon=ft.Icons.COPY,
                        tooltip=t("license.copy"),
                        on_click=on_copy,
                    ),
                ],
                spacing=4,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            ft.Row(
                [
                    ft.ElevatedButton(
                        t("license.enter_key"), on_click=on_activate_next
                    ),
                ],
                alignment=ft.MainAxisAlignment.END,
            ),
        ]

    def render_activate() -> None:
        error_text = ft.Text("", size=13, color=ft.Colors.ERROR, visible=False)
        hint_text = ft.Text(t("license.key_hint"), size=12, color=ft.Colors.OUTLINE)
        key_field = ft.TextField(
            label=t("license.enter_key"),
            hint_text="FF3-XXXX-XXXX-XXXX",
            max_length=_KEY_MAX_LEN,
            dense=True,
            autofocus=True,
        )
        activate_button = ft.ElevatedButton(t("license.activate"))

        async def on_change(e: ft.ControlEvent) -> None:
            formatted = format_license_key(key_field.value or "")
            if key_field.value != formatted:
                key_field.value = formatted
            hint_text.value = t("license.key_hint")
            error_text.visible = False
            try:
                content.update()
            except Exception:  # noqa: BLE001
                pass

        async def on_activate(e: ft.ControlEvent) -> None:
            key = (key_field.value or "").strip()
            activate_button.disabled = True
            error_text.visible = False
            try:
                dialog.update()
            except Exception:  # noqa: BLE001
                pass
            try:
                from filefinder.core import license as license_core

                ok, code = await asyncio.to_thread(license_core.activate, key)
            except Exception as exc:  # noqa: BLE001
                logger.error("License activation call failed: %s", exc)
                ok, code = False, "invalid_format"
            if ok:
                state["step"] = "success"
                refresh()
                if on_activated is not None:
                    try:
                        await maybe_await(on_activated())
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("on_activated callback failed: %s", exc)
                return
            error_text.value = t(
                f"license.error_{code}"
                if code in _ERROR_CODES
                else "license.error_invalid_format"
            )
            error_text.visible = True
            activate_button.disabled = False
            try:
                dialog.update()
            except Exception:  # noqa: BLE001
                pass

        async def on_back(e: ft.ControlEvent) -> None:
            state["step"] = "main"
            refresh()

        key_field.on_change = on_change
        activate_button.on_click = on_activate
        content.controls = [
            key_field,
            hint_text,
            error_text,
            ft.Row(
                [
                    ft.TextButton(t("license.back"), on_click=on_back),
                    activate_button,
                ],
                alignment=ft.MainAxisAlignment.END,
                spacing=10,
            ),
        ]

    def render_success() -> None:
        async def on_start(e: ft.ControlEvent) -> None:
            page.pop_dialog()

        content.controls = [
            ft.Row(
                [
                    ft.Text("🎉", size=36),
                    ft.Text(
                        t("license.activated"),
                        size=18,
                        weight=ft.FontWeight.BOLD,
                    ),
                ],
                spacing=10,
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            ft.Text(t("license.activated_body"), size=13),
            ft.Row(
                [ft.ElevatedButton(t("license.start"), on_click=on_start)],
                alignment=ft.MainAxisAlignment.END,
            ),
        ]

    def render() -> None:
        step = state["step"]
        if step == "pay":
            render_pay()
        elif step == "contact":
            render_contact()
        elif step == "activate":
            render_activate()
        elif step == "success":
            render_success()
        else:
            render_main()

    async def on_close(e: ft.ControlEvent) -> None:
        page.pop_dialog()

    close_button.on_click = on_close
    render()
    page.show_dialog(dialog)
