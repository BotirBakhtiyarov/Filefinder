"""Chat view: conversation list, provider selector, RAG toggle, input row."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import flet as ft

from filefinder.core import chat as chat_core
from filefinder.core.config import get_config, save_config, t
from filefinder.ui import is_pro, kv_get, kv_set
from filefinder.ui.components.chat_bubble import ChatBubble, split_thinking

if TYPE_CHECKING:  # pragma: no cover
    from filefinder.ui.app import FileFinderApp

logger = logging.getLogger(__name__)

_PROVIDERS = list(chat_core.PROVIDER_CHOICES)
_KV_CONVERSATION = "current_conversation"


def _normalize_history_item(item: Any) -> tuple[str, str]:
    """Extract ``(role, content)`` from a chat history record.

    Tolerates dicts, tuples and ORM-like objects so the view works with the
    ``Database``/``ChatEngine`` history format whatever its exact shape.
    """
    if isinstance(item, dict):
        return str(item.get("role", "assistant")), str(item.get("content", ""))
    if isinstance(item, (tuple, list)):
        if len(item) >= 4:  # (id, conversation_id, role, content, ...)
            return str(item[2]), str(item[3])
        if len(item) >= 2:
            return str(item[0]), str(item[1])
    role = getattr(item, "role", "assistant")
    content = getattr(item, "content", "")
    return str(role), str(content)


class ChatView(ft.Column):
    """RAG chat view.

    Args:
        app: owning :class:`FileFinderApp` (used for snackbars).
    """

    def __init__(self, app: FileFinderApp) -> None:
        self._app = app
        cfg = get_config()
        self._engine = self._create_engine()
        self._conversation_id: str | None = None

        self.provider_dropdown = ft.Dropdown(
            label=t("chat.provider"),
            value=cfg.llm_provider if cfg.llm_provider in _PROVIDERS else "deepseek",
            options=[
                ft.dropdown.Option(p, chat_core.PROVIDER_LABELS.get(p, p))
                for p in _PROVIDERS
            ],
            on_select=self._on_provider_change,
            width=170,
            dense=True,
        )
        self.rag_switch = ft.Switch(
            label=t("chat.rag"),
            value=True,
            on_change=self._on_rag_toggle,
        )
        self.new_chat_button = ft.TextButton(
            content=ft.Row(
                [ft.Icon(ft.Icons.ADD_COMMENT, size=16), ft.Text(t("chat.new_chat"))],
                spacing=6,
                tight=True,
            ),
            on_click=self._on_new_chat,
        )
        self.messages = ft.ListView(expand=True, spacing=8, auto_scroll=True)

        # File attachments (available when RAG is on).
        self._attachments: list[str] = []
        self._picker = ft.FilePicker()
        self.attach_button = ft.IconButton(
            icon=ft.Icons.ADD,
            tooltip=t("chat.attach"),
            on_click=self._on_attach,
            visible=True,
        )
        self.attachment_chips = ft.Row(spacing=6, wrap=True, visible=False)

        self.input_field = ft.TextField(
            hint_text=t("chat.input_placeholder"),
            expand=True,
            multiline=True,
            min_lines=1,
            max_lines=4,
            shift_enter=True,
            on_submit=self._on_send,
            dense=True,
        )
        self.send_button = ft.IconButton(
            icon=ft.Icons.SEND,
            tooltip=t("chat.send"),
            on_click=self._on_send,
        )

        super().__init__(
            controls=[
                ft.Row(
                    [self.provider_dropdown, self.rag_switch, self.new_chat_button],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Divider(height=1),
                self.messages,
                self.attachment_chips,
                ft.Row(
                    [self.attach_button, self.input_field, self.send_button],
                    vertical_alignment=ft.CrossAxisAlignment.END,
                ),
            ],
            expand=True,
            spacing=10,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def did_mount(self) -> None:
        """Load the current conversation once the control is on the page."""
        services = getattr(self.page, "services", None)
        if services is not None:
            if self._picker not in services:
                services.append(self._picker)
        elif self._picker not in self.page.overlay:
            self.page.overlay.append(self._picker)
        self._apply_tier_state()
        self.page.run_task(self._load_history)

    # ------------------------------------------------------------------
    # Freemium gating
    # ------------------------------------------------------------------

    def _apply_tier_state(self) -> None:
        """Disable the RAG switch for free-tier users.

        Called on mount; the app recreates this view after any tier
        change, so the state is re-applied automatically.
        """
        if is_pro():
            return
        self.rag_switch.value = False
        self.rag_switch.disabled = True
        self.rag_switch.tooltip = t("license.pro_only")
        self.attach_button.visible = False

    @staticmethod
    def _chat_quota_warning() -> str | None:
        """Return the low-quota warning text for free users, if any.

        Reads ``chat_queries`` counter/limit from the license module;
        the warning appears once usage reaches 15 queries.
        """
        if is_pro():
            return None
        try:
            from filefinder.core import license as license_core

            limit = license_core.get_limit("chat_queries")
            used = license_core.get_counter("chat_queries")
        except Exception as exc:  # noqa: BLE001
            logger.debug("chat quota unavailable: %s", exc)
            return None
        if not isinstance(limit, int) or limit <= 0:
            return None
        remaining = max(0, limit - used)
        if used >= 15 and remaining > 0:
            return t("license.chat_warning", remaining=remaining)
        return None

    # ------------------------------------------------------------------
    # Engine / conversation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_engine() -> Any:
        """Return the chat engine singleton (constructing one if needed)."""
        from filefinder.core import chat as chat_core

        factory = getattr(chat_core, "get_chat_engine", None)
        if callable(factory):
            return factory()
        return chat_core.ChatEngine()

    def _ensure_conversation_id(self) -> str:
        """Return the active conversation id, creating one if necessary."""
        if self._conversation_id:
            return self._conversation_id
        conv_id: str | None = None
        try:
            from filefinder.core.db import get_db

            conv_id = kv_get(get_db(), _KV_CONVERSATION)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read current conversation id: %s", exc)
        if not conv_id:
            conv_id = self._engine.new_conversation()
            self._store_conversation_id(conv_id)
        self._conversation_id = conv_id
        return conv_id

    def _store_conversation_id(self, conv_id: str) -> None:
        try:
            from filefinder.core.db import get_db

            kv_set(get_db(), _KV_CONVERSATION, conv_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist conversation id: %s", exc)

    async def _load_history(self) -> None:
        """Load and render the persisted history of the current conversation."""
        try:
            conv_id = await asyncio.to_thread(self._ensure_conversation_id)
            history = await asyncio.to_thread(self._engine.get_history, conv_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load chat history: %s", exc)
            return
        # Clear first — did_mount fires again when the view is re-mounted
        # (navigating away and back), which would duplicate every bubble.
        self.messages.controls.clear()
        for item in history or []:
            role, content = _normalize_history_item(item)
            if content:
                if role == "assistant":
                    thinking, answer = split_thinking(content)
                    self.messages.controls.append(
                        ChatBubble(
                            role,
                            answer,
                            thinking=thinking,
                            think_label=t("chat.thinking"),
                        )
                    )
                else:
                    self.messages.controls.append(ChatBubble(role, content))
        if self.messages.controls:
            self.update()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_provider_change(self, e: ft.ControlEvent) -> None:
        cfg = get_config()
        cfg.llm_provider = e.control.value
        try:
            save_config(cfg)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save provider choice: %s", exc)
            self._app.show_snackbar(t("error.generic", error=exc), error=True)

    async def _on_new_chat(self, e: ft.ControlEvent) -> None:
        try:
            conv_id = await asyncio.to_thread(self._engine.new_conversation)
        except Exception as exc:  # noqa: BLE001
            self._app.show_snackbar(t("error.generic", error=exc), error=True)
            return
        self._conversation_id = conv_id
        self._store_conversation_id(conv_id)
        self._attachments.clear()
        self._refresh_attachment_chips()
        self.messages.controls.clear()
        self.update()

    # ------------------------------------------------------------------
    # Attachments (RAG file upload)
    # ------------------------------------------------------------------

    async def _on_rag_toggle(self, e: ft.ControlEvent) -> None:
        self.attach_button.visible = bool(self.rag_switch.value)
        if not self.rag_switch.value:
            self._attachments.clear()
            self._refresh_attachment_chips()
        self.update()

    async def _on_attach(self, e: ft.ControlEvent) -> None:
        """Pick files to attach as extra context for the next message."""
        try:
            picked = await self._picker.pick_files(
                dialog_title=t("chat.attach"),
                allow_multiple=True,
            )
        except Exception as exc:  # noqa: BLE001
            self._app.show_snackbar(t("error.generic", error=exc), error=True)
            return
        for f in picked or []:
            path = getattr(f, "path", None)
            if path and path not in self._attachments:
                self._attachments.append(path)
        self._refresh_attachment_chips()
        self.update()

    def _refresh_attachment_chips(self) -> None:
        """Re-render the attachment chips row above the input."""
        from pathlib import Path

        chips: list[ft.Control] = []
        for path in self._attachments:
            chips.append(
                ft.Chip(
                    label=ft.Text(Path(path).name),
                    on_delete=lambda ev, p=path: self._remove_attachment(p),
                )
            )
        self.attachment_chips.controls = chips
        self.attachment_chips.visible = bool(chips)

    def _remove_attachment(self, path: str) -> None:
        if path in self._attachments:
            self._attachments.remove(path)
        self._refresh_attachment_chips()
        self.update()

    # ------------------------------------------------------------------
    # Sending (streaming)
    # ------------------------------------------------------------------

    async def _on_chunk(
        self, bubble: ChatBubble, channel: str, chunk: str
    ) -> None:
        """Apply one streamed chunk to the bubble (runs on the UI task)."""
        if channel == "think":
            bubble.append_thinking(chunk)
        else:
            bubble.append_answer(chunk)

    async def _on_send(self, e: ft.ControlEvent) -> None:
        text = (self.input_field.value or "").strip()
        if not text:
            return
        # Free tier: RAG is a Pro feature — gate it even if the switch
        # somehow stayed enabled.
        if self.rag_switch.value and not self._app.require_pro("rag"):
            return
        self.input_field.value = ""
        self.send_button.disabled = True

        attachments = list(self._attachments)
        self._attachments.clear()
        self._refresh_attachment_chips()
        display_text = text
        if attachments:
            names = ", ".join(Path(p).name for p in attachments)
            display_text = f"{text}\n📎 {names}"
        self.messages.controls.append(ChatBubble("user", display_text))

        bubble = ChatBubble("assistant", "", think_label=t("chat.thinking"))
        bubble.set_text(t("chat.typing"))
        self.messages.controls.append(bubble)
        self.update()

        def _stream_worker() -> None:
            first_chunk = True
            for channel, chunk in self._engine.send_message_stream(
                text,
                conv_id,
                bool(self.rag_switch.value),
                attachments or None,
            ):
                if first_chunk and channel == "answer":
                    bubble.set_text("")  # clear the typing indicator
                    first_chunk = False
                self.page.run_task(self._on_chunk, bubble, channel, chunk)

        try:
            conv_id = await asyncio.to_thread(self._ensure_conversation_id)
            await asyncio.to_thread(_stream_worker)
        except Exception as exc:
            logger.exception("Chat send failed")
            self._app.show_snackbar(t("chat.error", error=exc), error=True)
            if not bubble.current_text:
                bubble.set_text(t("chat.llm_unavailable"))
        finally:
            self.send_button.disabled = False
            self.update()
        try:
            warning = await asyncio.to_thread(self._chat_quota_warning)
        except Exception as exc:  # noqa: BLE001
            logger.debug("quota check failed: %s", exc)
            warning = None
        if warning:
            self._app.show_snackbar(warning)
