"""RAG chat engine.

Builds prompts from retrieved search context plus recent conversation
history and routes them to the configured LLM provider (DeepSeek cloud
with a built-in key, a local Ollama server, or a custom OpenAI-compatible
endpoint). All failures degrade to the localized ``chat.llm_unavailable``
string instead of raising.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any

from filefinder.core import constants
from filefinder.core import license as license_mod
from filefinder.core.config import get_config, t
from filefinder.core.db import get_db

logger = logging.getLogger(__name__)

# Number of recent history messages included in the prompt.
_HISTORY_LIMIT = 10

# Provider defaults.
_OLLAMA_DEFAULT_URL = "http://localhost:11434"
_OLLAMA_DEFAULT_MODEL = "qwen2.5:3b"
_DEEPSEEK_DEFAULT_URL = "https://api.deepseek.com"
_DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"

# Provider presets: URL is applied automatically; DeepSeek uses the
# built-in (obfuscated) API key so the user never enters one, Ollama is
# local and keyless, and "custom" takes a manual base URL + key + model.
# All non-Ollama providers use the OpenAI-compatible /chat/completions API.
LLM_PROVIDERS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "url": _DEEPSEEK_DEFAULT_URL,
        "model": _DEEPSEEK_DEFAULT_MODEL,
        "needs_key": False,
        "builtin": True,
        "kind": "openai",
    },
    "ollama": {
        "url": _OLLAMA_DEFAULT_URL,
        "model": _OLLAMA_DEFAULT_MODEL,
        "needs_key": False,
        "builtin": False,
        "kind": "ollama",
    },
    "custom": {
        "url": "",
        "model": "",
        "needs_key": True,
        "builtin": False,
        "kind": "openai",
    },
}

# Providers selectable in the UI dropdowns ("off" disables the LLM).
PROVIDER_CHOICES: list[str] = ["deepseek", "ollama", "custom", "off"]

# Human-readable labels for the provider dropdowns.
PROVIDER_LABELS: dict[str, str] = {
    "deepseek": "DeepSeek (Cloud)",
    "ollama": "Ollama (local)",
    "custom": "Custom",
    "off": "Off",
}

_REQUEST_TIMEOUT = 120

# Conversation title length derived from the first user message.
_TITLE_LENGTH = 40

# Attachment caps for chat file uploads (characters, not tokens).
_ATTACHMENT_MAX_CHARS = 4000
_ATTACHMENT_TOTAL_MAX = 12000


class LLMUnavailableError(Exception):
    """Raised when the configured LLM provider cannot serve a request."""


# ----------------------------------------------------------------------
# Database helpers (defensive over the Database facade's chat CRUD)
# ----------------------------------------------------------------------

def _add_message(db: Any, conversation_id: str, role: str, content: str) -> None:
    """Append a chat message via whichever facade method exists."""
    for name in ("add_chat_message", "add_message"):
        method = getattr(db, name, None)
        if callable(method):
            method(conversation_id, role, content)
            return
    raise AttributeError("Database facade has no chat insert method")


def _fetch_history(db: Any, conversation_id: str) -> list[dict[str, Any]]:
    """Fetch chat history rows as dicts via whichever facade method exists."""
    for name in ("get_chat_history", "get_history", "get_messages"):
        method = getattr(db, name, None)
        if callable(method):
            rows = method(conversation_id) or []
            result: list[dict[str, Any]] = []
            for row in rows:
                if isinstance(row, dict):
                    result.append(row)
                else:
                    try:
                        result.append(dict(row))
                    except (TypeError, ValueError):
                        continue
            return result
    return []


def _fetch_conversations(db: Any) -> list[dict[str, Any]]:
    """Fetch conversation summaries via whichever facade method exists."""
    for name in ("list_conversations", "get_conversations"):
        method = getattr(db, name, None)
        if callable(method):
            rows = method() or []
            result: list[dict[str, Any]] = []
            for row in rows:
                if isinstance(row, dict):
                    result.append(row)
                else:
                    try:
                        result.append(dict(row))
                    except (TypeError, ValueError):
                        continue
            return result
    return []


# ----------------------------------------------------------------------
# Chat engine
# ----------------------------------------------------------------------

class ChatEngine:
    """RAG chat over the indexed corpus with pluggable LLM providers."""

    # --------------------------------------------------------------
    # Conversations / history
    # --------------------------------------------------------------

    def new_conversation(self) -> str:
        """Create a new conversation id (uuid4 hex)."""
        conversation_id = uuid.uuid4().hex
        logger.debug("New conversation %s", conversation_id)
        return conversation_id

    def get_history(self, conversation_id: str) -> list[dict[str, Any]]:
        """Return all messages of a conversation in chronological order."""
        return _fetch_history(get_db(), conversation_id)

    def list_conversations(self) -> list[dict[str, Any]]:
        """List conversations, latest first, titled by the first user message.

        Returns:
            List of dicts with ``conversation_id``, ``title`` and
            ``timestamp`` (last activity, if available) keys.
        """
        conversations = _fetch_conversations(get_db())
        normalized: list[dict[str, Any]] = []
        for row in conversations:
            conversation_id = str(
                row.get("conversation_id", row.get("id", ""))
            )
            if not conversation_id:
                continue
            title = row.get("title")
            if not title:
                title = self._derive_title(conversation_id)
            normalized.append(
                {
                    "conversation_id": conversation_id,
                    "title": title,
                    "timestamp": row.get(
                        "timestamp", row.get("last_timestamp", row.get("updated_at"))
                    ),
                }
            )

        def _sort_key(item: dict[str, Any]) -> float:
            try:
                return float(item.get("timestamp") or 0.0)
            except (TypeError, ValueError):
                return 0.0

        normalized.sort(key=_sort_key, reverse=True)
        return normalized

    def _derive_title(self, conversation_id: str) -> str:
        """Build a conversation title from its first user message."""
        for message in _fetch_history(get_db(), conversation_id):
            if message.get("role") == "user":
                content = str(message.get("content", "")).strip().splitlines()
                if content and content[0]:
                    title = content[0]
                    if len(title) > _TITLE_LENGTH:
                        title = title[:_TITLE_LENGTH].rstrip() + "…"
                    return title
        return conversation_id[:8]

    # --------------------------------------------------------------
    # RAG context & prompt
    # --------------------------------------------------------------

    def _build_rag_context(self, text: str) -> str:
        """Search the index and format top results as a context block.

        Search failures are logged and yield an empty context so chat
        still works without an index or embedding model.
        """
        try:
            from filefinder.core.search import get_search_engine

            results = get_search_engine().search(text, top_k=constants.RAG_TOP_K)
        except Exception as exc:  # noqa: BLE001 - chat must not crash
            logger.warning("RAG search failed: %s", exc)
            return ""
        if not results:
            return ""
        blocks = []
        for result in results:
            blocks.append(f"[{result.path}]\n{result.snippet}")
        return "\n\n".join(blocks)

    def _build_attachment_context(self, paths: list[str]) -> str:
        """Extract text from user-attached files for the prompt context.

        Files are parsed with :mod:`filefinder.core.file_parser` (text,
        office docs, PDF with OCR fallback, images via OCR) and capped so
        the prompt stays within reasonable size.
        """
        from filefinder.core import file_parser

        blocks: list[str] = []
        total = 0
        for path in paths or []:
            try:
                text = (file_parser.extract_text(path) or "").strip()
            except Exception as exc:  # noqa: BLE001 - skip unreadable files
                logger.warning("Attachment parse failed for %s: %s", path, exc)
                continue
            if not text:
                text = "(no extractable text)"
            text = text[:_ATTACHMENT_MAX_CHARS]
            block = f"[attachment: {path}]\n{text}"
            blocks.append(block)
            total += len(block)
            if total >= _ATTACHMENT_TOTAL_MAX:
                break
        return "\n\n".join(blocks)

    def _compose_context(
        self, text: str, use_rag: bool, attachment_paths: list[str] | None
    ) -> str:
        """Combine RAG search results and attached files into one context."""
        parts = []
        if use_rag:
            rag = self._build_rag_context(text)
            if rag:
                parts.append(rag)
        if attachment_paths:
            attachments = self._build_attachment_context(attachment_paths)
            if attachments:
                parts.append(attachments)
        return "\n\n".join(parts)

    def _build_prompt(
        self,
        text: str,
        context: str,
        history: list[dict[str, Any]],
    ) -> str:
        """Assemble the full prompt (English scaffold, i18n-agnostic)."""
        parts = [
            ("You are FileFinder, a helpful assistant that answers questions "
            "about the user's local files. Answer in the same language as "
            "the user's message. Be concise."),
        ]
        if context:
            parts.append(
                "Relevant file excerpts:\n" + context
            )
        recent = history[-_HISTORY_LIMIT:]
        if recent:
            lines = []
            for message in recent:
                role = str(message.get("role", "user"))
                content = str(message.get("content", ""))
                speaker = "User" if role == "user" else "Assistant"
                lines.append(f"{speaker}: {content}")
            parts.append("Conversation so far:\n" + "\n".join(lines))
        parts.append(f"User: {text}\nAssistant:")
        return "\n\n".join(parts)

    # --------------------------------------------------------------
    # LLM routing
    # --------------------------------------------------------------

    def _call_ollama(self, prompt: str, api_url: str, model: str) -> str:
        """Call a local Ollama server via ``/api/generate``."""
        import requests  # lazy import

        response = requests.post(
            f"{api_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        answer = str(payload.get("response", "")).strip()
        if not answer:
            raise LLMUnavailableError("Ollama returned an empty response")
        return answer

    def _call_openai_compatible(
        self, prompt: str, api_url: str, model: str, api_key: str
    ) -> str:
        """Call an OpenAI-compatible endpoint via ``/chat/completions``.

        Works for the built-in DeepSeek provider and any custom
        OpenAI-compatible base URL.
        """
        import requests  # lazy import

        response = requests.post(
            f"{api_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        try:
            answer = str(
                payload["choices"][0]["message"]["content"]
            ).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailableError(
                f"Unexpected chat-completions response shape: {exc}"
            ) from exc
        if not answer:
            raise LLMUnavailableError("LLM returned an empty response")
        return answer

    # Backwards-compatible alias.
    _call_deepseek = _call_openai_compatible

    def _resolve_provider(self) -> tuple[dict[str, Any], str, str, str]:
        """Resolve ``(preset, api_url, model, api_key)`` from the config.

        The base URL and default model come from :data:`LLM_PROVIDERS`
        presets; the user's config values override them (mainly for the
        ``custom`` provider or self-hosted endpoints). Removed/unknown
        providers (e.g. ``openai``/``gemini``/``kimi`` from older configs)
        fall back to ``deepseek``; its API key comes from the built-in
        obfuscated blob, never from the user config. The user-entered
        ``llm_api_key`` is only used for the ``custom`` provider.

        Raises:
            LLMUnavailableError: provider is off or misconfigured.
        """
        config = get_config()
        provider = (getattr(config, "llm_provider", "off") or "off").lower()

        if provider == "off":
            raise LLMUnavailableError("LLM provider is off")
        preset = LLM_PROVIDERS.get(provider)
        if preset is None:
            # Stale provider from an older config: treat as deepseek.
            logger.info("Unknown LLM provider %r; falling back to deepseek", provider)
            provider = "deepseek"
            preset = LLM_PROVIDERS["deepseek"]

        # A stored URL that matches ANOTHER provider's preset is stale
        # (left over from a previous provider choice) — ignore it.
        config_url = getattr(config, "llm_api_url", "") or ""
        known_urls = {p["url"] for p in LLM_PROVIDERS.values()}
        if provider != "custom" and (not config_url or config_url in known_urls):
            api_url = preset["url"]
        else:
            api_url = config_url or preset["url"]
        # Same for a stored model left over from another provider.
        config_model = getattr(config, "llm_model", "") or ""
        known_models = {p["model"] for p in LLM_PROVIDERS.values()}
        if provider != "custom" and (
            not config_model or config_model in known_models
        ):
            model = preset["model"]
        else:
            model = config_model or preset["model"]
        if not api_url:
            raise LLMUnavailableError(f"No API URL configured for {provider}")
        if not model:
            raise LLMUnavailableError(f"No model configured for {provider}")
        if preset.get("builtin"):
            from filefinder.core.builtin_key import get_builtin_llm_key  # lazy

            api_key = get_builtin_llm_key()
            if not api_key:
                raise LLMUnavailableError("built-in LLM key not configured")
        else:
            api_key = getattr(config, "llm_api_key", "") or ""
            if preset["needs_key"] and not api_key:
                raise LLMUnavailableError(f"{provider} API key is not configured")
        return preset, api_url, model, api_key

    def _generate(self, prompt: str) -> str:
        """Route the prompt to the configured provider (non-streaming).

        Raises:
            LLMUnavailableError: provider is off, misconfigured or failing.
        """
        preset, api_url, model, api_key = self._resolve_provider()
        if preset["kind"] == "ollama":
            return self._call_ollama(prompt, api_url, model)
        return self._call_openai_compatible(prompt, api_url, model, api_key)

    # --------------------------------------------------------------
    # Streaming
    # --------------------------------------------------------------

    def _stream_ollama(self, prompt: str, api_url: str, model: str) -> Any:
        """Yield raw text chunks from Ollama's streaming ``/api/generate``."""
        import json

        import requests  # lazy import

        with requests.post(
            f"{api_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": True},
            stream=True,
            timeout=_REQUEST_TIMEOUT,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                chunk = payload.get("response", "")
                if chunk:
                    yield str(chunk)
                # Newer Ollama puts reasoning in a separate "thinking" field
                # (deepseek-r1, qwen3) instead of inline <think> tags.
                thinking = payload.get("thinking", "")
                if thinking:
                    yield ("think", str(thinking))

    def _stream_openai_compatible(
        self, prompt: str, api_url: str, model: str, api_key: str
    ) -> Any:
        """Yield ``(channel, chunk)`` from an OpenAI-compatible SSE stream.

        ``channel`` is ``"think"`` for ``reasoning_content`` deltas
        (DeepSeek-R1 style thinking, etc.) and ``"answer"`` otherwise.
        """
        import json

        import requests  # lazy import

        with requests.post(
            f"{api_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
            stream=True,
            timeout=_REQUEST_TIMEOUT,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", "replace")
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                    delta = payload["choices"][0]["delta"]
                except (ValueError, KeyError, IndexError, TypeError):
                    continue
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    yield ("think", str(reasoning))
                content = delta.get("content")
                if content:
                    yield ("answer", str(content))

    @staticmethod
    def _parse_think_tags(chunks: Any) -> Any:
        """Split a raw chunk stream into ``("think"|"answer", text)``.

        Handles ``<think>…</think>`` tags even when split across chunks.
        """
        buffer = ""
        channel = "answer"
        for raw in chunks:
            if isinstance(raw, tuple):  # already channelled
                yield raw
                continue
            buffer += str(raw)
            while True:
                tag = "</think>" if channel == "think" else "<think>"
                idx = buffer.find(tag)
                if idx >= 0:
                    piece = buffer[:idx]
                    buffer = buffer[idx + len(tag):]
                    if piece:
                        yield (channel, piece)
                    channel = "answer" if channel == "think" else "think"
                    continue
                # Keep a hold-back so a split tag is not emitted early.
                hold = len(tag) - 1
                if len(buffer) > hold:
                    piece, buffer = buffer[:-hold], buffer[-hold:]
                    if piece:
                        yield (channel, piece)
                break
        if buffer:
            yield (channel, buffer)

    def _generate_stream(self, prompt: str) -> Any:
        """Yield ``(channel, chunk)`` tuples from the configured provider."""
        preset, api_url, model, api_key = self._resolve_provider()
        if preset["kind"] == "ollama":
            yield from self._parse_think_tags(
                self._stream_ollama(prompt, api_url, model)
            )
        else:
            yield from self._parse_think_tags(
                self._stream_openai_compatible(prompt, api_url, model, api_key)
            )

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------

    def send_message(
        self,
        text: str,
        conversation_id: str | None = None,
        use_rag: bool = True,
        attachment_paths: list[str] | None = None,
    ) -> str:
        """Send a user message and return the assistant's reply.

        Persists both messages to chat history. Any LLM failure (provider
        off, network error, bad response) results in the localized
        ``chat.llm_unavailable`` string rather than an exception.

        Args:
            text: the user's message.
            conversation_id: existing conversation; a new one is created
                when omitted.
            use_rag: whether to retrieve index context for the prompt.
            attachment_paths: files whose text content is added to the
                prompt context.

        Returns:
            The assistant's reply text.
        """
        text = (text or "").strip()
        if not text:
            return ""
        if not conversation_id:
            conversation_id = self.new_conversation()

        # Free tier: RAG is a premium feature and chat queries are capped.
        if use_rag and not license_mod.check_feature("rag"):
            logger.info("RAG chat rejected: premium feature on free tier")
            return t("license.pro_only")
        if not license_mod.is_pro():
            count = license_mod.increment_counter("chat_queries")
            limit = license_mod.get_limit("chat_queries")
            if limit is not None and count > limit:
                logger.info("Free-tier chat query limit exceeded")
                return t("license.chat_limit")

        db = get_db()
        history = _fetch_history(db, conversation_id)

        context = self._compose_context(text, use_rag, attachment_paths)
        prompt = self._build_prompt(text, context, history)

        try:
            answer = self._generate(prompt)
        except Exception as exc:  # noqa: BLE001 - any failure -> localized msg
            logger.warning("LLM unavailable: %s", exc)
            answer = t("chat.llm_unavailable")

        try:
            _add_message(db, conversation_id, "user", text)
            _add_message(db, conversation_id, "assistant", answer)
        except Exception as exc:  # noqa: BLE001 - persistence must not crash chat
            logger.error("Failed to persist chat messages: %s", exc)

        return answer

    def send_message_stream(
        self,
        text: str,
        conversation_id: str | None = None,
        use_rag: bool = True,
        attachment_paths: list[str] | None = None,
    ) -> Any:
        """Streaming variant of :meth:`send_message`.

        A generator yielding ``(channel, chunk)`` tuples where ``channel``
        is ``"think"`` (reasoning) or ``"answer"``. Both messages are
        persisted when the stream completes. On any provider failure a
        single ``("answer", t("chat.llm_unavailable"))`` is yielded.

        Returns:
            A generator; its ``.gi_frame``-local value is not used — the
            final conversation id is persisted identically to
            :meth:`send_message`.
        """
        text = (text or "").strip()
        if not text:
            return
        if not conversation_id:
            conversation_id = self.new_conversation()

        # Free tier: RAG is a premium feature and chat queries are capped.
        if use_rag and not license_mod.check_feature("rag"):
            logger.info("RAG chat rejected: premium feature on free tier")
            yield ("answer", t("license.pro_only"))
            return
        if not license_mod.is_pro():
            count = license_mod.increment_counter("chat_queries")
            limit = license_mod.get_limit("chat_queries")
            if limit is not None and count > limit:
                logger.info("Free-tier chat query limit exceeded")
                yield ("answer", t("license.chat_limit"))
                return

        db = get_db()
        history = _fetch_history(db, conversation_id)
        context = self._compose_context(text, use_rag, attachment_paths)
        prompt = self._build_prompt(text, context, history)

        answer_parts: list[str] = []
        think_parts: list[str] = []
        failed = False
        try:
            for channel, chunk in self._generate_stream(prompt):
                if channel == "think":
                    think_parts.append(chunk)
                else:
                    answer_parts.append(chunk)
                yield (channel, chunk)
        except Exception as exc:  # noqa: BLE001 - any failure -> localized msg
            logger.warning("LLM stream failed: %s", exc)
            failed = True

        answer = "".join(answer_parts).strip()
        thinking = "".join(think_parts).strip()
        if failed or not answer:
            answer = t("chat.llm_unavailable")
            yield ("answer", answer)

        # Persist with the thinking embedded so a reload can re-render it.
        stored = f"<think>{thinking}</think>\n{answer}" if thinking else answer
        try:
            _add_message(db, conversation_id, "user", text)
            _add_message(db, conversation_id, "assistant", stored)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to persist chat messages: %s", exc)


_engine: ChatEngine | None = None
_engine_lock = threading.Lock()


def get_chat_engine() -> ChatEngine:
    """Return the module-level :class:`ChatEngine` singleton."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = ChatEngine()
    return _engine
