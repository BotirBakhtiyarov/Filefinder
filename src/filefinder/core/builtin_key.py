"""Built-in LLM key (obfuscated blob).

PUBLIC BUILD: the blob is intentionally empty. The real built-in key only
exists in the private repository and in official releases.
"""

from __future__ import annotations

_BLOB: list[int] = []


def get_builtin_llm_key() -> str:
    """Return the built-in LLM API key (empty in public builds)."""
    if not _BLOB:
        return ""
    return bytes(b ^ 0x5A for b in _BLOB).decode("utf-8", errors="ignore")
