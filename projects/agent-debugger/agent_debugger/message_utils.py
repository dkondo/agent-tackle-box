"""Shared helpers for normalizing message payloads across adb modules."""

from __future__ import annotations

import json
from typing import Any

_TYPE_ALIASES = {
    "assistant": "ai",
    "user": "human",
}


def message_type(msg: Any) -> str:
    """Return a canonical message type string."""
    raw = ""
    if isinstance(msg, dict):
        raw = str(msg.get("type") or msg.get("role") or "")
    else:
        raw = str(getattr(msg, "type", "") or getattr(msg, "role", ""))
    raw = raw.lower()
    if not raw:
        return "unknown"
    return _TYPE_ALIASES.get(raw, raw)


def message_content(msg: Any) -> Any:
    """Return message content regardless of payload shape."""
    if isinstance(msg, dict):
        return msg.get("content", "")
    return getattr(msg, "content", "")


def message_tool_calls(msg: Any) -> list[Any]:
    """Return message tool calls as a list."""
    if isinstance(msg, dict):
        calls = msg.get("tool_calls", [])
    else:
        calls = getattr(msg, "tool_calls", [])
    return calls if isinstance(calls, list) else []


def message_name(msg: Any, default: str = "tool") -> str:
    """Return message name regardless of payload shape."""
    if isinstance(msg, dict):
        return str(msg.get("name", default))
    return str(getattr(msg, "name", default))


def content_to_text(content: Any) -> str:
    """Normalize message content into readable plain text."""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                parts.append(str(text) if text is not None else str(item))
            else:
                parts.append(str(item))
        return " ".join(parts).strip()
    return str(content)


def extract_chat_text(content: Any) -> str | None:
    """Extract user-facing text from JSON-shaped content.

    Returns the value of a "text" field when content is a dict with a non-empty
    string "text" key, or a JSON-string that parses to such a dict. Returns
    None otherwise (caller should fall back to its own default). Empty strings
    return None so callers preserve their original fallback path instead of
    silently rendering nothing when a payload has blank text plus useful metadata.
    """
    if isinstance(content, str):
        stripped = content.lstrip()
        if not stripped.startswith("{"):
            return None
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return None
        content = parsed
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str) and text:
            return text
    return None
