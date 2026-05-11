"""Messages panel: full message history."""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.widgets import RichLog

from agent_debugger.message_utils import (
    content_to_text,
    message_content,
    message_name,
    message_tool_calls,
    message_type,
)


class MessagesPanel(RichLog):
    """Panel showing full message history."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the messages panel with line-wrapping enabled.

        Long structured content (e.g. dict-shaped AI message bodies with
        `recommendations` lists) would otherwise be clipped at the viewport
        width. ``wrap=True`` lets RichLog flow content onto multiple lines at
        the actual viewport width when rendered. We intentionally do NOT
        override ``write()`` to pre-compute the width: RichLog's region size
        is unreliable when the panel sits inside a TabbedContent that hasn't
        been fully laid out yet, and passing a stale width=1 makes every
        character wrap onto its own line.
        """
        kwargs.setdefault("wrap", True)
        super().__init__(**kwargs)

    def update_messages(self, messages: list[Any]) -> None:
        """Update the messages display."""
        self.clear()
        user_count = 0

        for msg in messages:
            msg_type = message_type(msg)
            content = content_to_text(message_content(msg))
            tool_calls = message_tool_calls(msg)
            name = message_name(msg, "tool")

            if msg_type == "human":
                user_count += 1
                preview = content if len(content) <= 40 else f"{content[:37]}..."
                self.write(
                    Text(
                        f'Messages after user #{user_count}: "{preview}"',
                        style="bold yellow",
                    )
                )
                self.write(Text(f"$ {content}", style="green"))
            elif msg_type == "ai":
                if tool_calls:
                    for tc in tool_calls:
                        tc_name = (
                            tc.get("name", "tool")
                            if isinstance(tc, dict)
                            else getattr(tc, "name", "tool")
                        )
                        tc_args = (
                            tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                        )
                        self.write(Text(f"> {tc_name}({tc_args})", style="dim"))
                elif content:
                    self.write(Text(content, style="cyan"))
            elif msg_type == "tool":
                self.write(Text(f"  [{name}]: returned", style="yellow"))
            elif msg_type == "system":
                self.write(Text("[system prompt]", style="dim red"))
