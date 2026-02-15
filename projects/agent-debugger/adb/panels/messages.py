"""Messages panel: full message history."""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.widgets import RichLog


class MessagesPanel(RichLog):
    """Panel showing full message history."""

    def update_messages(self, messages: list[Any]) -> None:
        """Update the messages display."""
        self.clear()
        user_count = 0

        for msg in messages:
            if isinstance(msg, dict):
                msg_type = (
                    msg.get("type") or msg.get("role") or "unknown"
                )
                raw_content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])
                name = msg.get("name", "tool")
            else:
                msg_type = getattr(msg, "type", "unknown")
                raw_content = getattr(msg, "content", "")
                tool_calls = getattr(msg, "tool_calls", [])
                name = getattr(msg, "name", "tool")

            # Normalize content
            if isinstance(raw_content, list):
                content = " ".join(
                    str(c.get("text", c)) if isinstance(c, dict) else str(c)
                    for c in raw_content
                )
            else:
                content = str(raw_content) if raw_content else ""

            if msg_type == "human":
                user_count += 1
                preview = (
                    content if len(content) <= 40
                    else f"{content[:37]}..."
                )
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
                            tc.get("args", {})
                            if isinstance(tc, dict)
                            else getattr(tc, "args", {})
                        )
                        self.write(
                            Text(f"> {tc_name}({tc_args})", style="dim")
                        )
                elif content:
                    self.write(Text(content, style="cyan"))
            elif msg_type == "tool":
                self.write(
                    Text(f"  [{name}]: returned", style="yellow")
                )
            elif msg_type == "system":
                self.write(Text("[system prompt]", style="dim red"))
