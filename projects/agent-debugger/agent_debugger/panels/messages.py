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
        width with no way to see the rest. Wrap forces the content to flow
        onto multiple lines so the full payload is visible.
        """
        kwargs.setdefault("wrap", True)
        kwargs.setdefault("min_width", 1)
        super().__init__(**kwargs)

    def write(  # type: ignore[override]
        self,
        content: Any,
        width: int | None = None,
        expand: bool = False,
        shrink: bool = True,
        scroll_end: bool | None = None,
        animate: bool = False,
    ) -> MessagesPanel:
        """Write content forced to wrap at the current container width."""
        if self.wrap and width is None and self._size_known:
            content_width = self.scrollable_content_region.width
            if content_width > 0:
                width = content_width
        return super().write(  # type: ignore[return-value]
            content,
            width=width,
            expand=expand,
            shrink=shrink,
            scroll_end=scroll_end,
            animate=animate,
        )

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
