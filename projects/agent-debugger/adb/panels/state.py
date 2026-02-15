"""State panel: shows current graph state with expandable messages."""

from __future__ import annotations

import json
from typing import Any

from rich.text import Text
from textual.widgets import Static


class StatePanel(Static):
    """Panel showing current graph state with expandable messages."""

    can_focus = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state: dict[str, Any] = {}
        self._messages_expanded: bool = False
        self._current_node: str | None = None
        self._custom_lines: list[str] | None = None

    def update_state(self, state: dict[str, Any]) -> None:
        """Update the state display."""
        self._state = state
        self._custom_lines = None
        self._refresh_display()

    def update_custom_lines(
        self, lines: list[str], *, state: dict[str, Any] | None = None
    ) -> None:
        """Update the state display with pre-rendered lines."""
        if state is not None:
            self._state = state
        self._custom_lines = lines
        self._refresh_display()

    def set_current_node(self, node: str | None) -> None:
        """Set the currently executing node."""
        self._current_node = node
        self._refresh_display()

    def on_click(self, event: Any) -> None:
        """Toggle messages expansion on click."""
        self._messages_expanded = not self._messages_expanded
        self._refresh_display()

    def _format_content(self, content: Any, max_len: int = 50) -> str:
        """Format message content for display."""
        if isinstance(content, str) and content.startswith("{"):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "text" in parsed:
                    content = parsed["text"]
            except (json.JSONDecodeError, ValueError):
                pass
        if isinstance(content, dict) and "text" in content:
            content = content["text"]
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
            content = " ".join(parts) if parts else str(content)
        content = str(content)
        if len(content) > max_len:
            return f'"{content[:max_len]}..."'
        return f'"{content}"'

    def _refresh_display(self) -> None:
        """Refresh the panel display."""
        lines: list[str] = []

        if self._current_node:
            lines.append(
                f"[bold magenta]node:[/bold magenta] {self._current_node}"
            )

        if self._custom_lines is not None:
            if self._custom_lines:
                lines.extend(self._custom_lines)
            else:
                lines.append("[dim]No state yet.[/dim]")
            self.update(Text.from_markup("\n".join(lines)))
            return

        values = self._state

        # Show messages
        if "messages" in values:
            messages = values.get("messages", [])
            count = len(messages)
            indicator = "▼" if self._messages_expanded else "▶"
            lines.append(
                f"[cyan]{indicator} messages:[/cyan] [{count}]"
                " [dim](click to toggle)[/dim]"
            )
            if self._messages_expanded and messages:
                for i, msg in reversed(list(enumerate(messages))):
                    if isinstance(msg, dict):
                        msg_type = (
                            msg.get("type")
                            or msg.get("role")
                            or "unknown"
                        )
                        content = msg.get("content", "")
                        tool_calls = msg.get("tool_calls", [])
                    else:
                        msg_type = getattr(msg, "type", "unknown")
                        content = getattr(msg, "content", "")
                        tool_calls = getattr(msg, "tool_calls", [])

                    if msg_type == "human":
                        display = self._format_content(content)
                        lines.append(
                            f"  [green]{i}. user:[/green] {display}"
                        )
                    elif msg_type == "ai":
                        if tool_calls:
                            names = ", ".join(
                                tc.get("name", "?")
                                if isinstance(tc, dict)
                                else getattr(tc, "name", "?")
                                for tc in tool_calls
                            )
                            lines.append(
                                f"  [cyan]{i}. ai:[/cyan]"
                                f" [dim]tools: {names}[/dim]"
                            )
                        elif content:
                            display = self._format_content(content)
                            lines.append(
                                f"  [cyan]{i}. ai:[/cyan] {display}"
                            )
                        else:
                            lines.append(
                                f"  [cyan]{i}. ai:[/cyan]"
                                " [dim](empty)[/dim]"
                            )
                    elif msg_type == "tool":
                        name = (
                            msg.get("name", "tool")
                            if isinstance(msg, dict)
                            else getattr(msg, "name", "tool")
                        )
                        lines.append(
                            f"  [yellow]{i}. tool:[/yellow] {name}"
                        )
                    elif msg_type == "system":
                        lines.append(f"  [red]{i}. system[/red]")
                    else:
                        lines.append(f"  [dim]{i}. {msg_type}[/dim]")

        # Show other state keys (excluding messages)
        for key, value in values.items():
            if key == "messages":
                continue
            display = str(value)
            if len(display) > 60:
                display = display[:57] + "..."
            lines.append(f"[cyan]{key}:[/cyan] {display}")

        if not lines:
            lines.append("[dim]No state yet.[/dim]")

        self.update(Text.from_markup("\n".join(lines)))
