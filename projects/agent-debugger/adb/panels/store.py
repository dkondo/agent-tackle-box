"""Store panel: generic key-value browser for LangGraph BaseStore."""

from __future__ import annotations

import json
from typing import Any

from rich.text import Text
from textual.widgets import Static


class StorePanel(Static):
    """Generic Store browser -- works with any agent's Store."""

    can_focus = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._items: dict[str, dict[str, Any]] = {}
        self._custom_lines: list[str] | None = None

    @property
    def items(self) -> dict[str, dict[str, Any]]:
        """Return the current generic store items."""
        return self._items

    @property
    def custom_lines(self) -> list[str] | None:
        """Return custom rendered lines when a renderer is active."""
        return self._custom_lines

    def update_store(self, items: dict[str, dict[str, Any]]) -> None:
        """Update the store display with namespace -> {key: value} items."""
        self._items = items
        self._custom_lines = None
        self._refresh_display()

    def update_custom_lines(self, lines: list[str]) -> None:
        """Update panel with pre-rendered lines from a custom renderer."""
        self._custom_lines = lines
        self._refresh_display()

    def _format_value(self, value: Any, max_len: int = 60) -> str:
        """Format a value for compact display."""
        if isinstance(value, dict):
            try:
                s = json.dumps(value, default=str)
            except (TypeError, ValueError):
                s = str(value)
        else:
            s = str(value)
        if len(s) > max_len:
            s = s[: max_len - 3] + "..."
        return s

    def _refresh_display(self) -> None:
        """Refresh the store display."""
        lines: list[str] = []

        if self._custom_lines is not None:
            if not self._custom_lines:
                lines.append("[dim]No memory yet.[/dim]")
            else:
                lines.extend(self._custom_lines)
            self.update(Text.from_markup("\n".join(lines)))
            return

        if not self._items:
            lines.append("[dim]No store data.[/dim]")
            self.update(Text.from_markup("\n".join(lines)))
            return

        for namespace, entries in sorted(self._items.items()):
            lines.append(f"[bold cyan]{namespace}/[/bold cyan]")
            if isinstance(entries, dict):
                for key, value in entries.items():
                    display = self._format_value(value)
                    lines.append(f"  {key}: {display}")
            else:
                lines.append(f"  {self._format_value(entries)}")

        self.update(Text.from_markup("\n".join(lines)))
