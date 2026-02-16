"""Logs panel: real-time debug logs."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rich.text import Text
from textual.widgets import RichLog

_LEVEL_STYLES = {
    "info": "blue",
    "debug": "dim",
    "warning": "yellow",
    "error": "red",
}


class LogsPanel(RichLog):
    """Panel showing real-time debug logs."""

    def log(self, message: str, level: str = "info") -> None:
        """Add a log entry."""
        ts = datetime.now().strftime("%H:%M:%S")
        style = _LEVEL_STYLES.get(level, "white")
        self.write(
            Text(f"[{ts}] [{level.upper()}] {message}", style=style)
        )
