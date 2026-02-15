"""Traces panel: stores and displays trace URLs."""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.widgets import RichLog


class TracesPanel(RichLog):
    """Panel showing trace URLs collected during runs."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._urls: list[str] = []

    def add_trace(self, url: str) -> None:
        """Add a trace URL to the panel."""
        self._urls.append(url)
        self.write(Text(f"[{len(self._urls)}] {url}", style="blue underline"))

    def get_latest_url(self) -> str | None:
        """Return the latest trace URL if available."""
        return self._urls[-1] if self._urls else None

    def clear_urls(self) -> None:
        """Clear stored URLs and display."""
        self._urls.clear()
        self.clear()
