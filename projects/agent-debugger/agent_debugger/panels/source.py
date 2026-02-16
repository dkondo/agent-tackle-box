"""Source panel: source code view with breakpoint markers and current line."""

from __future__ import annotations

import linecache
from typing import Any

from rich.text import Text
from textual.widgets import RichLog


class SourcePanel(RichLog):
    """Panel showing source code with breakpoint markers and current line."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._filename: str | None = None
        self._current_line: int = 0
        self._breakpoint_lines: set[int] = set()
        self._context_lines: int = 10  # Lines to show above/below current

    def show_source(
        self,
        filename: str,
        lineno: int,
        breakpoint_lines: set[int] | None = None,
    ) -> None:
        """Show source code centered on the given line."""
        self._filename = filename
        self._current_line = lineno
        self._breakpoint_lines = breakpoint_lines or set()
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Refresh the source code display."""
        self.clear()

        if not self._filename:
            self.write(Text("No source file.", style="dim"))
            return

        # Invalidate linecache to get fresh content
        linecache.checkcache(self._filename)
        lines = linecache.getlines(self._filename)

        if not lines:
            self.write(Text(f"Cannot read: {self._filename}", style="red"))
            return

        # Show filename header
        self.write(Text(f"  {self._filename}", style="bold underline"))
        self.write(Text(""))

        # Calculate window
        start = max(0, self._current_line - self._context_lines - 1)
        end = min(len(lines), self._current_line + self._context_lines)

        for i in range(start, end):
            line_num = i + 1
            line_text = lines[i].rstrip()

            # Build prefix: breakpoint marker + current line indicator
            if line_num in self._breakpoint_lines:
                bp_marker = "●"
            else:
                bp_marker = " "

            if line_num == self._current_line:
                indicator = "►"
                style = "bold white on rgb(60,60,80)"
            else:
                indicator = " "
                style = ""

            prefix = f" {bp_marker}{indicator}{line_num:5d} │ "

            if style:
                self.write(Text(f"{prefix}{line_text}", style=style))
            else:
                self.write(Text(f"{prefix}{line_text}"))

    def clear_source(self) -> None:
        """Clear the source display."""
        self._filename = None
        self._current_line = 0
        self.clear()
        self.write(Text("No breakpoint active.", style="dim"))
