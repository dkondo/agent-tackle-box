"""Stack panel: shows call stack when at a breakpoint."""

from __future__ import annotations

import os
from types import FrameType
from typing import Any

from rich.text import Text
from textual.widgets import Static


class StackPanel(Static):
    """Panel showing the call stack at a breakpoint."""

    can_focus = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._frames: list[tuple[str, int, str]] = []

    def update_frame(self, frame: FrameType | None) -> None:
        """Update the stack display from a frame."""
        self._frames = []
        if frame is None:
            self._refresh_display()
            return

        # Walk up the stack
        f: FrameType | None = frame
        while f is not None:
            filename = f.f_code.co_filename
            lineno = f.f_lineno
            func_name = f.f_code.co_name

            # Shorten filename
            short = self._shorten_path(filename)
            self._frames.append((short, lineno, func_name))
            f = f.f_back

        self._refresh_display()

    def clear_frame(self) -> None:
        """Clear the stack display."""
        self._frames = []
        self.update(Text("[dim]Not at a breakpoint.[/dim]"))

    def _shorten_path(self, path: str) -> str:
        """Shorten a file path for display."""
        # Try to make it relative to cwd
        try:
            rel = os.path.relpath(path)
            if len(rel) < len(path):
                return rel
        except ValueError:
            pass
        # Fallback: just the basename with parent
        parts = path.rsplit("/", 2)
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        return path

    def _refresh_display(self) -> None:
        """Refresh the stack display."""
        if not self._frames:
            self.update(Text("[dim]Not at a breakpoint.[/dim]"))
            return

        lines: list[str] = []
        lines.append("[bold cyan]Call Stack:[/bold cyan]")

        # Show frames with most recent first, limit depth
        max_frames = 15
        for i, (filename, lineno, func_name) in enumerate(
            self._frames[:max_frames]
        ):
            if i == 0:
                marker = "►"
                style = "bold white"
            else:
                marker = " "
                style = "dim"

            lines.append(
                f"  {marker} [{style}]{func_name}[/{style}]"
                f" ({filename}:{lineno})"
            )

        if len(self._frames) > max_frames:
            lines.append(
                f"  ... {len(self._frames) - max_frames} more frames"
            )

        self.update(Text.from_markup("\n".join(lines)))
