"""Variables panel: shows frame locals/globals when at a breakpoint."""

from __future__ import annotations

from types import FrameType
from typing import Any

from rich.text import Text
from textual.widgets import Static

# Types to skip in variable display
_SKIP_TYPES = (type, type(lambda: None), type(print))


class VariablesPanel(Static):
    """Panel showing variables from the current frame."""

    can_focus = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._frame: FrameType | None = None

    def update_frame(self, frame: FrameType | None) -> None:
        """Update with a new frame's locals."""
        self._frame = frame
        self._refresh_display()

    def clear_frame(self) -> None:
        """Clear the frame display."""
        self._frame = None
        self.update(Text("[dim]Not at a breakpoint.[/dim]"))

    def _format_value(self, value: Any, max_len: int = 80) -> str:
        """Format a value for display."""
        try:
            s = repr(value)
        except Exception:
            s = f"<{type(value).__name__}>"
        if len(s) > max_len:
            s = s[: max_len - 3] + "..."
        return s

    def _refresh_display(self) -> None:
        """Refresh the variable display."""
        if self._frame is None:
            self.update(Text("[dim]Not at a breakpoint.[/dim]"))
            return

        lines: list[str] = []
        lines.append("[bold cyan]Locals:[/bold cyan]")

        local_vars = self._frame.f_locals
        for name, value in sorted(local_vars.items()):
            # Skip dunder, modules, and callable types
            if name.startswith("__") and name.endswith("__"):
                continue
            if isinstance(value, _SKIP_TYPES):
                continue
            display = self._format_value(value)
            lines.append(f"  {name} = {display}")

        # Show return value if present
        if "__return__" in local_vars:
            ret = self._format_value(local_vars["__return__"])
            lines.append(f"\n[bold green]Return:[/bold green] {ret}")

        # Show exception if present
        if "__exception__" in local_vars:
            exc = local_vars["__exception__"]
            lines.append(f"\n[bold red]Exception:[/bold red] {exc}")

        if len(lines) == 1:
            lines.append("  [dim](no locals)[/dim]")

        self.update(Text.from_markup("\n".join(lines)))
