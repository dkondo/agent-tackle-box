"""Tool calls panel: tool call history with args and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.text import Text
from textual.widgets import RichLog


@dataclass
class ToolCallRecord:
    """A completed tool call with its result."""

    name: str
    args: dict[str, Any]
    tool_call_id: str
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    node: str | None = None
    step: int | None = None


class ToolCallsPanel(RichLog):
    """Panel showing tool call history with args and results."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._records: list[ToolCallRecord] = []

    def add_tool_call(self, record: ToolCallRecord) -> None:
        """Add a tool call record."""
        self._records.append(record)
        self._refresh_display()

    def update_result(
        self,
        tool_call_id: str,
        result: Any = None,
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Update a tool call with its result."""
        for rec in self._records:
            if rec.tool_call_id == tool_call_id:
                rec.result = result
                rec.error = error
                rec.duration_ms = duration_ms
                break
        self._refresh_display()

    def _filter_empty(self, data: Any) -> Any:
        """Recursively filter empty values from dicts."""
        if isinstance(data, dict):
            filtered = {}
            for k, v in data.items():
                if v is None or v in ("", "0", 0, [], {}):
                    continue
                if isinstance(v, dict):
                    fv = self._filter_empty(v)
                    if fv:
                        filtered[k] = fv
                else:
                    filtered[k] = v
            return filtered
        return data

    def _refresh_display(self) -> None:
        """Refresh the tool calls display."""
        self.clear()
        if not self._records:
            self.write(Text("No tool calls yet.", style="dim"))
            return

        for i, tc in enumerate(self._records, 1):
            # Node/step annotation
            prefix = ""
            if tc.node:
                prefix = f"[{tc.node}] "
            self.write(Text(f"{prefix}[{i}] {tc.name}", style="bold cyan"))
            filtered_args = self._filter_empty(tc.args)
            self.write(Text(f"    args: {filtered_args}", style="dim"))
            if tc.result is not None:
                result_str = str(tc.result)[:100]
                self.write(Text(f"    result: {result_str}", style="green"))
            if tc.error:
                self.write(Text(f"    error: {tc.error}", style="red"))
            if tc.duration_ms > 0:
                self.write(
                    Text(f"    duration: {tc.duration_ms:.0f}ms", style="dim")
                )

    def clear_records(self) -> None:
        """Clear all tool call records."""
        self._records.clear()
        self.clear()
