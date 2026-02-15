"""Diff panel: shows state changes between turns."""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.widgets import RichLog


class DiffPanel(RichLog):
    """Panel showing state diff between turns."""

    def update_diff(
        self,
        previous: dict[str, Any] | None,
        current: dict[str, Any] | None,
    ) -> None:
        """Show diff between previous and current state."""
        self.clear()

        if previous is None or current is None:
            self.write(Text("No previous state to compare.", style="dim"))
            return

        try:
            from deepdiff import DeepDiff

            diff = DeepDiff(
                previous,
                current,
                ignore_order=True,
                verbose_level=2,
            )
        except ImportError:
            self.write(
                Text(
                    "Install deepdiff for state diffs: pip install deepdiff",
                    style="yellow",
                )
            )
            return

        if not diff:
            self.write(Text("No changes.", style="dim"))
            return

        # Format additions
        added = diff.get("dictionary_item_added", {})
        if added:
            self.write(Text("Added:", style="bold green"))
            for path, value in (
                added.items() if isinstance(added, dict) else []
            ):
                val_str = str(value)[:80]
                self.write(Text(f"  + {path}: {val_str}", style="green"))

        # Format removals
        removed = diff.get("dictionary_item_removed", {})
        if removed:
            self.write(Text("Removed:", style="bold red"))
            for path, value in (
                removed.items() if isinstance(removed, dict) else []
            ):
                val_str = str(value)[:80]
                self.write(Text(f"  - {path}: {val_str}", style="red"))

        # Format changes
        changed = diff.get("values_changed", {})
        if changed:
            self.write(Text("Changed:", style="bold yellow"))
            for path, change in (
                changed.items() if isinstance(changed, dict) else []
            ):
                if isinstance(change, dict):
                    old = str(change.get("old_value", ""))[:40]
                    new = str(change.get("new_value", ""))[:40]
                else:
                    old = "?"
                    new = str(change)[:40]
                self.write(
                    Text(f"  ~ {path}: {old} → {new}", style="yellow")
                )

        # Format list changes
        for key in ("iterable_item_added", "iterable_item_removed"):
            items = diff.get(key, {})
            if items:
                label = "List added" if "added" in key else "List removed"
                color = "green" if "added" in key else "red"
                self.write(Text(f"{label}:", style=f"bold {color}"))
                for path, value in (
                    items.items() if isinstance(items, dict) else []
                ):
                    val_str = str(value)[:80]
                    marker = "+" if "added" in key else "-"
                    self.write(
                        Text(
                            f"  {marker} {path}: {val_str}",
                            style=color,
                        )
                    )
