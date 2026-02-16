"""Unified breakpoint manager supporting Python and agent-level breakpoints."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class BreakpointType(Enum):
    """Types of breakpoints adb supports."""

    LINE = auto()  # Standard Python: file:line
    NODE = auto()  # Agent-level: break when graph node starts
    TOOL = auto()  # Agent-level: break when tool is called
    STATE = auto()  # Agent-level: break when state key changes
    TRANSITION = auto()  # Agent-level: break on every node transition


@dataclass
class Breakpoint:
    """A single breakpoint definition."""

    type: BreakpointType
    # For LINE: filename
    filename: str | None = None
    # For LINE: line number
    lineno: int | None = None
    # For NODE/TOOL: name of node or tool
    name: str | None = None
    # For STATE: state key to watch
    key: str | None = None
    # Whether this breakpoint is active
    enabled: bool = True
    # Hit count
    hits: int = 0

    def __str__(self) -> str:
        """Human-readable representation."""
        if self.type == BreakpointType.LINE:
            return f"line {self.filename}:{self.lineno}"
        elif self.type == BreakpointType.NODE:
            return f"node {self.name}"
        elif self.type == BreakpointType.TOOL:
            return f"tool {self.name}"
        elif self.type == BreakpointType.STATE:
            return f"state {self.key}"
        elif self.type == BreakpointType.TRANSITION:
            return "transition"
        return "unknown"


class BreakpointManager:
    """Manages all breakpoints for the debugger.

    Provides methods to add, remove, enable/disable, and query
    breakpoints of all types (Python line, node, tool, state, transition).
    """

    def __init__(self) -> None:
        self._breakpoints: list[Breakpoint] = []

    @property
    def breakpoints(self) -> list[Breakpoint]:
        """All registered breakpoints."""
        return self._breakpoints.copy()

    def add_node(self, name: str) -> Breakpoint:
        """Add a node breakpoint."""
        bp = Breakpoint(type=BreakpointType.NODE, name=name)
        self._breakpoints.append(bp)
        return bp

    def add_tool(self, name: str) -> Breakpoint:
        """Add a tool breakpoint."""
        bp = Breakpoint(type=BreakpointType.TOOL, name=name)
        self._breakpoints.append(bp)
        return bp

    def add_state(self, key: str) -> Breakpoint:
        """Add a state-change breakpoint."""
        bp = Breakpoint(type=BreakpointType.STATE, key=key)
        self._breakpoints.append(bp)
        return bp

    def add_transition(self) -> Breakpoint:
        """Add a transition breakpoint (breaks on every node transition)."""
        bp = Breakpoint(type=BreakpointType.TRANSITION)
        self._breakpoints.append(bp)
        return bp

    def add_line(self, filename: str, lineno: int) -> Breakpoint:
        """Add a Python line breakpoint."""
        bp = Breakpoint(type=BreakpointType.LINE, filename=filename, lineno=lineno)
        self._breakpoints.append(bp)
        return bp

    def remove(self, index: int) -> Breakpoint | None:
        """Remove a breakpoint by index. Returns the removed breakpoint."""
        if 0 <= index < len(self._breakpoints):
            return self._breakpoints.pop(index)
        return None

    def toggle(self, index: int) -> bool:
        """Toggle a breakpoint enabled/disabled. Returns new state."""
        if 0 <= index < len(self._breakpoints):
            bp = self._breakpoints[index]
            bp.enabled = not bp.enabled
            return bp.enabled
        return False

    def clear(self) -> int:
        """Remove all breakpoints. Returns count removed."""
        count = len(self._breakpoints)
        self._breakpoints.clear()
        return count

    def should_break_on_node(self, node_name: str) -> bool:
        """Check if we should break when a node starts."""
        for bp in self._breakpoints:
            if not bp.enabled:
                continue
            if bp.type == BreakpointType.NODE and bp.name == node_name:
                bp.hits += 1
                return True
            if bp.type == BreakpointType.TRANSITION:
                bp.hits += 1
                return True
        return False

    def should_break_on_tool(self, tool_name: str) -> bool:
        """Check if we should break when a tool is called."""
        for bp in self._breakpoints:
            if not bp.enabled:
                continue
            if bp.type == BreakpointType.TOOL and bp.name == tool_name:
                bp.hits += 1
                return True
        return False

    def should_break_on_state(self, old_state: dict[str, Any], new_state: dict[str, Any]) -> bool:
        """Check if a watched state key changed."""
        for bp in self._breakpoints:
            if not bp.enabled:
                continue
            if bp.type == BreakpointType.STATE and bp.key:
                old_val = old_state.get(bp.key)
                new_val = new_state.get(bp.key)
                if old_val != new_val:
                    bp.hits += 1
                    return True
        return False

    @property
    def node_names(self) -> set[str]:
        """Set of active node breakpoint names."""
        return {
            bp.name
            for bp in self._breakpoints
            if bp.enabled and bp.type == BreakpointType.NODE and bp.name
        }

    @property
    def tool_names(self) -> set[str]:
        """Set of active tool breakpoint names."""
        return {
            bp.name
            for bp in self._breakpoints
            if bp.enabled and bp.type == BreakpointType.TOOL and bp.name
        }

    @property
    def has_transition_break(self) -> bool:
        """Whether any transition breakpoint is active."""
        return any(bp.enabled and bp.type == BreakpointType.TRANSITION for bp in self._breakpoints)

    @property
    def line_breakpoints(self) -> list[Breakpoint]:
        """Active Python line breakpoints."""
        return [
            bp
            for bp in self._breakpoints
            if bp.enabled
            and bp.type == BreakpointType.LINE
            and bp.filename
            and bp.lineno is not None
        ]

    @property
    def has_line_breakpoints(self) -> bool:
        """Whether any Python line breakpoints are active."""
        return any(
            bp.enabled and bp.type == BreakpointType.LINE and bp.filename and bp.lineno is not None
            for bp in self._breakpoints
        )
