"""Debug event and command protocol types for thread communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from types import FrameType
from typing import Any

# ---------------------------------------------------------------------------
# Commands: UI -> worker thread
# ---------------------------------------------------------------------------


class DebugCommand(Enum):
    """Commands sent from the UI to the debugger worker thread."""

    CONTINUE = auto()
    STEP_OVER = auto()
    STEP_INTO = auto()
    STEP_OUT = auto()
    QUIT = auto()


# ---------------------------------------------------------------------------
# Events: worker thread -> UI
# ---------------------------------------------------------------------------


@dataclass
class NodeStartEvent:
    """A graph node is about to execute."""

    node: str
    step: int
    input: Any = None
    triggers: list[str] = field(default_factory=list)


@dataclass
class NodeEndEvent:
    """A graph node finished executing."""

    node: str
    step: int
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ToolCallEvent:
    """A tool call was made."""

    name: str
    args: dict[str, Any]
    tool_call_id: str
    node: str | None = None
    step: int | None = None


@dataclass
class ToolResultEvent:
    """A tool call returned a result."""

    tool_call_id: str
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class StateUpdateEvent:
    """Graph state was updated (checkpoint)."""

    values: dict[str, Any] = field(default_factory=dict)
    store_items: dict[str, dict[str, Any]] = field(default_factory=dict)
    store_source: str = "none"
    store_error: str | None = None
    step: int = 0
    next_nodes: list[str] = field(default_factory=list)
    checkpoint_id: str | None = None
    checkpoint_config: dict[str, Any] | None = None
    checkpoint_step: int | None = None


@dataclass
class BreakpointHit:
    """Execution paused at a breakpoint (Python or agent-level)."""

    frame: FrameType
    filename: str = ""
    lineno: int = 0
    node: str | None = None
    graph_state: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Fill filename/lineno from frame if not provided."""
        if self.frame and not self.filename:
            self.filename = self.frame.f_code.co_filename
        if self.frame and not self.lineno:
            self.lineno = self.frame.f_lineno


@dataclass
class AgentResponseEvent:
    """The agent produced a final response for this turn."""

    text: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentErrorEvent:
    """An error occurred during agent execution."""

    message: str = ""


@dataclass
class StreamTokenEvent:
    """An LLM token was streamed."""

    token: str = ""
    node: str | None = None


@dataclass
class RunFinishedEvent:
    """The agent run completed."""

    pass


# Union type for all events
DebugEvent = (
    NodeStartEvent
    | NodeEndEvent
    | ToolCallEvent
    | ToolResultEvent
    | StateUpdateEvent
    | BreakpointHit
    | AgentResponseEvent
    | AgentErrorEvent
    | StreamTokenEvent
    | RunFinishedEvent
)
