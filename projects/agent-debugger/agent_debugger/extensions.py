"""Optional extension interfaces for adb rendering and state mutations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class MemoryRenderModel:
    """Structured output for memory panel rendering."""

    lines: list[str] = field(default_factory=list)


@dataclass
class ChatRenderModel:
    """Structured output for chat response rendering."""

    lines: list[str] = field(default_factory=list)


@dataclass
class StateRenderModel:
    """Structured output for state panel rendering."""

    lines: list[str] = field(default_factory=list)


@dataclass
class ToolRenderModel:
    """Structured output for tools panel rendering."""

    lines: list[str] = field(default_factory=list)


@dataclass
class StateMutationResult:
    """Outcome returned by a state mutator."""

    applied: bool = False
    message: str | None = None


class StoreRenderer(Protocol):
    """Optional renderer for backend store display."""

    def render_store(self, snapshot: Mapping[str, Any]) -> MemoryRenderModel | None:
        """Render backend store content from a generic snapshot."""


class StateRenderer(Protocol):
    """Optional renderer for state panel display."""

    def render_state(self, snapshot: Mapping[str, Any]) -> StateRenderModel | None:
        """Render state content from a generic snapshot."""


class ChatOutputRenderer(Protocol):
    """Optional renderer for custom chat output blocks."""

    def can_render(self, payload: Mapping[str, Any]) -> bool:
        """Return True when this renderer should handle the payload."""

    def render_chat_output(
        self,
        payload: Mapping[str, Any],
        state: Mapping[str, Any],
        messages: Sequence[Any],
    ) -> ChatRenderModel | None:
        """Render chat blocks for a response payload."""


class ToolRenderer(Protocol):
    """Optional renderer for tools panel display."""

    def render_tools(self, snapshot: Mapping[str, Any]) -> ToolRenderModel | None:
        """Render tool call history from a generic snapshot."""


class StateMutator(Protocol):
    """Optional mutator for generic state mutation commands."""

    def mutate_state(
        self,
        mutation: str,
        args: Sequence[str],
        current_state: Mapping[str, Any],
        runner: Any,
    ) -> StateMutationResult | None:
        """Apply a named mutation against the graph state."""


class InputProvider(Protocol):
    """Optional provider for customizing graph input state."""

    def build_input(self, message: str) -> dict[str, Any]:
        """Build the input dict for graph.stream().

        Args:
            message: The user's chat message text.

        Returns:
            A dict to pass as input_data to graph.stream().
            Must include a "messages" key.
        """


# Backward-compatibility alias.
StateMutationProvider = StateMutator
