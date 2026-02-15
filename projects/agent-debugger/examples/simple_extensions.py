"""Optional extension implementations for the simple agent demo."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from adb.extensions import (
    ChatRenderModel,
    MemoryRenderModel,
    StateMutationResult,
)


class SimpleMemoryRenderer:
    """Render simple agent memory in a custom panel format."""

    def render_memory(
        self, snapshot: Mapping[str, Any]
    ) -> MemoryRenderModel | None:
        state = snapshot.get("state", {})
        if not isinstance(state, dict):
            return None
        memory = state.get("memory")
        if not isinstance(memory, dict):
            return None

        lines = [
            "[bold cyan]Simple Agent Memory[/bold cyan]",
            f"turn_count: {memory.get('turn_count', 0)}",
            f"last_user_message: {memory.get('last_user_message', '')}",
        ]
        recent = memory.get("recent_user_messages", [])
        if isinstance(recent, list) and recent:
            lines.append("recent_user_messages:")
            for item in recent:
                lines.append(f"  - {item}")

        return MemoryRenderModel(lines=lines)


class SimpleChatOutputRenderer:
    """Render recommendation payloads as card-like chat blocks."""

    def can_render(self, payload: Mapping[str, Any]) -> bool:
        recommendations = payload.get("recommendations")
        return isinstance(recommendations, list) and bool(recommendations)

    def render_chat_output(
        self,
        payload: Mapping[str, Any],
        state: Mapping[str, Any],
        messages: Sequence[Any],
    ) -> ChatRenderModel | None:
        recommendations = payload.get("recommendations")
        if not isinstance(recommendations, list) or not recommendations:
            return None

        lines = ["[bold green]Recommendations[/bold green]"]
        for idx, rec in enumerate(recommendations, start=1):
            if not isinstance(rec, dict):
                lines.append(f"{idx}. {rec}")
                continue
            title = rec.get("title", "Untitled")
            rationale = rec.get("rationale", "")
            lines.append(f"{idx}. [cyan]{title}[/cyan]")
            if rationale:
                lines.append(f"   [dim]{rationale}[/dim]")
        return ChatRenderModel(lines=lines)


class SimpleStateMutationProvider:
    """Apply simple state mutations through runner.update_graph_state."""

    def mutate_state(
        self,
        mutation: str,
        args: Sequence[str],
        current_state: Mapping[str, Any],
        runner: Any,
    ) -> StateMutationResult | None:
        if mutation not in {"memory", "all"}:
            return StateMutationResult(
                applied=False,
                message=f"Unsupported mutation '{mutation}'.",
            )

        success, detail = runner.update_graph_state({"memory": {}})
        if success:
            return StateMutationResult(
                applied=True,
                message="Persisted graph memory was cleared.",
            )
        return StateMutationResult(
            applied=False,
            message=f"Graph memory clear failed: {detail}",
        )
