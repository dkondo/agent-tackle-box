"""Optional extension implementations for the simple agent demo."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from adb.extensions import (
    ChatRenderModel,
    MemoryRenderModel,
    StateMutationResult,
)

MEMORY_NAMESPACE_PREFIX = "memories/simple_agent/"
MEMORY_KEY = "session"


class SimpleMemoryRenderer:
    """Render simple agent memory in a custom panel format."""

    def render_memory(
        self, snapshot: Mapping[str, Any]
    ) -> MemoryRenderModel | None:
        store_items = snapshot.get("store_items", {})
        if not isinstance(store_items, dict):
            return None
        memory: Mapping[str, Any] | None = None
        for namespace, entries in store_items.items():
            if not isinstance(namespace, str) or not isinstance(entries, dict):
                continue
            if not namespace.startswith(MEMORY_NAMESPACE_PREFIX):
                continue
            candidate = entries.get(MEMORY_KEY)
            if isinstance(candidate, dict):
                memory = candidate
                break
        if memory is None:
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


class SimpleStateMutator:
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
        store = getattr(getattr(runner, "graph", None), "store", None)
        if store is None:
            return StateMutationResult(
                applied=False,
                message="No backend store is configured.",
            )

        deleted = 0
        try:
            namespaces = store.list_namespaces(
                prefix=("memories", "simple_agent"),
                limit=1000,
                offset=0,
            )
            for raw_namespace in namespaces:
                if not isinstance(raw_namespace, (tuple, list)):
                    continue
                namespace = tuple(str(part) for part in raw_namespace)
                entries = store.search(
                    namespace,
                    limit=1000,
                    offset=0,
                    refresh_ttl=False,
                )
                for item in entries:
                    key = getattr(item, "key", None)
                    if key is None and isinstance(item, dict):
                        key = item.get("key")
                    if key is None:
                        continue
                    store.delete(namespace, str(key))
                    deleted += 1
        except Exception as e:
            return StateMutationResult(
                applied=False,
                message=f"Store clear failed: {e}",
            )

        if deleted:
            return StateMutationResult(
                applied=True,
                message=f"Cleared {deleted} store item(s).",
            )
        return StateMutationResult(
            applied=True,
            message="No simple-agent store items to clear.",
        )
