"""Simple test agent for adb debugging.

Run with: uv run adb run examples/simple_agent.py
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

MEMORY_PREFIX = ("memories", "simple_agent")
MEMORY_KEY = "session"


def _memory_namespace(runtime: Runtime[Any]) -> tuple[str, ...]:
    """Choose a stable namespace for this run's long-term memory."""
    context = runtime.context
    if isinstance(context, dict):
        user_id = context.get("user_id")
        if user_id is not None:
            return (*MEMORY_PREFIX, str(user_id))
    return (*MEMORY_PREFIX, "default")


def greeter(state: dict, runtime: Runtime[Any]) -> dict:
    """A simple node that greets the user."""
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None
    memory: dict[str, Any] = {}
    store = runtime.store
    namespace = _memory_namespace(runtime)

    if store is not None:
        existing = store.get(namespace, MEMORY_KEY, refresh_ttl=False)
        if existing is not None and isinstance(existing.value, dict):
            memory = dict(existing.value)

    if last_msg:
        if isinstance(last_msg, dict):
            content = last_msg.get("content", "")
        else:
            content = getattr(last_msg, "content", "")
        user_text = str(content)
        response = f"Hello! You said: {user_text}"
    else:
        user_text = ""
        response = "Hello! How can I help?"

    turn_count = int(memory.get("turn_count", 0)) + 1
    recent_user_messages = list(memory.get("recent_user_messages", []))
    if user_text:
        recent_user_messages.append(user_text)
    recent_user_messages = recent_user_messages[-5:]

    memory_update: dict[str, Any] = {
        "turn_count": turn_count,
        "last_user_message": user_text,
        "recent_user_messages": recent_user_messages,
    }
    if store is not None:
        store.put(namespace, MEMORY_KEY, memory_update)

    return {
        "messages": [{"role": "ai", "content": response}],
    }


def build_graph():
    """Build a simple test graph."""
    builder = StateGraph(dict)
    builder.add_node("greeter", greeter)
    builder.add_edge(START, "greeter")
    builder.add_edge("greeter", END)
    return builder.compile(store=InMemoryStore())


# The graph object that adb auto-detects
graph = build_graph()
