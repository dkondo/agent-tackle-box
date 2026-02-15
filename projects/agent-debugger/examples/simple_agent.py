"""Simple test agent for adb debugging.

Run with: uv run adb run examples/simple_agent.py
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph


def greeter(state: dict) -> dict:
    """A simple node that greets the user."""
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None
    memory = state.get("memory", {})
    if not isinstance(memory, dict):
        memory = {}

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

    return {
        "messages": [{"role": "ai", "content": response}],
        "memory": memory_update,
    }


def build_graph():
    """Build a simple test graph."""
    builder = StateGraph(dict)
    builder.add_node("greeter", greeter)
    builder.add_edge(START, "greeter")
    builder.add_edge("greeter", END)
    return builder.compile()


# The graph object that adb auto-detects
graph = build_graph()
