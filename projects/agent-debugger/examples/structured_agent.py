"""Example agent that returns structured (dict-shaped) AI message content.

Demonstrates adb's default text extraction. The agent's AIMessage content is a
dict like {"text": "...", "recommendations": [...]} instead of a plain string,
which is a common pattern when an agent returns both display text and
metadata (e.g. product recommendations, citations, action buttons).

By default, adb extracts the "text" field and shows just that in the chat pane.
The full structured payload is still visible in the State and Messages panels.

Run with:
    uv run adb run examples/structured_agent.py

Try `--raw-chat` to see the difference:
    uv run adb run examples/structured_agent.py --raw-chat
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

GIFTS = [
    {"id": "g1", "title": "Hand-thrown ceramic mug", "price": 28},
    {"id": "g2", "title": "Vintage wool throw blanket", "price": 64},
    {"id": "g3", "title": "Beeswax candle set", "price": 19},
]


def recommender(state: dict[str, Any]) -> dict[str, Any]:
    """Produce an AIMessage whose content is a dict with text + recommendations."""
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    user_text = ""
    if isinstance(last, dict):
        user_text = str(last.get("content", "")).lower()
    elif last is not None:
        user_text = str(getattr(last, "content", "")).lower()

    if "expensive" in user_text or "premium" in user_text:
        picks = [g for g in GIFTS if g["price"] >= 50]
        text = "Here are some premium picks for you:"
    elif "cheap" in user_text or "budget" in user_text:
        picks = [g for g in GIFTS if g["price"] < 30]
        text = "Here are some budget-friendly picks:"
    else:
        picks = GIFTS
        text = "Here are a few gift ideas:"

    structured_content = {
        "text": text,
        "recommendations": picks,
        "metadata": {"agent": "structured_agent", "version": 1},
    }

    return {"messages": [AIMessage(content=structured_content)]}


builder = StateGraph(dict)
builder.add_node("recommender", recommender)
builder.add_edge(START, "recommender")
builder.add_edge("recommender", END)
graph = builder.compile()
