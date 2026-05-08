"""Example agent that returns structured (dict-shaped) AI message content.

Demonstrates adb's default text extraction. The agent emits dict-style messages
(rather than typed AIMessage objects) so the AI message `content` is itself a
dict like {"text": "...", "recommendations": [...]} instead of a plain string.
This is the realistic shape when an agent wants to surface both display text
and structured metadata (e.g. product recommendations, citations, action
buttons) in the same message.

LangChain's typed AIMessage rejects dict content via Pydantic validation, so we
use plain dicts in state -- LangGraph accepts these as long as message_type
returns "ai".

By default, adb extracts the "text" field and shows just that in the chat pane.
The full structured payload is still visible in the State and Messages panels.

Run with:
    uv run adb run examples/structured_agent.py

Try `--raw-chat` to see the difference:
    uv run adb run examples/structured_agent.py --raw-chat
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

WELCOME = (
    "Demo: structured-content recommender.\n"
    "Try: 'give me ideas', 'cheap ones', 'premium picks'.\n"
    "Each AI message content is a dict {'text': ..., 'recommendations': [...]}.\n"
    "By default adb extracts and shows just the text. Use --raw-chat to see "
    "the full structured payload in the chat pane."
)

GIFTS = [
    {"id": "g1", "title": "Hand-thrown ceramic mug", "price": 28},
    {"id": "g2", "title": "Vintage wool throw blanket", "price": 64},
    {"id": "g3", "title": "Beeswax candle set", "price": 19},
]


def _user_text(state: dict[str, Any]) -> str:
    messages = state.get("messages", [])
    if not messages:
        return ""
    last = messages[-1]
    if isinstance(last, dict):
        return str(last.get("content", "")).lower()
    return str(getattr(last, "content", "")).lower()


def recommender(state: dict[str, Any]) -> dict[str, Any]:
    """Produce a dict-style AI message whose content is itself a dict."""
    user_text = _user_text(state)

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

    # Use a plain dict instead of AIMessage. Pydantic-typed AIMessage rejects
    # dict content; plain-dict messages flow through LangGraph state untouched.
    return {"messages": [{"role": "ai", "content": structured_content}]}


builder = StateGraph(dict)
builder.add_node("recommender", recommender)
builder.add_edge(START, "recommender")
builder.add_edge("recommender", END)
graph = builder.compile()
