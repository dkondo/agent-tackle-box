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

Two modes:
- Default (USE_LITELLM unset): deterministic keyword-routed recommender. Useful
  for demos with no API keys.
- LLM mode (USE_LITELLM=1): routes through LiteLLM. The model sees the user
  message and returns a structured `{text, recommendations, metadata}` dict.
  Set `LITELLM_MODEL` (default: openai/gpt-4o-mini, which works against any
  OpenAI-compatible proxy via OPENAI_API_KEY/OPENAI_BASE_URL).

Run with:
    uv run adb run examples/structured_agent.py

LLM mode (with a LiteLLM proxy):
    USE_LITELLM=1 uv run adb run examples/structured_agent.py

Try `--raw-chat` to see the difference in the chat pane:
    uv run adb run examples/structured_agent.py --raw-chat
"""

from __future__ import annotations

import os
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from langgraph.graph import END, START, StateGraph

if load_dotenv is not None:
    load_dotenv()

USE_LITELLM_ENV = "USE_LITELLM"
DEFAULT_LITELLM_MODEL = "openai/gpt-4o-mini"

WELCOME = (
    "Demo: structured-content recommender.\n"
    "Try: 'give me ideas', 'cheap ones', 'premium picks'.\n"
    "Each AI message content is a dict {'text': ..., 'recommendations': [...]}.\n"
    "By default adb extracts and shows just the text. Use --raw-chat to see "
    "the full structured payload in the chat pane.\n"
    "Set USE_LITELLM=1 to route through LiteLLM (uses OPENAI_API_KEY / "
    "OPENAI_BASE_URL with model openai/gpt-4o-mini by default)."
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


def _format_text(lead: str, picks: list[dict[str, Any]]) -> str:
    """Format the human-readable text including the picks themselves."""
    lines = [lead]
    for i, pick in enumerate(picks, start=1):
        lines.append(f"  {i}. {pick['title']} (${pick['price']})")
    return "\n".join(lines)


def _litellm_enabled() -> bool:
    return os.getenv(USE_LITELLM_ENV, "").lower() in {"1", "true", "yes", "on"}


def _recommend_via_litellm(user_text: str) -> dict[str, Any]:
    """Route through LiteLLM and ask the model to pick gifts.

    The model is given the catalog and the user message, and asked to return
    a structured response with a one-line lead and a list of recommendation
    IDs. We then look up the IDs in the catalog so the response always
    references real items (the model's freedom is in selection, not invention).
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_litellm import ChatLiteLLM
    from pydantic import BaseModel, Field

    class _Recommendation(BaseModel):
        ids: list[str] = Field(description="List of gift IDs from the catalog.")
        lead: str = Field(description="One-line lead sentence introducing the picks.")

    catalog_lines = "\n".join(f"- {g['id']}: {g['title']} (${g['price']})" for g in GIFTS)
    system = (
        "You are a gift recommender. Pick 1-3 gifts from this catalog that "
        "best match the user's request, and return their IDs plus a short "
        "lead sentence. Only use IDs that exist in the catalog.\n\n"
        f"Catalog:\n{catalog_lines}"
    )

    model_name = os.getenv("LITELLM_MODEL", DEFAULT_LITELLM_MODEL)
    llm = ChatLiteLLM(model=model_name, temperature=0)
    structured = llm.with_structured_output(_Recommendation)
    result = structured.invoke([SystemMessage(content=system), HumanMessage(content=user_text)])

    # Map IDs back to catalog entries; ignore any IDs the model invented.
    by_id = {g["id"]: g for g in GIFTS}
    picks = [by_id[i] for i in result.ids if i in by_id]
    if not picks:
        picks = GIFTS  # graceful fallback if model returned no valid IDs
    return {
        "text": _format_text(result.lead, picks),
        "recommendations": picks,
        "metadata": {"agent": "structured_agent", "version": 1, "mode": "litellm"},
    }


def _recommend_deterministic(user_text: str) -> dict[str, Any]:
    """Keyword-routed offline recommender (no API keys required)."""
    if "expensive" in user_text or "premium" in user_text:
        picks = [g for g in GIFTS if g["price"] >= 50]
        lead = "Here are some premium picks for you:"
    elif "cheap" in user_text or "budget" in user_text:
        picks = [g for g in GIFTS if g["price"] < 30]
        lead = "Here are some budget-friendly picks:"
    else:
        picks = GIFTS
        lead = "Here are a few gift ideas:"
    return {
        "text": _format_text(lead, picks),
        "recommendations": picks,
        "metadata": {"agent": "structured_agent", "version": 1, "mode": "deterministic"},
    }


def recommender(state: dict[str, Any]) -> dict[str, Any]:
    """Produce a dict-style AI message whose content is itself a dict."""
    user_text = _user_text(state)
    if _litellm_enabled():
        try:
            structured_content = _recommend_via_litellm(user_text)
        except Exception as e:
            # Fall back to deterministic on any LLM error so the demo never
            # hard-crashes the debugger session.
            fallback = _recommend_deterministic(user_text)
            fallback["text"] = f"[LiteLLM error: {e}]\n{fallback['text']}"
            fallback["metadata"]["mode"] = "deterministic-fallback"
            structured_content = fallback
    else:
        structured_content = _recommend_deterministic(user_text)

    # Use a plain dict instead of AIMessage. Pydantic-typed AIMessage rejects
    # dict content; plain-dict messages flow through LangGraph state untouched.
    return {"messages": [{"role": "ai", "content": structured_content}]}


builder = StateGraph(dict)
builder.add_node("recommender", recommender)
builder.add_edge(START, "recommender")
builder.add_edge("recommender", END)
graph = builder.compile()
