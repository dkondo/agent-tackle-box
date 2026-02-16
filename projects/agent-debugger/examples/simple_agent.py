"""Simple test agent for adb debugging.

Run with: uv run adb run examples/simple_agent.py

Requires:
- Install `langchain-litellm` and `python-dotenv`
- Set `USE_LITELLM=1` in .env
- Optional: set `LITELLM_MODEL` (default: `gemini/gemini-2.0-flash`)
"""

from __future__ import annotations

import os
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

if load_dotenv is not None:
    load_dotenv()

MEMORY_PREFIX = ("memories", "simple_agent")
MEMORY_KEY = "session"
USE_LITELLM_ENV = "USE_LITELLM"
DEFAULT_LITELLM_MODEL = "gemini/gemini-2.0-flash"

SYSTEM_PROMPT = """\
You are a friendly greeting assistant. You MUST use your tools to generate greetings. \
Never write greetings yourself -- always call a tool instead.

Available tools:
- generate_greeting: Use this for any greeting request. Accepts a name and style \
(friendly, formal, or enthusiastic).
- time_based_greeting: Use this when the user mentions a time of day \
(morning, afternoon, evening) or a language (english, spanish, french).

When the user says hello, hi, or any greeting, call generate_greeting. \
When they mention a time of day, call time_based_greeting. \
After receiving the tool result, relay it to the user naturally.\
"""


@tool
def generate_greeting(
    name: str = "friend",
    style: str = "friendly",
    tool_call_id: str = "",
) -> str:
    """Mock greeting generation for a given name and style."""
    _ = tool_call_id
    clean_name = name.strip() or "friend"
    normalized_style = style.strip().lower() or "friendly"
    templates = {
        "friendly": f"Hey {clean_name}! Great to see you.",
        "formal": f"Hello {clean_name}. It is a pleasure to meet you.",
        "enthusiastic": f"Hi {clean_name}!!! So happy you are here!",
    }
    return templates.get(normalized_style, templates["friendly"])


@tool
def time_based_greeting(
    time_of_day: str = "day",
    language: str = "english",
    tool_call_id: str = "",
) -> str:
    """Mock a time-based greeting in a few languages."""
    _ = tool_call_id
    tod = time_of_day.strip().lower() or "day"
    lang = language.strip().lower() or "english"
    phrases = {
        ("morning", "english"): "Good morning!",
        ("afternoon", "english"): "Good afternoon!",
        ("evening", "english"): "Good evening!",
        ("morning", "spanish"): "Buenos dias!",
        ("afternoon", "spanish"): "Buenas tardes!",
        ("evening", "spanish"): "Buenas noches!",
        ("morning", "french"): "Bonjour!",
        ("evening", "french"): "Bonsoir!",
    }
    return phrases.get((tod, lang), "Hello!")


TOOLS = [generate_greeting, time_based_greeting]
TOOL_BY_NAME = {tool.name: tool for tool in TOOLS}


def _memory_namespace(runtime: Runtime[Any]) -> tuple[str, ...]:
    """Choose a stable namespace for this run's long-term memory."""
    context = runtime.context
    if isinstance(context, dict):
        user_id = context.get("user_id")
        if user_id is not None:
            return (*MEMORY_PREFIX, str(user_id))
    return (*MEMORY_PREFIX, "default")


def _content_to_text(content: Any) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                parts.append(str(text) if text is not None else str(item))
            else:
                parts.append(str(item))
        return " ".join(parts).strip()
    return str(content).strip()


def _message_type(msg: Any) -> str:
    if isinstance(msg, dict):
        return str(msg.get("type") or msg.get("role") or "")
    return str(getattr(msg, "type", "") or getattr(msg, "role", ""))


def _message_text(msg: Any) -> str:
    if isinstance(msg, dict):
        return _content_to_text(msg.get("content", ""))
    return _content_to_text(getattr(msg, "content", ""))


def _to_model_messages(messages: list[Any]) -> list[BaseMessage]:
    model_messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in messages:
        if isinstance(msg, BaseMessage):
            model_messages.append(msg)
            continue
        if not isinstance(msg, dict):
            continue
        msg_type = str(msg.get("type") or msg.get("role") or "").lower()
        content = msg.get("content", "")
        if msg_type in {"human", "user"}:
            model_messages.append(HumanMessage(content=content))
        elif msg_type in {"tool"}:
            model_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=str(msg.get("tool_call_id", "")),
                    name=str(msg.get("name", "")) or None,
                )
            )
        else:
            model_messages.append(
                AIMessage(
                    content=content,
                    tool_calls=list(msg.get("tool_calls", [])),
                )
            )
    return model_messages


def _last_user_text(messages: list[Any]) -> str:
    if not messages:
        return ""
    last_msg = messages[-1]
    if _message_type(last_msg).lower() in {"human", "user"}:
        return _message_text(last_msg)
    return ""


def _get_bound_model() -> Any:
    """Get the LLM bound with tools. Requires USE_LITELLM=1 and langchain-litellm."""
    if not os.getenv(USE_LITELLM_ENV, "").lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            f"Set {USE_LITELLM_ENV}=1 in .env to enable the LLM. "
            "No mock LLM fallback is available."
        )
    from langchain_litellm import ChatLiteLLM

    model_name = os.getenv("LITELLM_MODEL", DEFAULT_LITELLM_MODEL)
    llm = ChatLiteLLM(model=model_name, temperature=0)
    return llm.bind_tools(TOOLS)


def _agent_reply(messages: list[Any]) -> AIMessage:
    """Call the LLM and return its response."""
    bound_model = _get_bound_model()
    result = bound_model.invoke(_to_model_messages(messages))
    if isinstance(result, AIMessage):
        return result
    return AIMessage(content=_content_to_text(getattr(result, "content", "")))


def _run_tool_call(tool_call: dict[str, Any]) -> ToolMessage:
    tool_name = str(tool_call.get("name", ""))
    tool_call_id = str(tool_call.get("id", ""))
    args = tool_call.get("args", {})
    tool_obj = TOOL_BY_NAME.get(tool_name)

    if tool_obj is None:
        return ToolMessage(
            content=f"Unknown tool: {tool_name}",
            tool_call_id=tool_call_id,
            name=tool_name or "unknown_tool",
        )

    try:
        invoke_args = dict(args) if isinstance(args, dict) else {}
        invoke_args["tool_call_id"] = tool_call_id
        result = tool_obj.invoke(invoke_args)
    except Exception as exc:
        result = f"Tool execution error: {exc}"
    return ToolMessage(
        content=_content_to_text(result),
        tool_call_id=tool_call_id,
        name=tool_name,
    )


def _respond_with_optional_tool_calls(messages: list[Any]) -> list[Any]:
    assistant_message = _agent_reply(messages)
    tool_calls = list(assistant_message.tool_calls or [])
    if not tool_calls:
        return [assistant_message]

    tool_messages = [_run_tool_call(call) for call in tool_calls]
    followup_messages = [*messages, assistant_message, *tool_messages]
    final_message = _agent_reply(followup_messages)
    return [assistant_message, *tool_messages, final_message]


def greeter(state: dict, runtime: Runtime[Any]) -> dict:
    """A simple node that greets the user and may call mock tools."""
    messages = state.get("messages", [])
    memory: dict[str, Any] = {}
    store = runtime.store
    namespace = _memory_namespace(runtime)

    if store is not None:
        existing = store.get(namespace, MEMORY_KEY, refresh_ttl=False)
        if existing is not None and isinstance(existing.value, dict):
            memory = dict(existing.value)

    user_text = _last_user_text(messages)
    response_messages = _respond_with_optional_tool_calls(messages)

    turn_count = int(memory.get("turn_count", 0))
    recent_user_messages = list(memory.get("recent_user_messages", []))
    if user_text:
        turn_count += 1
        recent_user_messages.append(user_text)
    recent_user_messages = recent_user_messages[-5:]

    memory_update: dict[str, Any] = {
        "turn_count": turn_count,
        "last_user_message": user_text or memory.get("last_user_message", ""),
        "recent_user_messages": recent_user_messages,
    }
    if store is not None:
        store.put(namespace, MEMORY_KEY, memory_update)

    return {
        "messages": response_messages,
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
