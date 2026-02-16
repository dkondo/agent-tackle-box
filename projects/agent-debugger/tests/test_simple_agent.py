"""Tests for examples.simple_agent deterministic fallback behavior."""

from langchain_core.messages import AIMessage

from examples import simple_agent


def test_deterministic_path_without_litellm(monkeypatch):
    """Without USE_LITELLM, the deterministic path should emit tool calls."""
    monkeypatch.delenv(simple_agent.USE_LITELLM_ENV, raising=False)

    response_messages = simple_agent._respond_with_optional_tool_calls(
        [{"role": "user", "content": "hello"}]
    )

    # AI (tool_call) + ToolMessage + AI (final relay)
    assert len(response_messages) == 3
    assert isinstance(response_messages[0], AIMessage)
    assert len(response_messages[0].tool_calls) == 1
    assert response_messages[0].tool_calls[0]["name"] == "generate_greeting"
    assert isinstance(response_messages[2], AIMessage)
    assert list(response_messages[2].tool_calls or []) == []


def test_deterministic_time_of_day_reply(monkeypatch):
    """Fallback path should return a time_based_greeting tool call."""
    monkeypatch.delenv(simple_agent.USE_LITELLM_ENV, raising=False)

    response = simple_agent._agent_reply([{"role": "user", "content": "good morning there"}])

    assert isinstance(response, AIMessage)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "time_based_greeting"
    assert response.tool_calls[0]["args"]["time_of_day"] == "morning"


def test_graph_accumulates_messages_across_turns(monkeypatch):
    """Each new run should preserve prior turns in state.messages."""
    monkeypatch.delenv(simple_agent.USE_LITELLM_ENV, raising=False)
    graph = simple_agent.build_graph()

    def _final_values(text: str) -> dict:
        values = {}
        for mode, data in graph.stream(
            {"messages": [{"role": "human", "content": text}]},
            stream_mode=["values"],
        ):
            if mode == "values":
                values = data
        return values

    first_turn = _final_values("hello")
    first_messages = first_turn.get("messages", [])
    # human + AI(tool_call) + ToolMessage + AI(final)
    assert len(first_messages) == 4

    second_turn = _final_values("good morning")
    second_messages = second_turn.get("messages", [])
    # 4 from turn 1 + 4 from turn 2
    assert len(second_messages) == 8
    assert simple_agent._message_type(second_messages[0]) == "human"
    assert simple_agent._message_type(second_messages[4]) == "human"
    assert simple_agent._message_text(second_messages[4]) == "good morning"
