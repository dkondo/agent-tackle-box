"""Tests for debug events."""

import sys

from agent_debugger.events import (
    AgentErrorEvent,
    AgentResponseEvent,
    BreakpointHit,
    DebugCommand,
    NodeEndEvent,
    NodeStartEvent,
    RunFinishedEvent,
    StateUpdateEvent,
    ToolCallEvent,
    ToolResultEvent,
)


class TestDebugCommand:
    """Tests for DebugCommand enum."""

    def test_all_commands_exist(self):
        assert DebugCommand.CONTINUE
        assert DebugCommand.STEP_OVER
        assert DebugCommand.STEP_INTO
        assert DebugCommand.STEP_OUT
        assert DebugCommand.QUIT


class TestEvents:
    """Tests for debug event data classes."""

    def test_node_start_event(self):
        event = NodeStartEvent(node="search", step=1)
        assert event.node == "search"
        assert event.step == 1
        assert event.triggers == []

    def test_node_end_event(self):
        event = NodeEndEvent(node="search", step=1, error="boom")
        assert event.error == "boom"

    def test_tool_call_event(self):
        event = ToolCallEvent(
            name="search_listings",
            args={"query": "gifts"},
            tool_call_id="tc_123",
            node="agent",
            step=2,
        )
        assert event.name == "search_listings"
        assert event.node == "agent"

    def test_tool_result_event(self):
        event = ToolResultEvent(
            tool_call_id="tc_123",
            result="5 results",
            duration_ms=150.0,
        )
        assert event.duration_ms == 150.0

    def test_state_update_event(self):
        event = StateUpdateEvent(
            values={"messages": [1, 2]},
            step=3,
            next_nodes=["tools"],
            checkpoint_id="cp-1",
            checkpoint_config={"configurable": {"checkpoint_id": "cp-1"}},
            checkpoint_step=8,
        )
        assert event.next_nodes == ["tools"]
        assert event.store_source == "none"
        assert event.store_error is None
        assert event.checkpoint_id == "cp-1"
        assert event.checkpoint_step == 8

    def test_breakpoint_hit_fills_from_frame(self):
        # Get a real frame
        frame = sys._getframe()
        event = BreakpointHit(frame=frame)
        assert "test_events.py" in event.filename
        assert event.lineno > 0

    def test_agent_response_event(self):
        event = AgentResponseEvent(text="Hello!", payload={"recommendations": [1]})
        assert event.text == "Hello!"
        assert event.payload["recommendations"] == [1]

    def test_agent_error_event(self):
        event = AgentErrorEvent(message="timeout")
        assert event.message == "timeout"

    def test_run_finished_event(self):
        event = RunFinishedEvent()
        assert event is not None
