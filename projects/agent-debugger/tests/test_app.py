"""Tests for the adb Textual app."""

import inspect
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import pytest

from agent_debugger.app import DebuggerApp
from agent_debugger.breakpoints import BreakpointManager, BreakpointType
from agent_debugger.events import (
    AgentResponseEvent,
    BreakpointHit,
    DebugCommand,
    RunFinishedEvent,
    StateUpdateEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from agent_debugger.extensions import (
    ChatRenderModel,
    MemoryRenderModel,
    StateRenderModel,
    StateMutationResult,
    ToolRenderModel,
)
from agent_debugger.runner import AgentRunner
from agent_debugger.tracer import AgentTracer


def _make_graph():
    """Build a simple test graph."""
    from langgraph.graph import END, START, StateGraph

    def echo(state: dict) -> dict:
        messages = state.get("messages", [])
        last = messages[-1] if messages else {}
        content = last.get("content", "") if isinstance(last, dict) else str(last)
        return {"messages": [{"role": "ai", "content": f"echo: {content}"}]}

    builder = StateGraph(dict)
    builder.add_node("echo", echo)
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)
    return builder.compile()


def _simple_agent_break_line() -> int:
    """Return a stable executable line in examples.simple_agent.greeter."""
    import examples.simple_agent as simple_agent

    source_lines, start_lineno = inspect.getsourcelines(simple_agent.greeter)
    marker = 'messages = state.get("messages", [])'
    for offset, line in enumerate(source_lines):
        if marker in line:
            return start_lineno + offset
    raise AssertionError("Could not find greeter breakpoint marker line")


def _make_app(
    memory_renderer=None,
    store_renderer=None,
    state_renderer=None,
    output_renderer=None,
    tool_renderer=None,
    state_mutator=None,
):
    """Create a DebuggerApp for testing."""
    graph = _make_graph()
    eq = Queue()
    cq = Queue()
    bp = BreakpointManager()
    runner = AgentRunner(graph=graph, event_queue=eq, command_queue=cq, bp_manager=bp)
    return DebuggerApp(
        runner=runner,
        bp_manager=bp,
        memory_renderer=memory_renderer,
        store_renderer=store_renderer,
        state_renderer=state_renderer,
        output_renderer=output_renderer,
        tool_renderer=tool_renderer,
        state_mutator=state_mutator,
    )


@pytest.mark.asyncio
async def test_app_composes():
    """Test that the app composes without errors."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        assert app.query_one("#chat-log") is not None
        assert app.query_one("#state-panel") is not None
        assert app.query_one("#tools-panel") is not None
        assert app.query_one("#source-panel") is not None
        assert app.query_one("#vars-panel") is not None
        assert app.query_one("#stack-panel") is not None


@pytest.mark.asyncio
async def test_help_command():
    """Test /help command shows help text."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#chat-input")
        inp.value = "/help"
        await pilot.press("enter")
        await pilot.pause()


@pytest.mark.asyncio
async def test_break_command():
    """Test /break node command adds a breakpoint."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#chat-input")
        inp.value = "/break node echo"
        await pilot.press("enter")
        await pilot.pause()
        assert len(app.bp_manager.breakpoints) == 1
        assert app.bp_manager.breakpoints[0].name == "echo"


@pytest.mark.asyncio
async def test_break_node_missing_shows_error_and_does_not_add_breakpoint():
    """Invalid node breakpoint should show an error and not be added."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#chat-input")
        inp.value = "/break node does_not_exist"
        await pilot.press("enter")
        await pilot.pause()

        assert app.bp_manager.breakpoints == []
        chat_log = app.query_one("#chat-log")
        lines_text = [line.text for line in chat_log.lines]
        assert any(
            "Error: node 'does_not_exist' does not exist." in text
            for text in lines_text
        )


@pytest.mark.asyncio
async def test_breakpoint_banner_not_repeated_when_stepping():
    """Stepping should not append duplicate breakpoint chat banners."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        frame = sys._getframe()
        event = BreakpointHit(
            frame=frame,
            filename=frame.f_code.co_filename,
            lineno=frame.f_lineno,
            node="echo",
        )

        app._handle_event(event)
        app._suppress_breakpoint_chat_once = True
        app._handle_event(event)
        await pilot.pause()

        chat_log = app.query_one("#chat-log")
        lines_text = [line.text for line in chat_log.lines]
        breakpoint_lines = [
            text for text in lines_text if "⏸ Breakpoint:" in text
        ]
        assert len(breakpoint_lines) == 1


@pytest.mark.asyncio
async def test_all_break_commands_and_clear():
    """Test all /break variants and clear command."""
    app = _make_app()
    breakpoint_line = _simple_agent_break_line()
    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#chat-input")

        for cmd in (
            "/break node echo",
            "/break tool fake_tool",
            "/break state messages",
            "/break transition",
            f"/break line examples/simple_agent.py:{breakpoint_line}",
        ):
            inp.value = cmd
            await pilot.press("enter")
            await pilot.pause()

        bps = app.bp_manager.breakpoints
        assert len(bps) == 5
        assert [bp.type for bp in bps] == [
            BreakpointType.NODE,
            BreakpointType.TOOL,
            BreakpointType.STATE,
            BreakpointType.TRANSITION,
            BreakpointType.LINE,
        ]

        inp.value = "/breakpoints"
        await pilot.press("enter")
        await pilot.pause()
        tabs = app.query_one("#bottom-tabs")
        assert tabs.active == "bp-tab"

        inp.value = "/clear bp"
        await pilot.press("enter")
        await pilot.pause()
        assert app.bp_manager.breakpoints == []


@pytest.mark.asyncio
async def test_slash_debug_aliases_enqueue_debug_commands():
    """Test /c /n /s /r aliases map to debug commands at breakpoints."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#chat-input")

        for cmd, expected in (
            ("/c", DebugCommand.CONTINUE),
            ("/n", DebugCommand.STEP_OVER),
            ("/s", DebugCommand.STEP_INTO),
            ("/r", DebugCommand.STEP_OUT),
        ):
            app._at_breakpoint = True
            inp.value = cmd
            await pilot.press("enter")
            await pilot.pause()
            assert app.command_queue.get_nowait() == expected

        app._at_breakpoint = False
        inp.value = "/c"
        await pilot.press("enter")
        await pilot.pause()
        with pytest.raises(Empty):
            app.command_queue.get_nowait()


@pytest.mark.asyncio
async def test_long_debug_commands_skipped_from_history_at_breakpoints():
    """Long debug slash commands should not be persisted while paused."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#chat-input")
        app._input_history = []

        for cmd in ("/continue", "/next", "/step", "/return"):
            app._at_breakpoint = True
            inp.disabled = False
            inp.value = cmd
            await pilot.press("enter")
            await pilot.pause()

        assert app._input_history == []

        app._at_breakpoint = False
        inp.value = "/continue"
        await pilot.press("enter")
        await pilot.pause()
        assert app._input_history == ["/continue"]


@pytest.mark.asyncio
async def test_q_input_exits_app(monkeypatch):
    """Test entering 'q' in the input exits and signals the worker."""
    app = _make_app()
    exited = {"called": False}

    def _fake_exit(*args, **kwargs):
        exited["called"] = True

    monkeypatch.setattr(app, "exit", _fake_exit)

    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#chat-input")
        inp.value = "q"
        await pilot.press("enter")
        await pilot.pause()

    assert exited["called"] is True
    assert app.command_queue.get_nowait() == DebugCommand.QUIT


@pytest.mark.asyncio
async def test_slash_quit_exits_and_signals_worker(monkeypatch):
    """Test /quit exits and enqueues DebugCommand.QUIT."""
    app = _make_app()
    exited = {"called": False}

    def _fake_exit(*args, **kwargs):
        exited["called"] = True

    monkeypatch.setattr(app, "exit", _fake_exit)

    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#chat-input")
        inp.value = "/quit"
        await pilot.press("enter")
        await pilot.pause()

    assert exited["called"] is True
    assert app.command_queue.get_nowait() == DebugCommand.QUIT


@pytest.mark.asyncio
async def test_send_message_no_breakpoint():
    """Test sending a message without breakpoints runs to completion."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#chat-input")
        inp.value = "hello"
        await pilot.press("enter")
        # Wait for processing
        for _ in range(30):
            await pilot.pause()
            time.sleep(0.05)


@pytest.mark.asyncio
async def test_run_finished_resets_breakpoint_input_mode():
    """Run finished should clear breakpoint mode and re-enable input."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#chat-input")
        app._at_breakpoint = True
        app._processing = True
        inp.disabled = True

        app._handle_event(RunFinishedEvent())
        await pilot.pause()

        assert app._at_breakpoint is False
        assert app._processing is False
        assert inp.disabled is False


@pytest.mark.asyncio
async def test_tab_navigation_slash_commands():
    """Slash commands should switch the bottom tab deterministically."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#chat-input")
        tabs = app.query_one("#bottom-tabs")

        inp.value = "/logs"
        await pilot.press("enter")
        await pilot.pause()
        assert tabs.active == "logs-tab"

        inp.value = "/source"
        await pilot.press("enter")
        await pilot.pause()
        assert tabs.active == "source-tab"

        inp.value = "/messages"
        await pilot.press("enter")
        await pilot.pause()
        assert tabs.active == "messages-tab"


@pytest.mark.asyncio
async def test_memory_renderer_is_used_for_backend_store_snapshots():
    """Legacy memory renderer should only customize backend store snapshots."""

    class _MemoryRenderer:
        def render_memory(self, snapshot):
            return MemoryRenderModel(lines=["[bold]custom memory[/bold]"])

    app = _make_app(memory_renderer=_MemoryRenderer())
    async with app.run_test(size=(120, 40)) as pilot:
        app._handle_event(
            StateUpdateEvent(
                values={"memory": {"k": "v"}},
                store_items={"memories/u1": {"k": "v"}},
                store_source="backend",
                step=1,
            )
        )
        await pilot.pause()
        store_panel = app.query_one("#store-panel")
        assert store_panel.custom_lines == ["[bold]custom memory[/bold]"]


@pytest.mark.asyncio
async def test_store_auto_expands_when_memory_first_appears():
    """Store panel should auto-expand on first non-empty backend store snapshot."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        store_collapsible = app.query_one("#store-collapsible")
        assert store_collapsible.collapsed is True

        app._handle_event(StateUpdateEvent(values={"foo": "bar"}, step=1))
        await pilot.pause()
        assert store_collapsible.collapsed is True

        app._handle_event(
            StateUpdateEvent(
                values={"foo": "bar"},
                store_items={"memories/user-1": {"k": {"v": 1}}},
                store_source="backend",
                step=2,
            )
        )
        await pilot.pause()
        assert store_collapsible.collapsed is False


@pytest.mark.asyncio
async def test_store_stays_visible_across_breakpoint_step_hits():
    """Store should stay visible while stepping at breakpoints with store data."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        store_collapsible = app.query_one("#store-collapsible")
        frame = sys._getframe()

        app._handle_event(
            StateUpdateEvent(
                values={"x": 1},
                store_items={"memories/user-1": {"k": {"v": 1}}},
                store_source="backend",
                step=1,
            )
        )
        await pilot.pause()
        assert store_collapsible.collapsed is False

        # Simulate another step stop and ensure store is still shown.
        store_collapsible.collapsed = True
        app._suppress_breakpoint_chat_once = True
        second = BreakpointHit(
            frame=frame,
            filename=frame.f_code.co_filename,
            lineno=frame.f_lineno + 1,
            node="echo",
            graph_state={"x": 1},
        )
        app._handle_event(second)
        await pilot.pause()
        assert store_collapsible.collapsed is False


@pytest.mark.asyncio
async def test_memory_renderer_failure_falls_back_to_generic_store():
    """Renderer errors should fall back to generic backend store panel."""

    class _BrokenMemoryRenderer:
        def render_memory(self, snapshot):
            raise RuntimeError("boom")

    app = _make_app(memory_renderer=_BrokenMemoryRenderer())
    async with app.run_test(size=(120, 40)) as pilot:
        app._handle_event(
            StateUpdateEvent(
                values={"memory": {"k": "v"}},
                store_items={"memories/u1": {"k": "v"}},
                store_source="backend",
                step=1,
            )
        )
        await pilot.pause()
        store_panel = app.query_one("#store-panel")
        assert store_panel.custom_lines is None
        assert store_panel.items == {"memories/u1": {"k": "v"}}


@pytest.mark.asyncio
async def test_memory_renderer_is_ignored_without_backend_store_snapshot():
    """No backend store should not fall back to state-derived memory."""

    class _MemoryRenderer:
        def render_memory(self, snapshot):
            return MemoryRenderModel(lines=["[bold]custom memory[/bold]"])

    app = _make_app(memory_renderer=_MemoryRenderer())
    async with app.run_test(size=(120, 40)) as pilot:
        app._handle_event(
            StateUpdateEvent(values={"memory": {"k": "v"}}, step=1)
        )
        await pilot.pause()
        store_panel = app.query_one("#store-panel")
        assert store_panel.custom_lines is None
        assert store_panel.items == {}


@pytest.mark.asyncio
async def test_store_renderer_is_used_for_backend_snapshots():
    """Custom store renderer should render backend store snapshot lines."""

    class _StoreRenderer:
        def render_store(self, snapshot):
            return MemoryRenderModel(lines=["[bold]custom store[/bold]"])

    app = _make_app(store_renderer=_StoreRenderer())
    async with app.run_test(size=(120, 40)) as pilot:
        app._handle_event(
            StateUpdateEvent(
                values={"x": 1},
                store_items={"memories/u": {"k": "v"}},
                store_source="backend",
                step=1,
            )
        )
        await pilot.pause()
        store_panel = app.query_one("#store-panel")
        assert store_panel.custom_lines == ["[bold]custom store[/bold]"]


@pytest.mark.asyncio
async def test_state_renderer_is_used_for_state_updates():
    """Custom state renderer should replace default state panel rendering."""

    class _StateRenderer:
        def render_state(self, snapshot):
            return StateRenderModel(lines=["[bold]custom state[/bold]"])

    app = _make_app(state_renderer=_StateRenderer())
    async with app.run_test(size=(120, 40)) as pilot:
        app._handle_event(StateUpdateEvent(values={"foo": "bar"}, step=1))
        await pilot.pause()
        state_panel = app.query_one("#state-panel")
        rendered = state_panel.renderable
        assert rendered is not None
        assert "custom state" in rendered.plain


@pytest.mark.asyncio
async def test_output_renderer_handles_claimed_payload():
    """Output renderer should render claimed payloads in chat."""

    class _OutputRenderer:
        def __init__(self):
            self.called = False

        def can_render(self, payload):
            return bool(payload.get("recommendations"))

        def render_chat_output(self, payload, state, messages):
            self.called = True
            return ChatRenderModel(
                lines=["[bold green]Recommendations[/bold green]"]
            )

    renderer = _OutputRenderer()
    app = _make_app(output_renderer=renderer)
    async with app.run_test(size=(120, 40)) as pilot:
        app._handle_event(
            AgentResponseEvent(
                text="fallback",
                payload={"recommendations": [{"title": "A"}]},
            )
        )
        await pilot.pause()
        assert renderer.called is True


@pytest.mark.asyncio
async def test_output_renderer_empty_lines_falls_back_to_plain_text():
    """Empty renderer output should not suppress response text fallback."""

    class _OutputRenderer:
        def can_render(self, payload):
            return True

        def render_chat_output(self, payload, state, messages):
            return ChatRenderModel(lines=[])

    app = _make_app(output_renderer=_OutputRenderer())
    async with app.run_test(size=(120, 40)) as pilot:
        app._handle_event(
            AgentResponseEvent(
                text="fallback text",
                payload={"type": "ai"},
            )
        )
        await pilot.pause()

        chat_log = app.query_one("#chat-log")
        lines_text = [line.text for line in chat_log.lines]
        assert any("fallback text" in text for text in lines_text)


@pytest.mark.asyncio
async def test_tool_renderer_is_used_for_tool_panel():
    """Tool renderer should replace default tool panel rendering."""

    class _ToolRenderer:
        def render_tools(self, snapshot):
            assert isinstance(snapshot.get("tool_calls"), list)
            return ToolRenderModel(lines=["[bold]custom tools[/bold]"])

    app = _make_app(tool_renderer=_ToolRenderer())
    async with app.run_test(size=(120, 40)) as pilot:
        app._turn_counter = 1
        app._handle_event(
            ToolCallEvent(
                name="search",
                args={"query": "gifts"},
                tool_call_id="tc_1",
            )
        )
        await pilot.pause()

        panel = app.query_one("#tools-panel")
        lines_text = [line.text for line in panel.lines]
        assert any("custom tools" in text for text in lines_text)


@pytest.mark.asyncio
async def test_invoke_agent_does_not_clear_tool_history(monkeypatch):
    """New turns should preserve the existing Tools pane history."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = app.query_one("#tools-panel")
        called = {"clear": False}

        def _mark_clear():
            called["clear"] = True

        monkeypatch.setattr(panel, "clear_records", _mark_clear)

        inp = app.query_one("#chat-input")
        inp.value = "hello"
        await pilot.press("enter")
        await pilot.pause()

        assert called["clear"] is False


@pytest.mark.asyncio
async def test_clear_command_calls_state_mutator():
    """`/clear <mutation>` should clear local state and invoke provider."""

    class _Provider:
        def __init__(self):
            self.called = False
            self.mutation = ""

        def mutate_state(self, mutation, args, current_state, runner):
            self.called = True
            self.mutation = mutation
            return StateMutationResult(
                applied=True,
                message="mutation applied",
            )

    provider = _Provider()
    app = _make_app(state_mutator=provider)
    app._current_state = {"memory": {"x": 1}, "messages": [{"role": "ai"}]}

    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#chat-input")
        inp.value = "/clear memory"
        await pilot.press("enter")
        await pilot.pause()

        assert provider.called is True
        assert provider.mutation == "memory"
        assert app._current_state == {}


def test_tracer_breakpoint_and_continue():
    """Test tracer breakpoint fires and continue resumes (direct, no UI).

    Uses graph.invoke (sync) so the node function executes in the
    same thread as sys.settrace -- matching the real runner behavior.
    """
    from examples.simple_agent import graph

    eq = Queue()
    cq = Queue()
    bp = BreakpointManager()
    bp.add_node("greeter")
    tracer = AgentTracer(event_queue=eq, command_queue=cq, bp_manager=bp)

    def run_graph():
        graph.invoke(
            {"messages": [{"role": "human", "content": "test"}]},
            config={"callbacks": [tracer]},
        )

    t = threading.Thread(target=run_graph, daemon=True)
    t.start()

    # Wait for breakpoint
    hit = None
    for _ in range(100):
        try:
            event = eq.get(timeout=0.1)
            if isinstance(event, BreakpointHit):
                hit = event
                break
        except Exception:
            pass

    assert hit is not None, "Breakpoint should have fired"
    assert hit.node == "greeter"
    # Should land in the actual user file, not <string> or queue.py
    assert "simple_agent" in hit.filename, (
        f"Expected simple_agent.py, got {hit.filename}"
    )

    # Send continue
    cq.put(DebugCommand.CONTINUE)

    # Wait for thread to finish
    t.join(timeout=5)
    assert not t.is_alive(), "Graph should have finished after continue"


def test_tracer_line_breakpoint_and_continue():
    """Test Python file:line breakpoints pause on matching line."""
    from examples.simple_agent import graph

    eq = Queue()
    cq = Queue()
    bp = BreakpointManager()
    breakpoint_line = _simple_agent_break_line()
    bp.add_line(
        str(Path("examples/simple_agent.py").resolve()),
        breakpoint_line,
    )
    tracer = AgentTracer(event_queue=eq, command_queue=cq, bp_manager=bp)

    def run_graph():
        graph.invoke(
            {"messages": [{"role": "human", "content": "test"}]},
            config={"callbacks": [tracer]},
        )

    t = threading.Thread(target=run_graph, daemon=True)
    t.start()

    hit = None
    for _ in range(120):
        try:
            event = eq.get(timeout=0.1)
            if isinstance(event, BreakpointHit):
                hit = event
                break
        except Exception:
            pass

    assert hit is not None, "Expected line breakpoint hit"
    assert "simple_agent.py" in hit.filename
    assert hit.lineno == breakpoint_line

    cq.put(DebugCommand.CONTINUE)
    t.join(timeout=5)
    assert not t.is_alive(), "Graph should have finished after continue"


def test_tracer_nested_chain_start_end_attribution():
    """Test nested chain callbacks emit correctly attributed end events."""
    from examples.simple_agent import graph
    from agent_debugger.events import NodeEndEvent, NodeStartEvent

    eq = Queue()
    cq = Queue()
    tracer = AgentTracer(
        event_queue=eq,
        command_queue=cq,
        bp_manager=BreakpointManager(),
    )

    graph.invoke(
        {"messages": [{"role": "human", "content": "test"}]},
        config={"callbacks": [tracer]},
    )

    starts: list[str] = []
    ends: list[str] = []
    for _ in range(100):
        try:
            event = eq.get_nowait()
        except Exception:
            break
        if isinstance(event, NodeStartEvent):
            starts.append(event.node)
        elif isinstance(event, NodeEndEvent):
            ends.append(event.node)

    assert starts == ["LangGraph", "greeter"]
    assert ends == ["greeter", "LangGraph"]


def test_tracer_quit_command_unblocks_breakpoint():
    """Test QUIT command releases a tracer waiting at a breakpoint."""
    from examples.simple_agent import graph

    eq = Queue()
    cq = Queue()
    bp = BreakpointManager()
    bp.add_node("greeter")
    tracer = AgentTracer(event_queue=eq, command_queue=cq, bp_manager=bp)

    def run_graph():
        try:
            graph.invoke(
                {"messages": [{"role": "human", "content": "test"}]},
                config={"callbacks": [tracer]},
            )
        except Exception:
            # QUIT from bdb intentionally interrupts execution.
            pass

    t = threading.Thread(target=run_graph, daemon=True)
    t.start()

    hit = None
    for _ in range(100):
        try:
            event = eq.get(timeout=0.1)
            if isinstance(event, BreakpointHit):
                hit = event
                break
        except Exception:
            pass

    assert hit is not None, "Expected breakpoint hit"
    cq.put(DebugCommand.QUIT)
    t.join(timeout=5)
    assert not t.is_alive(), "Thread should exit after QUIT command"


def test_runner_tool_call_and_result_ids_match():
    """Test tool call events use the same tool_call_id as tool results."""
    from typing import Annotated

    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition

    from agent_debugger.events import RunFinishedEvent, ToolCallEvent, ToolResultEvent

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    class State(dict):
        messages: Annotated[list, add_messages]

    def agent(_: dict) -> dict:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 1, "b": 2},
                            "id": "call_123",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode([add]))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )
    builder.add_edge("tools", END)
    graph = builder.compile()

    eq = Queue()
    cq = Queue()
    runner = AgentRunner(
        graph=graph,
        event_queue=eq,
        command_queue=cq,
        bp_manager=BreakpointManager(),
    )

    runner.invoke("hello")

    call_ids: set[str] = set()
    result_ids: set[str] = set()
    for _ in range(300):
        try:
            event = eq.get(timeout=0.1)
        except Exception:
            continue
        if isinstance(event, ToolCallEvent):
            call_ids.add(event.tool_call_id)
        elif isinstance(event, ToolResultEvent):
            result_ids.add(event.tool_call_id)
        elif isinstance(event, RunFinishedEvent):
            break

    assert "call_123" in call_ids
    assert "call_123" in result_ids


def test_runner_emits_response_payload_for_simple_agent_memory():
    """Runner should emit AI response payload for the simple agent."""
    from agent_debugger.events import RunFinishedEvent
    from examples.simple_agent import graph

    eq = Queue()
    cq = Queue()
    runner = AgentRunner(
        graph=graph,
        event_queue=eq,
        command_queue=cq,
        bp_manager=BreakpointManager(),
    )

    runner.invoke("hello")

    response = None
    for _ in range(400):
        try:
            event = eq.get(timeout=0.1)
        except Exception:
            continue
        if isinstance(event, AgentResponseEvent):
            response = event
        elif isinstance(event, RunFinishedEvent):
            break

    assert response is not None
    assert response.payload.get("type") == "ai"
    assert "recommendations" not in response.payload


def test_runner_update_graph_state_without_checkpointer_returns_error():
    """update_graph_state should fail gracefully when no checkpointer exists."""
    from examples.simple_agent import graph

    runner = AgentRunner(
        graph=graph,
        event_queue=Queue(),
        command_queue=Queue(),
        bp_manager=BreakpointManager(),
    )
    ok, message = runner.update_graph_state({"memory": {}})
    assert ok is False
    assert "checkpointer" in message


def test_runner_emits_state_update_for_simple_agent_values_mode():
    """Runner should emit state updates for non-checkpoint graphs."""
    from agent_debugger.events import RunFinishedEvent
    from examples.simple_agent import graph

    eq = Queue()
    runner = AgentRunner(
        graph=graph,
        event_queue=eq,
        command_queue=Queue(),
        bp_manager=BreakpointManager(),
    )

    runner.invoke("hello")

    states: list[dict] = []
    for _ in range(300):
        try:
            event = eq.get(timeout=0.1)
        except Exception:
            continue
        if isinstance(event, StateUpdateEvent):
            states.append(event.values)
        elif isinstance(event, RunFinishedEvent):
            break

    assert states, "Expected at least one StateUpdateEvent"
    assert any("messages" in state for state in states)


def test_runner_emits_backend_store_items_when_store_available():
    """Runner should include backend store snapshot in state update events."""

    class _Item:
        def __init__(self, key, value):
            self.key = key
            self.value = value

    class _Store:
        def list_namespaces(self, **kwargs):
            return [("memories", "u1")]

        def search(self, namespace_prefix, **kwargs):
            if namespace_prefix == ("memories", "u1"):
                return [_Item("a", {"x": 1})]
            return []

    class _Graph:
        def __init__(self):
            self.store = _Store()

        def stream(self, input_data, config=None, stream_mode=None):
            yield ("values", {"messages": [{"role": "ai", "content": "ok"}]})

    eq = Queue()
    runner = AgentRunner(
        graph=_Graph(),  # type: ignore[arg-type]
        event_queue=eq,
        command_queue=Queue(),
        bp_manager=BreakpointManager(),
    )
    runner.invoke("hello")

    event = None
    for _ in range(100):
        try:
            candidate = eq.get(timeout=0.1)
        except Exception:
            continue
        if isinstance(candidate, StateUpdateEvent):
            event = candidate
            break

    assert event is not None
    assert event.store_source == "backend"
    assert event.store_items == {"memories/u1": {"a": {"x": 1}}}


def test_runner_reports_store_none_when_no_backend_store():
    """Runner should explicitly report missing backend store."""
    from examples.simple_agent import graph

    eq = Queue()
    runner = AgentRunner(
        graph=graph,
        event_queue=eq,
        command_queue=Queue(),
        bp_manager=BreakpointManager(),
    )
    runner.invoke("hello")

    event = None
    for _ in range(100):
        try:
            candidate = eq.get(timeout=0.1)
        except Exception:
            continue
        if isinstance(candidate, StateUpdateEvent):
            event = candidate
            break

    assert event is not None
    assert event.store_source in {"none", "unsupported", "error"}
    assert event.store_items == {}


def test_update_input_placeholder_logs_widget_errors(monkeypatch):
    """Test widget lookup failures are logged instead of silently swallowed."""
    app = _make_app()
    messages: list[str] = []

    def _fake_query_one(*args, **kwargs):
        raise RuntimeError("boom")

    def _fake_log(message: str, level: str = "info"):
        messages.append(f"{level}:{message}")

    monkeypatch.setattr(app, "query_one", _fake_query_one)
    monkeypatch.setattr(app, "_log", _fake_log)

    app._update_input_placeholder()
    assert any(
        "Failed to update input placeholder" in m for m in messages
    )


def test_log_falls_back_to_stderr_when_logs_panel_missing(monkeypatch, capsys):
    """Test _log prints to stderr if the logs panel is unavailable."""
    app = _make_app()

    def _fake_query_one(*args, **kwargs):
        raise RuntimeError("missing")

    monkeypatch.setattr(app, "query_one", _fake_query_one)
    app._log("hello", "info")
    captured = capsys.readouterr()
    assert "log panel unavailable" in captured.err


def test_poll_events_recovers_stale_processing_state():
    """If worker exits without a handled RunFinished event, UI recovers."""
    app = _make_app()
    app._processing = True
    app._at_breakpoint = False
    app._poll_events()
    assert app._processing is False


def test_runner_emits_response_for_assistant_role_messages():
    """Assistant-role messages should produce AgentResponseEvent."""

    class _Graph:
        def stream(self, input_data, config=None, stream_mode=None):
            yield (
                "values",
                {"messages": [{"role": "assistant", "content": "hello"}]},
            )

    eq = Queue()
    runner = AgentRunner(
        graph=_Graph(),  # type: ignore[arg-type]
        event_queue=eq,
        command_queue=Queue(),
        bp_manager=BreakpointManager(),
    )
    runner.invoke("hello")

    response = None
    for _ in range(100):
        try:
            event = eq.get(timeout=0.1)
        except Exception:
            continue
        if isinstance(event, AgentResponseEvent):
            response = event
        if isinstance(event, RunFinishedEvent):
            break

    assert response is not None
    assert response.payload["type"] == "ai"
    assert response.text == "hello"


def test_runner_emits_tool_result_for_role_tool_messages():
    """Tool-role messages should produce ToolResultEvent."""

    class _Graph:
        def stream(self, input_data, config=None, stream_mode=None):
            yield (
                "values",
                {
                    "messages": [
                        {
                            "role": "tool",
                            "tool_call_id": "call_1",
                            "content": "ok",
                        }
                    ]
                },
            )

    eq = Queue()
    runner = AgentRunner(
        graph=_Graph(),  # type: ignore[arg-type]
        event_queue=eq,
        command_queue=Queue(),
        bp_manager=BreakpointManager(),
    )
    runner.invoke("hello")

    tool_result = None
    for _ in range(100):
        try:
            event = eq.get(timeout=0.1)
        except Exception:
            continue
        if isinstance(event, ToolResultEvent):
            tool_result = event
        if isinstance(event, RunFinishedEvent):
            break

    assert tool_result is not None
    assert tool_result.tool_call_id == "call_1"
    assert tool_result.result == "ok"


@pytest.mark.asyncio
async def test_messages_panel_renders_assistant_role_content():
    """Messages panel should render OpenAI-style assistant roles."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        app._handle_event(
            StateUpdateEvent(
                values={
                    "messages": [
                        {"role": "human", "content": "hi"},
                        {"role": "assistant", "content": "hello there"},
                    ]
                },
                step=1,
            )
        )
        await pilot.pause()

        panel = app.query_one("#messages-panel")
        lines_text = [line.text for line in panel.lines]
        assert any("hello there" in text for text in lines_text)


def test_runner_checks_state_breakpoints_against_full_state_snapshots(monkeypatch):
    """State breakpoints should compare successive full-state snapshots."""
    bp = BreakpointManager()
    bp.add_state("counter")

    class _Graph:
        pass

    runner = AgentRunner(
        graph=_Graph(),  # type: ignore[arg-type]
        event_queue=Queue(),
        command_queue=Queue(),
        bp_manager=bp,
    )

    hits: list[str] = []
    monkeypatch.setattr(
        runner.tracer,
        "request_break_on_next_user_frame",
        lambda: hits.append("hit"),
    )

    runner._emit_state_update({"other": 1})
    runner._emit_state_update({"other": 1, "counter": 0})
    runner._emit_state_update({"other": 2, "counter": 0})

    assert hits == ["hit"]


def test_step_actions_replace_pending_debug_commands(monkeypatch):
    """Repeated step actions should keep only the most recent command."""
    app = _make_app()
    app._at_breakpoint = True
    monkeypatch.setattr(app, "_log", lambda *args, **kwargs: None)

    app.action_step_over()
    app.action_step_into()
    app.action_step_out()

    assert app.command_queue.get_nowait() == DebugCommand.STEP_OUT
    with pytest.raises(Empty):
        app.command_queue.get_nowait()


def test_tool_panel_filter_preserves_zero_values():
    """Tool args filtering should keep meaningful zero values."""
    from agent_debugger.panels.tools import ToolCallsPanel

    panel = ToolCallsPanel()
    filtered = panel._filter_empty(
        {"offset": 0, "count": "0", "blank": "", "none": None}
    )
    assert filtered == {"offset": 0, "count": "0"}


@pytest.mark.asyncio
async def test_tool_panel_groups_calls_by_turn():
    """Default tools panel should annotate records with turn headers."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        app._turn_counter = 1
        app._handle_event(
            ToolCallEvent(
                name="search",
                args={"q": "a"},
                tool_call_id="tc_1",
            )
        )
        app._turn_counter = 2
        app._handle_event(
            ToolCallEvent(
                name="rank",
                args={"q": "b"},
                tool_call_id="tc_2",
            )
        )
        await pilot.pause()

        panel = app.query_one("#tools-panel")
        lines_text = [line.text for line in panel.lines]
        assert any("Turn 1" in text for text in lines_text)
        assert any("Turn 2" in text for text in lines_text)


@pytest.mark.asyncio
async def test_breakpoint_placeholder_has_no_trailing_parenthesis():
    """Breakpoint placeholder copy should not include a trailing ')'."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        app.bp_manager.add_node("echo")
        app._update_input_placeholder()
        await pilot.pause()

        inp = app.query_one("#chat-input")
        assert not inp.placeholder.endswith(")")


@pytest.mark.asyncio
async def test_poll_events_recovery_reenables_input():
    """Stale processing recovery should re-enable the input widget."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#chat-input")
        app._processing = True
        app._at_breakpoint = False
        inp.disabled = True

        app._poll_events()
        await pilot.pause()

        assert app._processing is False
        assert inp.disabled is False


@pytest.mark.asyncio
async def test_breakpoint_event_failure_auto_continues_worker(monkeypatch):
    """Breakpoint UI failures should auto-continue to avoid deadlock."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        app._processing = True

        def _boom(_event):
            raise RuntimeError("render failed")

        monkeypatch.setattr(app, "_handle_breakpoint", _boom)
        event = BreakpointHit(frame=sys._getframe())
        app._handle_event(event)
        await pilot.pause()

        assert app._at_breakpoint is False
        assert app.command_queue.get_nowait() == DebugCommand.CONTINUE
