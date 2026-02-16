"""Tests for debug key handling (c/n/s/r) at breakpoints."""

import sys
import threading
from queue import Queue

from agent_debugger.app import DebuggerApp
from agent_debugger.breakpoints import BreakpointManager
from agent_debugger.events import BreakpointHit, DebugCommand
from agent_debugger.runner import AgentRunner
from agent_debugger.tracer import AgentTracer


def _make_graph():
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


def _make_app_with_breakpoint():
    """Create a DebuggerApp with a breakpoint on 'echo' node."""
    graph = _make_graph()
    eq = Queue()
    cq = Queue()
    bp = BreakpointManager()
    bp.add_node("echo")
    runner = AgentRunner(graph=graph, event_queue=eq, command_queue=cq, bp_manager=bp)
    return DebuggerApp(runner=runner, bp_manager=bp), cq


def test_step_over_command_direct():
    """Test that STEP_OVER command works at the tracer level."""
    from examples.simple_agent import graph

    eq = Queue()
    cq = Queue()
    bp = BreakpointManager()
    bp.add_node("greeter")
    tracer = AgentTracer(event_queue=eq, command_queue=cq, bp_manager=bp)

    hits = []

    def run_graph():
        graph.invoke(
            {"messages": [{"role": "human", "content": "test"}]},
            config={"callbacks": [tracer]},
        )

    t = threading.Thread(target=run_graph, daemon=True)
    t.start()

    # Collect breakpoint hits, send step_over for first 3, then continue
    for _ in range(200):
        try:
            event = eq.get(timeout=0.1)
            if isinstance(event, BreakpointHit):
                hits.append(event)
                if len(hits) < 4:
                    cq.put(DebugCommand.STEP_OVER)
                else:
                    cq.put(DebugCommand.CONTINUE)
        except Exception:
            pass
        if not t.is_alive():
            break

    t.join(timeout=5)

    # Should have hit multiple lines (stepping through greeter)
    assert len(hits) >= 2, f"Expected multiple breakpoint hits from stepping, got {len(hits)}"
    # Each hit should be on a different line
    lines = [h.lineno for h in hits]
    assert len(set(lines)) > 1, f"Expected different lines, got {lines}"
    assert len(hits) >= 2  # sanity: step-over produced multiple hits


def test_user_return_after_continue_does_not_wait_for_command(monkeypatch):
    """Tracer should not block again on return after explicit continue."""
    tracer = AgentTracer(
        event_queue=Queue(),
        command_queue=Queue(),
        bp_manager=BreakpointManager(),
    )
    tracer._last_debug_command = DebugCommand.CONTINUE

    monkeypatch.setattr(tracer, "_is_library_frame", lambda frame: False)

    def _unexpected_wait():
        raise AssertionError("user_return should not wait after CONTINUE")

    monkeypatch.setattr(tracer, "_wait_for_command", _unexpected_wait)
    tracer.user_return(sys._getframe(), None)
