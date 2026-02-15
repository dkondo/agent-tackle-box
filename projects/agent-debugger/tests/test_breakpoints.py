"""Tests for the breakpoint manager."""

from agent_debugger.breakpoints import BreakpointManager, BreakpointType


class TestBreakpointManager:
    """Tests for BreakpointManager."""

    def test_add_node_breakpoint(self):
        mgr = BreakpointManager()
        bp = mgr.add_node("search")
        assert bp.type == BreakpointType.NODE
        assert bp.name == "search"
        assert bp.enabled is True
        assert len(mgr.breakpoints) == 1

    def test_add_tool_breakpoint(self):
        mgr = BreakpointManager()
        bp = mgr.add_tool("search_listings")
        assert bp.type == BreakpointType.TOOL
        assert bp.name == "search_listings"

    def test_add_state_breakpoint(self):
        mgr = BreakpointManager()
        bp = mgr.add_state("messages")
        assert bp.type == BreakpointType.STATE
        assert bp.key == "messages"

    def test_add_transition_breakpoint(self):
        mgr = BreakpointManager()
        bp = mgr.add_transition()
        assert bp.type == BreakpointType.TRANSITION

    def test_add_line_breakpoint(self):
        mgr = BreakpointManager()
        bp = mgr.add_line("agent.py", 42)
        assert bp.type == BreakpointType.LINE
        assert bp.filename == "agent.py"
        assert bp.lineno == 42

    def test_should_break_on_node(self):
        mgr = BreakpointManager()
        mgr.add_node("search")
        assert mgr.should_break_on_node("search") is True
        assert mgr.should_break_on_node("other") is False

    def test_should_break_on_tool(self):
        mgr = BreakpointManager()
        mgr.add_tool("search_listings")
        assert mgr.should_break_on_tool("search_listings") is True
        assert mgr.should_break_on_tool("other_tool") is False

    def test_should_break_on_state_change(self):
        mgr = BreakpointManager()
        mgr.add_state("messages")
        old = {"messages": [1, 2]}
        new = {"messages": [1, 2, 3]}
        assert mgr.should_break_on_state(old, new) is True

    def test_should_not_break_on_unchanged_state(self):
        mgr = BreakpointManager()
        mgr.add_state("messages")
        old = {"messages": [1, 2]}
        new = {"messages": [1, 2]}
        assert mgr.should_break_on_state(old, new) is False

    def test_transition_breaks_on_any_node(self):
        mgr = BreakpointManager()
        mgr.add_transition()
        assert mgr.should_break_on_node("any_node") is True
        assert mgr.should_break_on_node("another_node") is True

    def test_remove_breakpoint(self):
        mgr = BreakpointManager()
        mgr.add_node("search")
        mgr.add_tool("tool1")
        assert len(mgr.breakpoints) == 2
        removed = mgr.remove(0)
        assert removed is not None
        assert removed.name == "search"
        assert len(mgr.breakpoints) == 1

    def test_toggle_breakpoint(self):
        mgr = BreakpointManager()
        mgr.add_node("search")
        assert mgr.breakpoints[0].enabled is True
        mgr.toggle(0)
        assert mgr.breakpoints[0].enabled is False
        # Disabled breakpoint should not fire
        assert mgr.should_break_on_node("search") is False

    def test_clear_all(self):
        mgr = BreakpointManager()
        mgr.add_node("a")
        mgr.add_tool("b")
        mgr.add_state("c")
        count = mgr.clear()
        assert count == 3
        assert len(mgr.breakpoints) == 0

    def test_hit_count(self):
        mgr = BreakpointManager()
        mgr.add_node("search")
        mgr.should_break_on_node("search")
        mgr.should_break_on_node("search")
        assert mgr.breakpoints[0].hits == 2

    def test_node_names_property(self):
        mgr = BreakpointManager()
        mgr.add_node("a")
        mgr.add_node("b")
        mgr.add_tool("c")
        assert mgr.node_names == {"a", "b"}

    def test_tool_names_property(self):
        mgr = BreakpointManager()
        mgr.add_tool("x")
        mgr.add_tool("y")
        mgr.add_node("z")
        assert mgr.tool_names == {"x", "y"}

    def test_has_line_breakpoints_property(self):
        mgr = BreakpointManager()
        assert mgr.has_line_breakpoints is False
        mgr.add_line("agent.py", 42)
        assert mgr.has_line_breakpoints is True
        assert len(mgr.line_breakpoints) == 1
