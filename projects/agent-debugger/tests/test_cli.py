"""Tests for adb CLI behavior."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from click.testing import CliRunner

from agent_debugger import cli


class DummyMemoryRenderer:
    def render_memory(self, snapshot):
        return None


class DummyOutputRenderer:
    def can_render(self, payload):
        return False

    def render_chat_output(self, payload, state, messages):
        return None


class DummyStoreRenderer:
    def render_store(self, snapshot):
        return None


class DummyStateRenderer:
    def render_state(self, snapshot):
        return None


class DummyStateMutator:
    def mutate_state(self, mutation, args, current_state, runner):
        return None


def _install_fake_compiled_state_graph(monkeypatch):
    """Install a minimal langgraph.graph.state module for CLI tests."""
    langgraph_mod = types.ModuleType("langgraph")
    graph_mod = types.ModuleType("langgraph.graph")
    state_mod = types.ModuleType("langgraph.graph.state")

    class CompiledStateGraph:
        pass

    state_mod.CompiledStateGraph = CompiledStateGraph
    graph_mod.state = state_mod
    langgraph_mod.graph = graph_mod

    monkeypatch.setitem(sys.modules, "langgraph", langgraph_mod)
    monkeypatch.setitem(sys.modules, "langgraph.graph", graph_mod)
    monkeypatch.setitem(sys.modules, "langgraph.graph.state", state_mod)
    return CompiledStateGraph


def test_attach_invokes_run_app_with_thread_id(monkeypatch):
    """attach should load the graph and forward thread id to _run_app."""
    captured: dict[str, object] = {}

    def _fake_run_app(graph, thread_id=None, **kwargs):
        captured["graph"] = graph
        captured["thread_id"] = thread_id

    monkeypatch.setattr(cli, "_load_graph", lambda _: object())
    monkeypatch.setattr(cli, "_run_app", _fake_run_app)

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["attach", "dummy.module:graph", "--thread-id", "t-123"],
    )

    assert result.exit_code == 0, result.output
    assert captured["thread_id"] == "t-123"
    assert captured["graph"] is not None


def test_attach_rejects_invalid_graph_ref():
    """attach should reject graph refs without module:attribute format."""
    runner = CliRunner()
    result = runner.invoke(cli.main, ["attach", "badref"])
    assert result.exit_code != 0
    assert "Graph reference must be 'module:attribute'" in result.output


def test_run_auto_detects_graph_and_invokes_app(monkeypatch):
    """run should auto-detect graph object from script namespace."""
    captured: dict[str, object] = {}

    def _fake_run_app(graph, thread_id=None, **kwargs):
        captured["graph"] = graph
        captured["thread_id"] = thread_id

    _install_fake_compiled_state_graph(monkeypatch)
    monkeypatch.setattr(cli, "_run_app", _fake_run_app)

    script = Path("tests/_tmp_graph_script.py")
    script.write_text(
        "from langgraph.graph.state import CompiledStateGraph\n"
        "class _Graph(CompiledStateGraph):\n"
        "    pass\n"
        "graph = _Graph()\n",
        encoding="utf-8",
    )
    try:
        runner = CliRunner()
        result = runner.invoke(cli.main, ["run", str(script)])

        assert result.exit_code == 0, result.output
        assert "Auto-detected graph: graph" in result.output
        assert captured["graph"] is not None
        assert captured["thread_id"] is None
    finally:
        script.unlink(missing_ok=True)


def test_run_with_graph_attr_and_thread_id(monkeypatch):
    """run should respect --graph and --thread-id options."""
    captured: dict[str, object] = {}

    def _fake_run_app(graph, thread_id=None, **kwargs):
        captured["graph"] = graph
        captured["thread_id"] = thread_id

    monkeypatch.setattr(cli, "_run_app", _fake_run_app)

    script = Path("tests/_tmp_graph_attr_script.py")
    script.write_text("graph = object()\n", encoding="utf-8")
    try:
        runner = CliRunner()
        result = runner.invoke(
            cli.main,
            ["run", str(script), "--graph", "graph", "--thread-id", "thr-1"],
        )

        assert result.exit_code == 0, result.output
        assert captured["graph"] is not None
        assert captured["thread_id"] == "thr-1"
    finally:
        script.unlink(missing_ok=True)


def test_attach_forwards_store_snapshot_options(monkeypatch):
    """attach should pass store snapshot controls to _run_app."""
    captured: dict[str, object] = {}

    def _fake_run_app(graph, thread_id=None, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "_load_graph", lambda _: object())
    monkeypatch.setattr(cli, "_run_app", _fake_run_app)

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "attach",
            "dummy.module:graph",
            "--store-prefix",
            "memories,user-1",
            "--store-max-namespaces",
            "7",
            "--store-items-per-namespace",
            "9",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["store_prefix"] == ("memories", "user-1")
    assert captured["store_max_namespaces"] == 7
    assert captured["store_items_per_namespace"] == 9


def test_attach_loads_optional_extensions(monkeypatch):
    """attach should load renderer/provider refs and pass them to _run_app."""
    captured: dict[str, object] = {}

    def _fake_run_app(graph, thread_id=None, **kwargs):
        captured["graph"] = graph
        captured["thread_id"] = thread_id
        captured.update(kwargs)

    monkeypatch.setattr(cli, "_load_graph", lambda _: object())
    monkeypatch.setattr(cli, "_run_app", _fake_run_app)

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "attach",
            "dummy.module:graph",
            "--memory-renderer",
            "tests.test_cli:DummyMemoryRenderer",
            "--output-renderer",
            "tests.test_cli:DummyOutputRenderer",
            "--store-renderer",
            "tests.test_cli:DummyStoreRenderer",
            "--state-renderer",
            "tests.test_cli:DummyStateRenderer",
            "--state-mutator",
            "tests.test_cli:DummyStateMutator",
        ],
    )

    assert result.exit_code == 0, result.output
    assert isinstance(captured["memory_renderer"], DummyMemoryRenderer)
    assert isinstance(captured["store_renderer"], DummyStoreRenderer)
    assert isinstance(captured["state_renderer"], DummyStateRenderer)
    assert isinstance(captured["output_renderer"], DummyOutputRenderer)
    assert isinstance(
        captured["state_mutator"],
        DummyStateMutator,
    )


def test_attach_extension_load_failure_warns_and_falls_back(monkeypatch):
    """Invalid extension references should warn and use default behavior."""
    captured: dict[str, object] = {}

    def _fake_run_app(graph, thread_id=None, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "_load_graph", lambda _: object())
    monkeypatch.setattr(cli, "_run_app", _fake_run_app)

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "attach",
            "dummy.module:graph",
            "--memory-renderer",
            "badref",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Warning: failed to load memory renderer 'badref'" in result.output
    assert captured["memory_renderer"] is None


def test_attach_supports_legacy_state_mutation_provider_flag(monkeypatch):
    """Legacy --state-mutation-provider should still map to state_mutator."""
    captured: dict[str, object] = {}

    def _fake_run_app(graph, thread_id=None, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "_load_graph", lambda _: object())
    monkeypatch.setattr(cli, "_run_app", _fake_run_app)

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "attach",
            "dummy.module:graph",
            "--state-mutation-provider",
            "tests.test_cli:DummyStateMutator",
        ],
    )

    assert result.exit_code == 0, result.output
    assert isinstance(captured["state_mutator"], DummyStateMutator)


def test_run_errors_when_graph_attr_missing():
    """run should fail when --graph points to a missing variable."""
    script = Path("tests/_tmp_graph_missing_script.py")
    script.write_text("graph = object()\n", encoding="utf-8")
    try:
        runner = CliRunner()
        result = runner.invoke(cli.main, ["run", str(script), "--graph", "missing"])
        assert result.exit_code != 0
        assert "Script has no variable 'missing'" in result.output
    finally:
        script.unlink(missing_ok=True)


def test_run_errors_for_non_graph_script(monkeypatch):
    """run should fail when script has no CompiledStateGraph."""
    _install_fake_compiled_state_graph(monkeypatch)
    script = Path("tests/_tmp_no_graph_script.py")
    script.write_text("x = 1\n", encoding="utf-8")
    try:
        runner = CliRunner()
        result = runner.invoke(cli.main, ["run", str(script)])
        assert result.exit_code != 0
        assert "No CompiledStateGraph found in script" in result.output
    finally:
        script.unlink(missing_ok=True)
