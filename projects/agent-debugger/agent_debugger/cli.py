"""CLI entry point for adb."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from queue import Queue
from typing import Any

import click

from agent_debugger.breakpoints import BreakpointManager
from agent_debugger.extensions import (
    ChatOutputRenderer,
    MemoryRenderer,
    StateMutator,
    StateRenderer,
    StoreRenderer,
    ToolRenderer,
)


def _load_graph(graph_ref: str) -> Any:
    """Load a graph from a module:attribute reference.

    Args:
        graph_ref: A string like "my_module:graph" or "my_package.module:app".

    Returns:
        The graph object.
    """
    if ":" not in graph_ref:
        raise click.BadParameter(f"Graph reference must be 'module:attribute', got '{graph_ref}'")
    module_path, attr_name = graph_ref.rsplit(":", 1)

    # Add cwd to sys.path so local modules can be found
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise click.BadParameter(f"Cannot import module '{module_path}': {e}") from e

    try:
        graph = getattr(module, attr_name)
    except AttributeError as e:
        raise click.BadParameter(f"Module '{module_path}' has no attribute '{attr_name}'") from e

    return graph


def _load_ref(ref: str, kind: str) -> Any:
    """Load an object from a module:attribute reference."""
    if ":" not in ref:
        raise click.BadParameter(f"{kind} reference must be 'module:attribute', got '{ref}'")
    module_path, attr_name = ref.rsplit(":", 1)
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise click.BadParameter(f"Cannot import module '{module_path}' for {kind}: {e}") from e

    try:
        return getattr(module, attr_name)
    except AttributeError as e:
        raise click.BadParameter(
            f"Module '{module_path}' has no attribute '{attr_name}' for {kind}"
        ) from e


def _load_optional_extension(
    ref: str | None,
    *,
    kind: str,
    required_methods: tuple[str, ...],
) -> Any | None:
    """Load extension object and gracefully fall back on failure."""
    if not ref:
        return None

    try:
        obj = _load_ref(ref, kind)
        if isinstance(obj, type):
            obj = obj()
        for method in required_methods:
            fn = getattr(obj, method, None)
            if not callable(fn):
                raise TypeError(f"{kind} must define callable '{method}'")
        return obj
    except Exception as e:
        click.echo(
            f"Warning: failed to load {kind} '{ref}': {e}. Falling back to default behavior.",
            err=True,
        )
        return None


def _run_app(
    graph: Any,
    thread_id: str | None = None,
    memory_renderer: MemoryRenderer | None = None,
    store_renderer: StoreRenderer | None = None,
    state_renderer: StateRenderer | None = None,
    output_renderer: ChatOutputRenderer | None = None,
    tool_renderer: ToolRenderer | None = None,
    state_mutator: StateMutator | None = None,
    state_mutation_provider: StateMutator | None = None,
    store_prefix: tuple[str, ...] | None = None,
    store_max_namespaces: int = 20,
    store_items_per_namespace: int = 20,
) -> None:
    """Initialize and run the debugger app."""
    from agent_debugger.app import DebuggerApp
    from agent_debugger.runner import AgentRunner

    event_queue: Queue = Queue()
    command_queue: Queue = Queue()
    bp_manager = BreakpointManager()

    runner = AgentRunner(
        graph=graph,
        event_queue=event_queue,
        command_queue=command_queue,
        bp_manager=bp_manager,
        store_namespace_prefix=store_prefix,
        store_max_namespaces=store_max_namespaces,
        store_items_per_namespace=store_items_per_namespace,
    )

    if thread_id:
        runner.configure(thread_id=thread_id)

    # Set up PYTHONBREAKPOINT support
    import agent_debugger as adb_module

    sys.breakpointhook = adb_module.set_trace

    app = DebuggerApp(
        runner=runner,
        bp_manager=bp_manager,
        memory_renderer=memory_renderer,
        store_renderer=store_renderer,
        state_renderer=state_renderer,
        output_renderer=output_renderer,
        tool_renderer=tool_renderer,
        state_mutator=state_mutator or state_mutation_provider,
    )
    app.run()


def _parse_store_prefix(raw: str | None) -> tuple[str, ...] | None:
    """Parse comma-delimited store namespace prefix."""
    if not raw:
        return None
    parts = [part.strip() for part in raw.split(",")]
    filtered = tuple(part for part in parts if part)
    return filtered or None


@click.group()
@click.version_option(version="0.1.0", prog_name="adb")
def main() -> None:
    """adb: Agent Debugger for LangChain/LangGraph."""
    pass


@main.command()
@click.argument("graph_ref")
@click.option(
    "--thread-id",
    "-t",
    default=None,
    help="Thread ID for checkpointed graphs.",
)
@click.option(
    "--memory-renderer",
    default=None,
    help="Optional memory renderer (module:Class).",
)
@click.option(
    "--store-renderer",
    default=None,
    help="Optional store renderer (module:Class).",
)
@click.option(
    "--state-renderer",
    default=None,
    help="Optional state renderer (module:Class).",
)
@click.option(
    "--output-renderer",
    default=None,
    help="Optional chat output renderer (module:Class).",
)
@click.option(
    "--tool-renderer",
    default=None,
    help="Optional tools pane renderer (module:Class).",
)
@click.option(
    "--state-mutator",
    "--state-mutation-provider",
    "state_mutator",
    default=None,
    help="Optional state mutator (module:Class).",
)
@click.option(
    "--store-prefix",
    default=None,
    help="Optional backend store namespace prefix (comma-separated).",
)
@click.option(
    "--store-max-namespaces",
    default=20,
    type=int,
    show_default=True,
    help="Maximum number of store namespaces to show.",
)
@click.option(
    "--store-items-per-namespace",
    default=20,
    type=int,
    show_default=True,
    help="Maximum number of items per namespace to show.",
)
def attach(
    graph_ref: str,
    thread_id: str | None,
    memory_renderer: str | None,
    store_renderer: str | None,
    state_renderer: str | None,
    output_renderer: str | None,
    tool_renderer: str | None,
    state_mutator: str | None,
    store_prefix: str | None,
    store_max_namespaces: int,
    store_items_per_namespace: int,
) -> None:
    """Attach to a LangGraph graph object.

    GRAPH_REF is a module:attribute reference, e.g. 'my_agent:graph'.
    """
    graph = _load_graph(graph_ref)
    loaded_memory_renderer = _load_optional_extension(
        memory_renderer,
        kind="memory renderer",
        required_methods=("render_memory",),
    )
    loaded_store_renderer = _load_optional_extension(
        store_renderer,
        kind="store renderer",
        required_methods=("render_store",),
    )
    loaded_state_renderer = _load_optional_extension(
        state_renderer,
        kind="state renderer",
        required_methods=("render_state",),
    )
    loaded_output_renderer = _load_optional_extension(
        output_renderer,
        kind="output renderer",
        required_methods=("can_render", "render_chat_output"),
    )
    loaded_tool_renderer = _load_optional_extension(
        tool_renderer,
        kind="tool renderer",
        required_methods=("render_tools",),
    )
    loaded_state_mutator = _load_optional_extension(
        state_mutator,
        kind="state mutator",
        required_methods=("mutate_state",),
    )
    _run_app(
        graph,
        thread_id=thread_id,
        memory_renderer=loaded_memory_renderer,
        store_renderer=loaded_store_renderer,
        state_renderer=loaded_state_renderer,
        output_renderer=loaded_output_renderer,
        tool_renderer=loaded_tool_renderer,
        state_mutator=loaded_state_mutator,
        store_prefix=_parse_store_prefix(store_prefix),
        store_max_namespaces=max(1, store_max_namespaces),
        store_items_per_namespace=max(1, store_items_per_namespace),
    )


@main.command()
@click.argument("script", type=click.Path(exists=True))
@click.option(
    "--graph",
    "-g",
    "graph_attr",
    default=None,
    help="Attribute name of the graph in the script (default: auto-detect).",
)
@click.option(
    "--thread-id",
    "-t",
    default=None,
    help="Thread ID for checkpointed graphs.",
)
@click.option(
    "--memory-renderer",
    default=None,
    help="Optional memory renderer (module:Class).",
)
@click.option(
    "--store-renderer",
    default=None,
    help="Optional store renderer (module:Class).",
)
@click.option(
    "--state-renderer",
    default=None,
    help="Optional state renderer (module:Class).",
)
@click.option(
    "--output-renderer",
    default=None,
    help="Optional chat output renderer (module:Class).",
)
@click.option(
    "--tool-renderer",
    default=None,
    help="Optional tools pane renderer (module:Class).",
)
@click.option(
    "--state-mutator",
    "--state-mutation-provider",
    "state_mutator",
    default=None,
    help="Optional state mutator (module:Class).",
)
@click.option(
    "--store-prefix",
    default=None,
    help="Optional backend store namespace prefix (comma-separated).",
)
@click.option(
    "--store-max-namespaces",
    default=20,
    type=int,
    show_default=True,
    help="Maximum number of store namespaces to show.",
)
@click.option(
    "--store-items-per-namespace",
    default=20,
    type=int,
    show_default=True,
    help="Maximum number of items per namespace to show.",
)
def run(
    script: str,
    graph_attr: str | None,
    thread_id: str | None,
    memory_renderer: str | None,
    store_renderer: str | None,
    state_renderer: str | None,
    output_renderer: str | None,
    tool_renderer: str | None,
    state_mutator: str | None,
    store_prefix: str | None,
    store_max_namespaces: int,
    store_items_per_namespace: int,
) -> None:
    """Run a script and debug its LangGraph graph.

    SCRIPT is a Python file containing a CompiledStateGraph.
    """
    script_path = Path(script).resolve()

    # Add script's directory to sys.path
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Execute the script to find the graph
    namespace: dict[str, Any] = {"__name__": "__main__", "__file__": str(script_path)}
    with open(script_path) as f:
        code = compile(f.read(), str(script_path), "exec")
        exec(code, namespace)  # noqa: S102

    # Find the graph object
    graph = None
    if graph_attr:
        graph = namespace.get(graph_attr)
        if graph is None:
            raise click.BadParameter(f"Script has no variable '{graph_attr}'")
    else:
        # Auto-detect: look for CompiledStateGraph instances
        from langgraph.graph.state import CompiledStateGraph

        for name, obj in namespace.items():
            if isinstance(obj, CompiledStateGraph):
                graph = obj
                click.echo(f"Auto-detected graph: {name}")
                break

    if graph is None:
        raise click.UsageError(
            "No CompiledStateGraph found in script. Use --graph to specify the attribute name."
        )

    loaded_memory_renderer = _load_optional_extension(
        memory_renderer,
        kind="memory renderer",
        required_methods=("render_memory",),
    )
    loaded_store_renderer = _load_optional_extension(
        store_renderer,
        kind="store renderer",
        required_methods=("render_store",),
    )
    loaded_state_renderer = _load_optional_extension(
        state_renderer,
        kind="state renderer",
        required_methods=("render_state",),
    )
    loaded_output_renderer = _load_optional_extension(
        output_renderer,
        kind="output renderer",
        required_methods=("can_render", "render_chat_output"),
    )
    loaded_tool_renderer = _load_optional_extension(
        tool_renderer,
        kind="tool renderer",
        required_methods=("render_tools",),
    )
    loaded_state_mutator = _load_optional_extension(
        state_mutator,
        kind="state mutator",
        required_methods=("mutate_state",),
    )
    _run_app(
        graph,
        thread_id=thread_id,
        memory_renderer=loaded_memory_renderer,
        store_renderer=loaded_store_renderer,
        state_renderer=loaded_state_renderer,
        output_renderer=loaded_output_renderer,
        tool_renderer=loaded_tool_renderer,
        state_mutator=loaded_state_mutator,
        store_prefix=_parse_store_prefix(store_prefix),
        store_max_namespaces=max(1, store_max_namespaces),
        store_items_per_namespace=max(1, store_items_per_namespace),
    )


if __name__ == "__main__":
    main()
