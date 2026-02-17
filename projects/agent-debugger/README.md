# adb: Agent Debugger

A terminal UI debugger that combines application-level agent inspection (state, memory, tool calls, messages) with Python-level debugging (breakpoints, stepping, variable inspection).

<img width="1679" height="1016" alt="image" src="https://github.com/user-attachments/assets/85f77ac9-45fc-4b66-ade9-bf79d148aa56" />

## Motivation

An agent developer needs to answer two kinds of questions simultaneously:

1. **Application-level**: "How did state or memory change? What tools were called and how?"
    
2. **Code-level**: "Why did this node produce that output? What's in the local variables at line 42? Why did the conditional branch go left?"
    
When you stop at a breakpoint inside a node, you want to see _both_ the Python locals _and_ the graph-level context (which node, what triggered it, the full agent state). 

The combined debugger `adb` makes "zoom in / zoom out" seamless

Also, `adb` enables **generative debugging** by providing interfaces and hooks for custom rendering of state, store, and chat output, and custom specification of inputs and state mutation.

The debugger is agent agnostic.  The engineering design is based agent's event stream, and it extends Python’s standard-library debugger class (`Bdb`) .

The tool is released on pypi as `agent-debugger`.

## Quick Start

```bash
# Install
uv add agent-debugger

# Debug a LangGraph agent script
adb run my_agent.py

# Attach to a specific graph object
adb attach my_module:graph

# Attach with optional renderers/providers
adb attach my_module:graph \
  --store-renderer my_mod:StoreRenderer \
  --state-renderer my_mod:StateRenderer \
  --output-renderer my_mod:ChatOutputRenderer \
  --tool-renderer my_mod:ToolRenderer \
  --state-mutator my_mod:StateMutator
```

## Run from Source

```bash
# Create/update local env from this repo
uv sync --dev

# Run adb directly from source (project root)
uv run adb run examples/simple_agent.py

# Equivalent module invocation
uv run python -m agent_debugger.cli run examples/simple_agent.py
```

## Simple Agent Demo

```bash
# Run simple_agent with all demo renderer/mutator extensions
uv run adb run examples/simple_agent.py \
  --store-renderer examples.simple_extensions:SimpleStoreRenderer \
  --output-renderer examples.simple_extensions:SimpleChatOutputRenderer \
  --tool-renderer examples.simple_extensions:SimpleToolRenderer \
  --state-mutator examples.simple_extensions:SimpleStateMutator

# Optional: enable LiteLLM tool-calling path in examples/simple_agent.py
# (example model uses Vertex + service account/ADC auth)
USE_LITELLM=1 LITELLM_MODEL=vertex_ai/gemini-2.0-flash uv run adb run examples/simple_agent.py \
  --store-renderer examples.simple_extensions:SimpleStoreRenderer \
  --output-renderer examples.simple_extensions:SimpleChatOutputRenderer \
  --tool-renderer examples.simple_extensions:SimpleToolRenderer \
  --state-mutator examples.simple_extensions:SimpleStateMutator
```

## Features

- **Application-level debugging**: See agent state, messages, tool calls, state diffs
- **Code-level debugging**: Set breakpoints, step through code, inspect variables
- **Agent-level breakpoints**: Break on node start, tool call, or state change
- **Optional renderers/providers**: Custom state, store, tools, chat output, and state mutation hooks
- **Persistent tool history**: Tool calls are kept across turns in the Tools pane and grouped by turn
- **`import agent_debugger as adb; adb.set_trace()`**: Drop into the debugger from anywhere in your agent code

## Usage

```bash
# Set a breakpoint on a node
/break node agent

# Set a breakpoint on a tool
/break tool search_listings

# Break when a state key changes
/break state messages

# Standard Python breakpoint
/break line my_agent.py:42

# Clear local UI context
/clear

# Local clear + optional mutator mutation
/clear memory
```

See `/help` in the TUI for all commands.

## Programmatic Breakpoints

You can drop into the adb debugger from anywhere in your agent code using Python's built-in `breakpoint()`:

```bash
PYTHONBREAKPOINT="adb.set_trace" adb run my_agent.py
```

Then add `breakpoint()` calls in your code:

```python
def my_node(state: dict) -> dict:
    messages = state.get("messages", [])
    breakpoint()  # execution pauses here in the adb UI
    return {"messages": [...]}
```

Or call `adb.set_trace()` directly:

```python
def my_node(state: dict) -> dict:
    import agent_debugger as adb; adb.set_trace()
    return {"messages": [...]}
```

## Debug Keys

When at a breakpoint, use pudb-style keys:

| Key | Action |
|-----|--------|
| `c` | Continue execution |
| `n` | Step over (next line) |
| `s` | Step into |
| `r` | Step out (return / finish) |

**Implementation note:** When a breakpoint hits, the Input widget is disabled (`inp.disabled = True`). This prevents it from consuming keystrokes, so `c`/`n`/`s`/`r` go to the App's `BINDINGS` instead. When the user presses `c` (continue), the Input is re-enabled and re-focused.

## Extensions

adb supports optional extensions that customize how your agent's data is displayed
and how input is constructed. Extensions are plain Python classes — no base class
required. adb uses duck typing, so any object with the right methods will work.

Pass extensions via CLI flags using `module:Class` references:

```bash
adb attach my_module:graph \
  --store-renderer my_ext:MyStoreRenderer \
  --state-renderer my_ext:MyStateRenderer \
  --output-renderer my_ext:MyChatOutputRenderer \
  --tool-renderer my_ext:MyToolRenderer \
  --state-mutator my_ext:MyStateMutator \
  --input-provider my_ext:MyInputProvider
```

If a class reference is provided, adb will instantiate it (call `Class()`) and
validate that the required methods exist. If loading fails, adb logs a warning
and falls back to default behavior.

### StoreRenderer

Customizes the **Store** panel in the right sidebar.

| Method | Signature | Description |
|--------|-----------|-------------|
| `render_store` | `(snapshot: Mapping[str, Any]) -> MemoryRenderModel \| None` | Render store contents. `snapshot` contains `{"store_items": {...}}`. |

```python
from agent_debugger.extensions import MemoryRenderModel

class MyStoreRenderer:
    def render_store(self, snapshot):
        items = snapshot.get("store_items", {})
        lines = ["[bold cyan]My Store[/bold cyan]"]
        for ns, entries in items.items():
            lines.append(f"  {ns}: {len(entries)} items")
        return MemoryRenderModel(lines=lines)
```

### StateRenderer

Customizes the **State** panel in the right sidebar.

| Method | Signature | Description |
|--------|-----------|-------------|
| `render_state` | `(snapshot: Mapping[str, Any]) -> StateRenderModel \| None` | Render state contents. `snapshot` contains `{"state": {...}}`. |

```python
from agent_debugger.extensions import StateRenderModel

class MyStateRenderer:
    def render_state(self, snapshot):
        state = snapshot.get("state", {})
        lines = [
            "[bold cyan]Agent State[/bold cyan]",
            f"messages: {len(state.get('messages', []))}",
            f"status: {state.get('status', 'unknown')}",
        ]
        return StateRenderModel(lines=lines)
```

### ChatOutputRenderer

Customizes how agent responses appear in the **main chat pane**. This is useful
when your agent returns structured output (e.g., JSON with a `text` field and
metadata) and you want to display only the relevant parts.

| Method | Signature | Description |
|--------|-----------|-------------|
| `can_render` | `(payload: Mapping[str, Any]) -> bool` | Return `True` if this renderer handles the given payload. |
| `render_chat_output` | `(payload, state, messages) -> ChatRenderModel \| None` | Render the payload into display lines. |

The `payload` dict contains the raw agent response (e.g., `additional_kwargs.parsed`
from the AI message, or parsed JSON content). Lines support
[Rich markup](https://rich.readthedocs.io/en/latest/markup.html) for styling.

```python
from agent_debugger.extensions import ChatRenderModel

class MyChatOutputRenderer:
    def can_render(self, payload):
        return "text" in payload

    def render_chat_output(self, payload, state, messages):
        text = payload.get("text", "")
        lines = [text]
        recs = payload.get("recommendations", [])
        if recs:
            lines.append("")
            lines.append("[bold]Recommendations:[/bold]")
            for i, rec in enumerate(recs, 1):
                title = rec.get("title", "")
                lines.append(f"  {i}. [cyan]{title}[/cyan]")
        return ChatRenderModel(lines=lines)
```

### ToolRenderer

Customizes the **Tools** tab in the bottom panel.

| Method | Signature | Description |
|--------|-----------|-------------|
| `render_tools` | `(snapshot: Mapping[str, Any]) -> ToolRenderModel \| None` | Render tool call history. `snapshot` contains `{"tool_calls": [...]}`. |

Each tool call dict has keys: `name`, `args`, `result`, `error`, `node`, `turn`.

```python
from agent_debugger.extensions import ToolRenderModel

class MyToolRenderer:
    def render_tools(self, snapshot):
        calls = snapshot.get("tool_calls", [])
        lines = [f"[bold]Tools ({len(calls)} calls)[/bold]"]
        for call in calls:
            name = call.get("name", "?")
            status = "[red]error[/red]" if call.get("error") else "[green]ok[/green]"
            lines.append(f"  {name} — {status}")
        return ToolRenderModel(lines=lines)
```

### StateMutator

Handles custom `/clear <mutation>` commands (e.g., `/clear memory`).

| Method | Signature | Description |
|--------|-----------|-------------|
| `mutate_state` | `(mutation, args, current_state, runner) -> StateMutationResult \| None` | Apply a named mutation. `runner` provides access to the graph and store. |

```python
from agent_debugger.extensions import StateMutationResult

class MyStateMutator:
    def mutate_state(self, mutation, args, current_state, runner):
        if mutation != "memory":
            return StateMutationResult(applied=False, message="Unknown mutation.")
        store = getattr(getattr(runner, "graph", None), "store", None)
        if store is None:
            return StateMutationResult(applied=False, message="No store configured.")
        # ... clear store entries ...
        return StateMutationResult(applied=True, message="Memory cleared.")
```

### InputProvider

Customizes the input dict sent to `graph.stream()`. Use this when your agent's
state schema requires fields beyond `messages` (e.g., `user_id`).

| Method | Signature | Description |
|--------|-----------|-------------|
| `build_input` | `(message: str) -> dict[str, Any]` | Build the input dict. Must include a `"messages"` key. |

```python
import getpass

class MyInputProvider:
    def build_input(self, message):
        return {
            "messages": [{"role": "human", "content": message}],
            "user_id": getpass.getuser(),
        }
```

### Render model line format

All render models (`MemoryRenderModel`, `StateRenderModel`, `ChatRenderModel`,
`ToolRenderModel`) contain a `lines: list[str]` field. Each string is rendered
as a separate line in the UI and supports
[Rich console markup](https://rich.readthedocs.io/en/latest/markup.html):

```
[bold cyan]Title[/bold cyan]       # bold cyan text
[dim]subtitle[/dim]               # dimmed text
[green]success[/green]            # green text
[red]error[/red]                  # red text
```

If markup parsing fails for a line, adb falls back to rendering it as plain text.

### Full example

See [`examples/simple_extensions.py`](examples/simple_extensions.py) for a
working implementation of `StoreRenderer`, `ChatOutputRenderer`, `ToolRenderer`,
and `StateMutator`.

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--env-file` | `.env` | Path to a dotenv file to load before starting. |
| `--thread-id`, `-t` | — | Thread ID for checkpointed graphs. |
| `--store-renderer` | — | Store panel renderer (`module:Class`). |
| `--state-renderer` | — | State panel renderer (`module:Class`). |
| `--output-renderer` | — | Chat output renderer (`module:Class`). |
| `--tool-renderer` | — | Tools panel renderer (`module:Class`). |
| `--state-mutator` | — | State mutator for `/clear` commands (`module:Class`). |
| `--input-provider` | — | Input provider for custom graph input (`module:Class`). |
| `--store-prefix` | — | Backend store namespace prefix (comma-separated). |
| `--store-max-namespaces` | `20` | Max store namespaces to display. |
| `--store-items-per-namespace` | `20` | Max items per namespace to display. |

## Design

See [`Design.md`](docs/Design.md).
