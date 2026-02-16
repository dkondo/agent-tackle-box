# adb: Agent Debugger for LangChain/LangGraph

A TUI debugger that combines application-level agent inspection (state, memory, tool calls, messages) with Python-level debugging (breakpoints, stepping, variable inspection).

## Quick Start

```bash
# Install
uv pip install -e .

# Debug a LangGraph agent script
adb run my_agent.py

# Attach to a specific graph object
adb attach my_module:graph

# Attach with optional renderers/providers
adb attach my_module:graph \
  --memory-renderer my_mod:MemoryRenderer \
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
  --memory-renderer examples.simple_extensions:SimpleMemoryRenderer \
  --output-renderer examples.simple_extensions:SimpleChatOutputRenderer \
  --tool-renderer examples.simple_extensions:SimpleToolRenderer \
  --state-mutator examples.simple_extensions:SimpleStateMutator

# Optional: enable LiteLLM tool-calling path in examples/simple_agent.py
# (example model uses Vertex + service account/ADC auth)
USE_LITELLM=1 LITELLM_MODEL=vertex_ai/gemini-2.0-flash uv run adb run examples/simple_agent.py \
  --memory-renderer examples.simple_extensions:SimpleMemoryRenderer \
  --output-renderer examples.simple_extensions:SimpleChatOutputRenderer \
  --tool-renderer examples.simple_extensions:SimpleToolRenderer \
  --state-mutator examples.simple_extensions:SimpleStateMutator
```

## Features

- **Application-level debugging**: See agent state, messages, tool calls, state diffs
- **Code-level debugging**: Set breakpoints, step through code, inspect variables
- **Agent-level breakpoints**: Break on node start, tool call, or state change
- **Optional renderers/providers**: Custom state, store, memory, chat output, and state mutation hooks
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

## Debug Keys

When at a breakpoint, use pudb-style keys:

| Key | Action |
|-----|--------|
| `c` | Continue execution |
| `n` | Step over (next line) |
| `s` | Step into |
| `r` | Step out (return / finish) |

**Implementation note:** When a breakpoint hits, the Input widget is disabled (`inp.disabled = True`). This prevents it from consuming keystrokes, so `c`/`n`/`s`/`r` go to the App's `BINDINGS` instead. When the user presses `c` (continue), the Input is re-enabled and re-focused.
