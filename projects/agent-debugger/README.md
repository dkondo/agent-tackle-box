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
  --state-mutator my_mod:StateMutator
```

## Run from Source

```bash
# Create/update local env from this repo
uv sync --dev

# Run adb directly from source (project root)
uv run adb run examples/simple_agent.py

# Equivalent module invocation
uv run python -m adb.cli run examples/simple_agent.py
```

## Features

- **Application-level debugging**: See agent state, messages, tool calls, state diffs
- **Code-level debugging**: Set breakpoints, step through code, inspect variables
- **Agent-level breakpoints**: Break on node start, tool call, or state change
- **Optional renderers/providers**: Custom state, store, memory, chat output, and state mutation hooks
- **`import adb; adb.set_trace()`**: Drop into the debugger from anywhere in your agent code

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
