# adb Design

## Purpose
`adb` is a terminal debugger for LangGraph/LangChain agents that combines two debugging layers in one UI:
- Agent-level execution: nodes, tool calls, message/state evolution, and store snapshots.
- Python-level execution: line breakpoints, stepping, stack/locals inspection via `bdb`.

The design goal is to let developers move between "what did the agent do?" and "why did this line do that?" without switching tools.

## Design Principles
1. Keep Python tracing and TUI rendering from blocking each other.
2. Preserve an agent-agnostic core that works for arbitrary graphs/state shapes.
3. Treat extension points (renderers/mutators) as optional and fail-safe.
4. Keep state/tool/message visibility first-class even when code stepping is active.

## High-Level Architecture
`adb` is split into a UI thread and a worker thread:

- UI thread (`DebuggerApp` in `agent_debugger/app.py`): Textual TUI, command handling, panel updates, keybindings.
- Worker thread (`AgentRunner` in `agent_debugger/runner.py`): runs `graph.stream(...)`, owns `AgentTracer`, emits runtime events.

Two queues bridge the threads:
- `event_queue`: worker -> UI (`NodeStartEvent`, `StateUpdateEvent`, `BreakpointHit`, etc.).
- `command_queue`: UI -> worker (`DebugCommand.CONTINUE/STEP_* /QUIT`).

This avoids event-loop conflicts: Textual keeps rendering while the worker can pause on breakpoints.

## Core Runtime Components
- `AgentTracer` (`agent_debugger/tracer.py`): inherits both `bdb.Bdb` and LangChain `BaseCallbackHandler`.
  - `bdb.Bdb` is Python’s standard-library debugger base class; it provides tracing hooks (`user_line`, `user_return`, `user_exception`), line breakpoint support, and step/continue/return control primitives.
  - Receives graph lifecycle callbacks (`on_chain_start/end`, `on_tool_start`).
  - Activates Python tracing only when needed (semantic breakpoints or line breakpoints).
  - Emits `BreakpointHit` events and blocks for UI commands when paused.
- `BreakpointManager` (`agent_debugger/breakpoints.py`):
  - Supports `line`, `node`, `tool`, `state`, and `transition` breakpoints.
  - Tracks enabled state and hit counts.
- `AgentRunner` (`agent_debugger/runner.py`):
  - Executes `graph.stream` with `stream_mode=["debug", "values", "updates"]`.
  - Normalizes stream chunks into UI events.
  - Extracts tool calls/results from messages, deduplicates events, emits agent responses.
  - Captures backend store snapshots via `snapshot_backend_store`.

## UI Composition
`DebuggerApp` (`agent_debugger/app.py`) composes:
- Left pane: chat log + input.
- Right pane: `Store`, `State`, `Variables`, `Stack` (collapsible).
- Bottom tabs: `Messages`, `Tools`, `Source`, `Diff`, `Breakpoints`, `Logs`.

Textual was chosen because it provides a mature, async-capable TUI framework with a strong widget/layout system (tabs, logs, collapsibles, keybindings, timers) that maps directly to this debugger’s multi-panel UX. In practice, Textual makes it straightforward to keep a responsive UI loop on the main thread while the agent/debugger runtime blocks in a separate worker thread at breakpoints.

Behavior highlights:
- Polls `event_queue` every 50ms.
- On `BreakpointHit`: disables input, focuses source/vars/stack, and waits for `c/n/s/r`.
- On continue: re-enables input, collapses debug panels, resumes spinner.
- Maintains per-turn tool history and command history (`.adb/history.json`).

## Extension and Customization Model
The debugger is intentionally generic, but supports optional protocols in `agent_debugger/extensions.py`:
- `StoreRenderer` and `MemoryRenderer` for Store panel rendering.
- `StateRenderer` for State panel rendering.
- `ToolRenderer` for Tools panel rendering.
- `ChatOutputRenderer` for response formatting in chat.
- `StateMutator` for `/clear <mutation>` behaviors against persisted graph state.

CLI (`agent_debugger/cli.py`) loads extensions from `module:attribute`, validates required methods, and falls back safely on errors.

## Store and State Design
Store data is backend-first and separate from graph state:
- `snapshot_backend_store` (`agent_debugger/store_backend.py`) reads `graph.store` using modern BaseStore APIs (`list_namespaces` + `search`) and legacy shapes when needed.
- Store panel never infers backend data from state when no backend snapshot exists.
- `StateUpdateEvent` carries both state and store metadata (`store_items`, `store_source`, `store_error`).

## End-to-End Flow
1. User enters text in the TUI.
2. UI invokes `AgentRunner` on worker thread.
3. Runner streams graph events, tracer emits callback-driven events.
4. UI updates state/messages/tools/diff/store in near-real time.
5. If a breakpoint triggers, worker blocks; UI drives stepping via command queue.
6. On `RunFinishedEvent`, UI clears processing state and returns to interactive input.

## Why It Works
The key design decision is pairing a LangGraph callback handler with `bdb` in one tracer while running agent execution in a dedicated worker thread. That preserves responsive UI rendering, semantic agent inspection, and precise line-level debugging in a single terminal workflow.
