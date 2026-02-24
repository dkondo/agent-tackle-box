"""AgentRunner: worker thread that runs the LangGraph graph."""

from __future__ import annotations

import json
import logging
import threading
from copy import deepcopy
from queue import Queue
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from agent_debugger.breakpoints import BreakpointManager
from agent_debugger.events import (
    AgentErrorEvent,
    AgentResponseEvent,
    RunFinishedEvent,
    StateUpdateEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from agent_debugger.message_utils import (
    content_to_text,
    message_content,
    message_type,
)
from agent_debugger.store_backend import snapshot_backend_store
from agent_debugger.tracer import AgentTracer

logger = logging.getLogger(__name__)


class AgentRunner:
    """Runs a LangGraph graph in a worker thread.

    Streams debug events to the UI via an event queue. Accepts
    commands from the UI via a command queue (for bdb control).
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        event_queue: Queue,
        command_queue: Queue,
        bp_manager: BreakpointManager,
        store_namespace_prefix: tuple[str, ...] | None = None,
        store_max_namespaces: int = 20,
        store_items_per_namespace: int = 20,
        input_provider: Any | None = None,
    ) -> None:
        self.graph = graph
        self.event_queue = event_queue
        self.command_queue = command_queue
        self.bp_manager = bp_manager
        self._input_provider = input_provider
        self.tracer = AgentTracer(
            event_queue=event_queue,
            command_queue=command_queue,
            bp_manager=bp_manager,
        )
        self._thread: threading.Thread | None = None
        self._config: dict[str, Any] = {}
        self._thread_id: str | None = None
        self._run_count: int = 0
        self._seen_tool_calls: set[str] = set()
        self._seen_tool_results: set[tuple[str, str, str]] = set()
        self._seen_responses: set[str] = set()
        self._last_state_signature: str | None = None
        self._last_state_for_breakpoints: dict[str, Any] = {}
        self._store_namespace_prefix = store_namespace_prefix
        self._store_max_namespaces = store_max_namespaces
        self._store_items_per_namespace = store_items_per_namespace
        self._replay_checkpoints: list[dict[str, Any]] = []
        self._replay_checkpoint_index: dict[str, int] = {}
        self._current_checkpoint_id: str | None = None
        self._cursor_config: dict[str, Any] | None = None
        self._pause_on_replay_start: bool = False

    def configure(
        self,
        thread_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Set runtime configuration."""
        self._thread_id = thread_id
        if config:
            self._config = config

    def invoke(self, message: str) -> None:
        """Start a new agent invocation in the worker thread."""
        self._thread = threading.Thread(
            target=self._run_in_thread,
            args=(message, False),
            daemon=True,
            name="adb-agent-worker",
        )
        self._thread.start()

    def invoke_replay(self) -> bool:
        """Start replay execution from the active replay cursor."""
        checkpoint_id = self.current_checkpoint_id
        if not checkpoint_id:
            self.event_queue.put(
                AgentErrorEvent(
                    message=(
                        "Replay cursor is not set. Run /rewind node <name> "
                        "or /forward node <name> first."
                    )
                )
            )
            self.event_queue.put(RunFinishedEvent())
            return False

        self._thread = threading.Thread(
            target=self._run_in_thread,
            args=(None, True),
            daemon=True,
            name="adb-agent-worker",
        )
        self._thread.start()
        return True

    def _run_in_thread(self, message: str | None, replay: bool) -> None:
        """Run the agent synchronously in the worker thread.

        IMPORTANT: Uses graph.stream() (sync) instead of graph.astream()
        (async). sys.settrace is per-thread, and astream runs sync node
        functions in a thread pool -- a different thread than where the
        trace is installed. Using sync stream keeps node execution in
        THIS thread so bdb breakpoints work correctly.
        """
        self.tracer.reset_step()
        self._run_count += 1
        # NOTE: _seen_tool_calls, _seen_tool_results, and _seen_responses
        # are intentionally NOT cleared between invocations.  The full
        # message history is re-scanned on each stream chunk, and resetting
        # these sets would cause historical tool calls / responses to be
        # re-emitted to the UI on every turn.
        self._last_state_signature = None
        self._last_state_for_breakpoints = {}

        if replay:
            input_data = None
            if self._pause_on_replay_start:
                self.tracer.request_break_on_next_user_frame()
                self._pause_on_replay_start = False
        elif self._input_provider is not None:
            input_data = self._input_provider.build_input(message or "")
        else:
            input_data = {"messages": [{"role": "human", "content": message or ""}]}
        config = self._build_runtime_config(include_callbacks=True)

        try:
            for chunk in self.graph.stream(
                input_data,
                config=config,
                stream_mode=["debug", "values", "updates"],
            ):
                self._process_stream_chunk(chunk)
        except Exception as e:
            logger.exception("Agent invocation failed")
            self.event_queue.put(AgentErrorEvent(message=str(e)))
        finally:
            self.event_queue.put(RunFinishedEvent())

    def _process_stream_chunk(self, chunk: tuple[str, Any]) -> None:
        """Process a single stream chunk and emit appropriate events."""
        mode, data = chunk

        if mode == "debug":
            event_type = data.get("type", "")
            payload = data.get("payload", {})

            if event_type == "checkpoint":
                values_raw = payload.get("values", {})
                values = values_raw if isinstance(values_raw, dict) else {}
                next_raw = payload.get("next", [])
                next_nodes = (
                    [str(node) for node in next_raw]
                    if isinstance(next_raw, (list, tuple))
                    else []
                )
                checkpoint_config = payload.get("config")
                checkpoint_id = self._extract_checkpoint_id(checkpoint_config)
                metadata = payload.get("metadata", {})
                stream_step_raw = data.get("step", 0)
                stream_step = stream_step_raw if isinstance(stream_step_raw, int) else 0
                checkpoint_step = None
                if isinstance(metadata, dict):
                    step = metadata.get("step")
                    if isinstance(step, int):
                        checkpoint_step = step
                if checkpoint_id and isinstance(checkpoint_config, dict):
                    self._register_checkpoint(
                        checkpoint_id=checkpoint_id,
                        checkpoint_config=checkpoint_config,
                        next_nodes=next_nodes,
                        step=checkpoint_step if checkpoint_step is not None else stream_step,
                        timestamp=data.get("timestamp"),
                    )
                    self.set_replay_cursor(checkpoint_config)
                self.tracer.update_graph_state(values)
                self._emit_state_update(
                    values=values,
                    step=stream_step,
                    next_nodes=next_nodes,
                    checkpoint_id=checkpoint_id,
                    checkpoint_config=(
                        checkpoint_config if isinstance(checkpoint_config, dict) else None
                    ),
                    checkpoint_step=checkpoint_step,
                )

                # Extract tool calls and results from messages
                self._extract_tool_info(values)

            elif event_type == "task":
                pass  # Handled by tracer callbacks

            elif event_type == "task_result":
                pass  # Handled by tracer callbacks

        elif mode == "values":
            if isinstance(data, dict):
                self.tracer.update_graph_state(data)
                self._emit_state_update(values=data, step=0)
                self._extract_tool_info(data)
                # Extract AI response from messages.
                # LangGraph may represent messages as dicts (with "type"
                # or "role" key) or as LangChain Message objects.
                messages = data.get("messages", [])
                if messages:
                    last = messages[-1]
                    msg_type = message_type(last)
                    if msg_type == "ai":
                        self._emit_agent_response(last)

        elif mode == "updates":
            # Extract AI response from the last message in each node
            # output.  Only checking the final message avoids re-emitting
            # historical responses when a node returns the full message
            # history (e.g. simple_agent.greeter).
            if isinstance(data, dict):
                for _, node_output in data.items():
                    if not isinstance(node_output, dict):
                        continue
                    msgs = node_output.get("messages", [])
                    if msgs:
                        last = msgs[-1]
                        if message_type(last) == "ai":
                            self._emit_agent_response(last)

    def _extract_tool_info(self, values: dict[str, Any]) -> None:
        """Extract tool calls and results from state messages.

        Emits ToolCallEvents from AIMessage.tool_calls and
        ToolResultEvents from ToolMessages. This captures tool
        usage regardless of whether the graph uses ToolNode or
        invokes tools manually inside a node function.
        """
        messages = values.get("messages", [])
        for msg in messages:
            msg_type = message_type(msg)

            # Extract tool calls from AI messages
            if msg_type == "ai":
                if isinstance(msg, dict):
                    tool_calls = msg.get("tool_calls", [])
                else:
                    tool_calls = getattr(msg, "tool_calls", []) or []

                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tc_id = str(tc.get("id", ""))
                        tc_name = str(tc.get("name", "unknown"))
                        tc_args = tc.get("args", {})
                    else:
                        tc_id = str(getattr(tc, "id", ""))
                        tc_name = str(getattr(tc, "name", "unknown"))
                        tc_args = getattr(tc, "args", {})

                    if not tc_id or tc_id in self._seen_tool_calls:
                        continue
                    self._seen_tool_calls.add(tc_id)
                    self.event_queue.put(
                        ToolCallEvent(
                            name=tc_name,
                            args=tc_args if isinstance(tc_args, dict) else {},
                            tool_call_id=tc_id,
                            node=self.tracer._current_node,
                            step=self.tracer._current_step,
                        )
                    )

            # Extract tool results from tool messages
            elif msg_type == "tool":
                if isinstance(msg, dict):
                    tool_call_id = str(msg.get("tool_call_id", ""))
                    result = msg.get("content")
                    error = msg.get("error")
                else:
                    tool_call_id = str(getattr(msg, "tool_call_id", ""))
                    result = getattr(msg, "content", None)
                    error = getattr(msg, "error", None)

                key = (
                    tool_call_id,
                    str(result),
                    str(error),
                )
                if key in self._seen_tool_results:
                    continue
                self._seen_tool_results.add(key)
                self.event_queue.put(
                    ToolResultEvent(
                        tool_call_id=tool_call_id,
                        result=result,
                        error=str(error) if error is not None else None,
                    )
                )

    @property
    def is_running(self) -> bool:
        """Whether the agent is currently running."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def current_checkpoint_id(self) -> str | None:
        """Return the active replay checkpoint id."""
        return self._current_checkpoint_id

    def arm_pause_on_replay_start(self) -> None:
        """Pause at the next user frame when replay execution starts."""
        self._pause_on_replay_start = True

    def supports_replay(self) -> tuple[bool, str]:
        """Whether graph replay is available for the current runner."""
        checkpointer = getattr(self.graph, "checkpointer", None)
        if checkpointer is None:
            return False, "Replay requires a graph checkpointer."
        if not self._thread_id:
            return False, "Replay requires --thread-id."
        if not hasattr(self.graph, "get_state_history") or not hasattr(self.graph, "get_state"):
            return False, "Graph does not support state history APIs."
        return True, "ok"

    def set_replay_cursor(self, checkpoint_config: dict[str, Any]) -> tuple[bool, str]:
        """Set the replay cursor config and active checkpoint id."""
        configurable = checkpoint_config.get("configurable")
        if not isinstance(configurable, dict):
            return False, "Checkpoint config is missing 'configurable'."

        checkpoint_id = configurable.get("checkpoint_id")
        if checkpoint_id is None:
            return False, "Checkpoint config is missing checkpoint_id."

        normalized_config = dict(checkpoint_config)
        normalized_config["configurable"] = dict(configurable)
        self._cursor_config = normalized_config
        self._current_checkpoint_id = str(checkpoint_id)
        return True, "Replay cursor updated."

    def seek_backward_to_node(
        self,
        node_name: str,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """Move replay cursor to the previous checkpoint targeting a node."""
        return self._seek_to_node(node_name=node_name, direction="backward")

    def seek_forward_to_node(
        self,
        node_name: str,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """Move replay cursor to the next checkpoint targeting a node."""
        return self._seek_to_node(node_name=node_name, direction="forward")

    def seek_nearest_to_node(
        self,
        node_name: str,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """Move replay cursor to the nearest checkpoint targeting a node."""
        return self._seek_to_node(node_name=node_name, direction="nearest")

    def _build_runtime_config(self, *, include_callbacks: bool) -> dict[str, Any]:
        """Build runtime config from configured options."""
        config: dict[str, Any] = {}
        if include_callbacks:
            config["callbacks"] = [self.tracer]
        if self._config:
            config.update(self._config)
        if self._thread_id:
            config.setdefault("configurable", {})
            config["configurable"]["thread_id"] = self._thread_id
        if self._cursor_config:
            cursor_cfg = self._cursor_config.get("configurable")
            if isinstance(cursor_cfg, dict):
                config.setdefault("configurable", {})
                config["configurable"].update(cursor_cfg)
        return config

    def _seek_to_node(
        self,
        *,
        node_name: str,
        direction: str,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """Seek the replay cursor to a checkpoint that schedules a node."""
        supported, reason = self.supports_replay()
        if not supported:
            return False, reason, None

        try:
            history = self._replay_history()
        except Exception as e:
            logger.exception("Failed to load checkpoint history")
            return False, f"Failed to load checkpoint history: {e}", None
        if not history:
            return False, "No checkpoint history is available for this thread.", None

        current_idx = self._current_history_index(history)
        target_idx = self._find_target_history_index(
            history=history,
            current_idx=current_idx,
            node_name=node_name,
            direction=direction,
        )
        if target_idx is None:
            return False, f"No {direction} checkpoint found for node '{node_name}'.", None

        snapshot = history[target_idx]
        checkpoint_config = getattr(snapshot, "config", None)
        if not isinstance(checkpoint_config, dict):
            return False, "Target checkpoint is missing config metadata.", None

        ok, cursor_msg = self.set_replay_cursor(checkpoint_config)
        if not ok:
            return False, cursor_msg, None

        payload = self._snapshot_to_payload(snapshot)
        target_id = payload.get("checkpoint_id", "unknown")
        if target_idx == current_idx:
            msg = f"Already at node '{node_name}' checkpoint ({target_id})."
        else:
            verb = "rewound" if target_idx < current_idx else "forwarded"
            msg = f"Replay cursor {verb} to node '{node_name}' ({target_id})."
        payload["target_index"] = target_idx
        payload["current_index"] = current_idx
        return True, msg, payload

    def _replay_history(self) -> list[Any]:
        """Return thread checkpoint history in chronological order."""
        base_config = self._build_runtime_config(include_callbacks=False)
        configurable = base_config.get("configurable")
        history_config: dict[str, Any] = {}
        if isinstance(configurable, dict):
            cfg = dict(configurable)
            cfg.pop("checkpoint_id", None)
            if cfg:
                history_config["configurable"] = cfg

        snapshots = list(self.graph.get_state_history(history_config, limit=5000))
        snapshots.reverse()
        return snapshots

    def _current_history_index(self, history: list[Any]) -> int:
        """Resolve active cursor index in chronological history."""
        id_to_index = {
            self._extract_checkpoint_id(getattr(snapshot, "config", None)): idx
            for idx, snapshot in enumerate(history)
        }
        checkpoint_id = self._current_checkpoint_id
        if checkpoint_id and checkpoint_id in id_to_index:
            return id_to_index[checkpoint_id]
        return max(0, len(history) - 1)

    def _find_target_history_index(
        self,
        *,
        history: list[Any],
        current_idx: int,
        node_name: str,
        direction: str,
    ) -> int | None:
        """Find target checkpoint index for a node search direction."""
        if direction == "backward":
            for idx in range(current_idx - 1, -1, -1):
                if node_name in self._snapshot_next_nodes(history[idx]):
                    return idx
            return None

        if direction == "forward":
            for idx in range(current_idx + 1, len(history)):
                if node_name in self._snapshot_next_nodes(history[idx]):
                    return idx
            return None

        if direction == "nearest":
            matches: list[tuple[int, int, int]] = []
            for idx, snapshot in enumerate(history):
                if node_name not in self._snapshot_next_nodes(snapshot):
                    continue
                distance = abs(idx - current_idx)
                if idx > current_idx:
                    tie_priority = 0
                elif idx < current_idx:
                    tie_priority = 1
                else:
                    tie_priority = -1
                matches.append((distance, tie_priority, idx))
            if not matches:
                return None
            matches.sort()
            return matches[0][2]

        return None

    @staticmethod
    def _snapshot_next_nodes(snapshot: Any) -> list[str]:
        """Normalize a StateSnapshot.next value to list[str]."""
        next_nodes = getattr(snapshot, "next", ())
        if isinstance(next_nodes, (list, tuple)):
            return [str(node) for node in next_nodes]
        return []

    def _snapshot_to_payload(self, snapshot: Any) -> dict[str, Any]:
        """Convert a checkpoint snapshot to UI-friendly replay payload."""
        values = getattr(snapshot, "values", {})
        if not isinstance(values, dict):
            values = {}
        checkpoint_config = getattr(snapshot, "config", None)
        checkpoint_id = self._extract_checkpoint_id(checkpoint_config)
        metadata = getattr(snapshot, "metadata", None)
        checkpoint_step = None
        if isinstance(metadata, dict):
            step = metadata.get("step")
            if isinstance(step, int):
                checkpoint_step = step

        return {
            "values": values,
            "next_nodes": self._snapshot_next_nodes(snapshot),
            "checkpoint_id": checkpoint_id,
            "checkpoint_config": checkpoint_config if isinstance(checkpoint_config, dict) else None,
            "checkpoint_step": checkpoint_step,
        }

    @staticmethod
    def _extract_checkpoint_id(checkpoint_config: dict[str, Any] | None) -> str | None:
        """Extract checkpoint_id from config payload."""
        if not isinstance(checkpoint_config, dict):
            return None
        configurable = checkpoint_config.get("configurable")
        if not isinstance(configurable, dict):
            return None
        checkpoint_id = configurable.get("checkpoint_id")
        if checkpoint_id is None:
            return None
        return str(checkpoint_id)

    def _register_checkpoint(
        self,
        *,
        checkpoint_id: str,
        checkpoint_config: dict[str, Any],
        next_nodes: list[str],
        step: int,
        timestamp: str | None,
    ) -> None:
        """Record a checkpoint seen during stream processing."""
        entry = {
            "checkpoint_id": checkpoint_id,
            "checkpoint_config": dict(checkpoint_config),
            "next_nodes": list(next_nodes),
            "step": step,
            "timestamp": timestamp,
        }
        existing = self._replay_checkpoint_index.get(checkpoint_id)
        if existing is None:
            self._replay_checkpoint_index[checkpoint_id] = len(self._replay_checkpoints)
            self._replay_checkpoints.append(entry)
            return
        self._replay_checkpoints[existing] = entry

    def _emit_state_update(
        self,
        values: dict[str, Any],
        *,
        step: int = 0,
        next_nodes: list[str] | None = None,
        checkpoint_id: str | None = None,
        checkpoint_config: dict[str, Any] | None = None,
        checkpoint_step: int | None = None,
    ) -> None:
        """Emit StateUpdateEvent only when state changed."""
        if self.bp_manager.should_break_on_state(self._last_state_for_breakpoints, values):
            self.tracer.request_break_on_next_user_frame()
        self._last_state_for_breakpoints = self._safe_state_copy(values)

        store_items, store_source, store_error = self.get_store_snapshot()

        try:
            signature = json.dumps(
                {
                    "values": values,
                    "store_items": store_items,
                    "store_source": store_source,
                    "store_error": store_error,
                    "step": step,
                    "next_nodes": next_nodes or [],
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_step": checkpoint_step,
                },
                sort_keys=True,
                default=str,
            )
        except Exception:
            signature = str((values, store_items, store_source, store_error))

        if signature == self._last_state_signature:
            return
        self._last_state_signature = signature

        self.event_queue.put(
            StateUpdateEvent(
                values=values,
                store_items=store_items,
                store_source=store_source,
                store_error=store_error,
                step=step,
                next_nodes=next_nodes or [],
                checkpoint_id=checkpoint_id,
                checkpoint_config=checkpoint_config,
                checkpoint_step=checkpoint_step,
            )
        )

    def _safe_state_copy(self, values: dict[str, Any]) -> dict[str, Any]:
        """Best-effort copy for state breakpoint comparisons."""
        try:
            return deepcopy(values)
        except Exception:
            return dict(values)

    def get_store_snapshot(
        self,
    ) -> tuple[dict[str, dict[str, Any]], str, str | None]:
        """Read a normalized snapshot from the graph backend store."""
        return snapshot_backend_store(
            self.graph,
            namespace_prefix=self._store_namespace_prefix,
            max_namespaces=self._store_max_namespaces,
            max_items_per_namespace=self._store_items_per_namespace,
        )

    def _emit_agent_response(self, msg: Any) -> None:
        """Emit a response event from an AI message-like payload."""
        normalized_type = message_type(msg)
        if isinstance(msg, dict):
            content = message_content(msg)
            payload = dict(msg)
            payload["type"] = normalized_type
        else:
            content = message_content(msg)
            payload = {
                "content": content,
                "type": normalized_type,
                "role": getattr(msg, "role", ""),
            }
            for attr in (
                "additional_kwargs",
                "response_metadata",
                "tool_calls",
                "name",
            ):
                if hasattr(msg, attr):
                    payload[attr] = getattr(msg, attr)

        text = content_to_text(content)
        try:
            sig = json.dumps(payload, sort_keys=True, default=str)
        except Exception:
            sig = repr((text, payload))
        if sig in self._seen_responses:
            return
        self._seen_responses.add(sig)

        self.event_queue.put(
            AgentResponseEvent(
                text=text,
                payload=payload,
            )
        )

    def update_graph_state(
        self, values: dict[str, Any], as_node: str | None = None
    ) -> tuple[bool, str]:
        """Update persisted graph state when the graph supports it."""
        if not hasattr(self.graph, "update_state"):
            return False, "Graph does not support update_state()."

        try:
            config = self._build_runtime_config(include_callbacks=False)
            self.graph.update_state(config=config, values=values, as_node=as_node)
        except Exception as e:
            return False, str(e)
        return True, "Graph state updated."
