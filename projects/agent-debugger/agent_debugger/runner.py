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
    ) -> None:
        self.graph = graph
        self.event_queue = event_queue
        self.command_queue = command_queue
        self.bp_manager = bp_manager
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
            args=(message,),
            daemon=True,
            name="adb-agent-worker",
        )
        self._thread.start()

    def _run_in_thread(self, message: str) -> None:
        """Run the agent synchronously in the worker thread.

        IMPORTANT: Uses graph.stream() (sync) instead of graph.astream()
        (async). sys.settrace is per-thread, and astream runs sync node
        functions in a thread pool -- a different thread than where the
        trace is installed. Using sync stream keeps node execution in
        THIS thread so bdb breakpoints work correctly.
        """
        self.tracer.reset_step()
        self._run_count += 1
        self._seen_tool_calls = set()
        self._seen_tool_results = set()
        self._seen_responses = set()
        self._last_state_signature = None
        self._last_state_for_breakpoints = {}

        input_data: dict[str, Any] = {
            "messages": [{"role": "human", "content": message}],
        }
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
                values = payload.get("values", {})
                next_nodes = payload.get("next", [])
                self.tracer.update_graph_state(values)
                self._emit_state_update(
                    values=values,
                    step=data.get("step", 0),
                    next_nodes=next_nodes,
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
        return config

    def _emit_state_update(
        self,
        values: dict[str, Any],
        *,
        step: int = 0,
        next_nodes: list[str] | None = None,
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
