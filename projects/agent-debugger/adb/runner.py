"""AgentRunner: worker thread that runs the LangGraph graph."""

from __future__ import annotations

import json
import logging
import threading
from queue import Queue
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from adb.breakpoints import BreakpointManager
from adb.events import (
    AgentErrorEvent,
    AgentResponseEvent,
    DebugCommand,
    RunFinishedEvent,
    StateUpdateEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from adb.tracer import AgentTracer

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
        self._seen_tool_results: set[tuple[str, str, str]] = set()
        self._last_state_signature: str | None = None

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
        self._seen_tool_results = set()
        self._last_state_signature = None

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
                    if isinstance(last, dict):
                        content = last.get("content", "")
                        msg_type = (
                            last.get("type", "")
                            or last.get("role", "")
                        )
                    else:
                        content = getattr(last, "content", "")
                        msg_type = getattr(last, "type", "")
                    if msg_type == "ai":
                        self._emit_agent_response(last)

        elif mode == "updates":
            # Extract AI response from node output updates too.
            if isinstance(data, dict):
                for node_name, node_output in data.items():
                    if not isinstance(node_output, dict):
                        continue
                    for msg in node_output.get("messages", []):
                        if isinstance(msg, dict):
                            content = msg.get("content", "")
                            msg_type = (
                                msg.get("type", "")
                                or msg.get("role", "")
                            )
                        else:
                            content = getattr(msg, "content", "")
                            msg_type = getattr(msg, "type", "")
                        if msg_type == "ai":
                            self._emit_agent_response(msg)

    def _extract_tool_info(self, values: dict[str, Any]) -> None:
        """Extract tool calls and results from state messages."""
        messages = values.get("messages", [])
        for msg in messages:
            if isinstance(msg, dict):
                msg_type = msg.get("type", "")
                if msg_type == "tool":
                    tool_call_id = msg.get("tool_call_id", "")
                    result = msg.get("content")
                    error = msg.get("error")
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
                            error=error,
                        )
                    )
            else:
                msg_type = getattr(msg, "type", "")
                if msg_type == "tool":
                    tool_call_id = getattr(msg, "tool_call_id", "")
                    result = getattr(msg, "content", None)
                    key = (
                        tool_call_id,
                        str(result),
                        "",
                    )
                    if key in self._seen_tool_results:
                        continue
                    self._seen_tool_results.add(key)
                    self.event_queue.put(
                        ToolResultEvent(
                            tool_call_id=tool_call_id,
                            result=result,
                        )
                    )

    @property
    def is_running(self) -> bool:
        """Whether the agent is currently running."""
        return self._thread is not None and self._thread.is_alive()

    def _build_runtime_config(
        self, *, include_callbacks: bool
    ) -> dict[str, Any]:
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
        try:
            signature = json.dumps(values, sort_keys=True, default=str)
        except Exception:
            signature = str(values)

        if signature == self._last_state_signature:
            return
        self._last_state_signature = signature

        self.event_queue.put(
            StateUpdateEvent(
                values=values,
                step=step,
                next_nodes=next_nodes or [],
            )
        )

    def _emit_agent_response(self, msg: Any) -> None:
        """Emit a response event from an AI message-like payload."""
        if isinstance(msg, dict):
            content = msg.get("content", "")
            payload = dict(msg)
            payload.setdefault("type", msg.get("type") or msg.get("role"))
        else:
            content = getattr(msg, "content", "")
            payload = {
                "content": content,
                "type": getattr(msg, "type", ""),
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

        self.event_queue.put(
            AgentResponseEvent(
                text=self._message_content_to_text(content),
                payload=payload,
            )
        )

    def _message_content_to_text(self, content: Any) -> str:
        """Normalize message content into a readable text string."""
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text is not None:
                        parts.append(str(text))
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return " ".join(parts).strip()
        return str(content)

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
