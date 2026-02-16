"""AgentTracer: bdb.Bdb subclass with LangGraph callback integration."""

from __future__ import annotations

import bdb
import linecache
import sys
import threading
from queue import Queue
from types import FrameType
from typing import Any, ClassVar

from langchain_core.callbacks import BaseCallbackHandler

from agent_debugger.breakpoints import BreakpointManager
from agent_debugger.events import (
    BreakpointHit,
    DebugCommand,
    NodeEndEvent,
    NodeStartEvent,
    ToolCallEvent,
)


class AgentTracer(bdb.Bdb, BaseCallbackHandler):
    """bdb-based tracer that is LangGraph-aware.

    Combines Python-level tracing (breakpoints, stepping) with
    LangGraph callback handling (node start/end, tool calls).

    Runs in the worker thread alongside the LangGraph graph.
    Communicates with the UI via thread-safe queues.
    """

    _current: ClassVar[AgentTracer | None] = None

    # Paths to skip in user_line (don't pause inside these).
    # We do NOT pass these to bdb's skip= parameter because bdb's
    # dispatch_call returns None for skipped modules, which prevents
    # tracing from propagating into child calls (including user code).
    _SKIP_PATHS = (
        # LangGraph / LangChain
        "langgraph/",
        "langchain_core/",
        "langchain/",
        # UI frameworks
        "textual/",
        "rich/",
        # debugger package itself (support new and legacy package paths)
        "/agent-debugger/agent_debugger/",
        "agent_debugger/tracer.py",
        "agent_debugger/runner.py",
        "agent_debugger/app.py",
        "agent_debugger/events.py",
        "agent_debugger/breakpoints.py",
        "agent_debugger/panels/",
        "agent_debugger/cli.py",
        "/adb/adb/",
        "adb/tracer.py",
        "adb/runner.py",
        "adb/app.py",
        "adb/events.py",
        "adb/breakpoints.py",
        "adb/panels/",
        "adb/cli.py",
        # Python stdlib
        "asyncio/",
        "concurrent/",
        "threading",
        "importlib/",
        "contextlib",
        "functools",
        "inspect",
        "typing",
        "collections/",
        "enum",
        "abc",
        "copyreg",
        "weakref",
        # Installed packages
        "site-packages/",
        # Python stdlib root (catches everything under lib/pythonX.Y/)
        "/lib/python",
    )

    def __init__(
        self,
        event_queue: Queue,
        command_queue: Queue,
        bp_manager: BreakpointManager,
    ) -> None:
        # Do NOT pass skip= to bdb -- see _SKIP_PATHS docstring above.
        super().__init__(skip=None)
        # bdb.__init__ doesn't call reset(), but set_step()/set_continue()
        # depend on attributes that reset() initializes.
        self.reset()
        self.event_queue = event_queue
        self.command_queue = command_queue
        self.bp_manager = bp_manager

        self._current_node: str | None = None
        self._current_step: int = 0
        self._graph_state: dict[str, Any] = {}
        self._node_stack: list[tuple[Any, str]] = []
        self._run_to_node: dict[Any, str] = {}
        self._paused_event = threading.Event()
        self._paused_event.set()  # Start unpaused
        self._break_on_next_user_frame = False
        self._pending_break_on_entry = False
        self._synced_line_breaks: set[tuple[str, int]] = set()
        self._last_debug_command: DebugCommand = DebugCommand.CONTINUE

        # Register as the current tracer singleton
        AgentTracer._current = self

    @classmethod
    def get_current(cls) -> AgentTracer | None:
        """Return the current active tracer, if any."""
        return cls._current

    # ------------------------------------------------------------------
    # bdb overrides
    # ------------------------------------------------------------------

    def _is_library_frame(self, frame: FrameType) -> bool:
        """Check if a frame is in library code we should skip.

        Returns True for:
        - Frames in known library paths (langgraph, langchain, etc.)
        - Synthetic frames (<string>, <frozen>, <module>, etc.)
        - Frames not backed by a real file on disk
        """
        import os

        filename = frame.f_code.co_filename

        # Skip synthetic/internal frames
        if filename.startswith("<") or not filename:
            return True

        # Skip frames not backed by a real file
        if not os.path.isfile(filename):
            return True

        # Skip known library paths
        return any(skip in filename for skip in self._SKIP_PATHS)

    def _sync_line_breakpoints(self) -> None:
        """Mirror manager line breakpoints into bdb breakpoints."""
        desired: set[tuple[str, int]] = set()
        for bp in self.bp_manager.line_breakpoints:
            # bdb uses canonicalized absolute paths internally.
            desired.add((self.canonic(bp.filename), int(bp.lineno)))

        if desired == self._synced_line_breaks:
            return

        # Remove previously synced line breakpoints.
        for filename, lineno in self._synced_line_breaks:
            self.clear_break(filename, lineno)

        # Add current line breakpoints.
        for filename, lineno in desired:
            self.set_break(filename, lineno)

        self._synced_line_breaks = desired

    def _activate_tracing(self, break_on_entry: bool = True) -> None:
        """Activate a lightweight trace that waits for user code.

        Instead of using bdb's set_step() (which traces EVERY line
        in every frame, including library code), we install a minimal
        trace function that only watches for function calls. When it
        sees a call into user code (non-library), it activates full
        bdb tracing on that frame only.
        """
        current_trace = sys.gettrace()

        # Already in full bdb tracing mode. If a semantic breakpoint is
        # requested, force a stop at the next line in user code.
        if current_trace == self.trace_dispatch:
            if break_on_entry:
                self.set_step()
            return

        self._break_on_next_user_frame = True
        if break_on_entry:
            self._pending_break_on_entry = True

        def _wait_for_user_code(frame: FrameType, event: str, arg: Any) -> Any:
            """Lightweight trace: only fires on 'call' events."""
            if not self._break_on_next_user_frame:
                return None  # Stop tracing

            if event == "call":
                if not self._is_library_frame(frame):
                    # Found user code! Switch to full bdb tracing.
                    self._break_on_next_user_frame = False
                    # bdb continue/stop semantics depend on botframe.
                    bot = frame
                    while bot.f_back is not None:
                        bot = bot.f_back
                    self.botframe = bot
                    # Install bdb's trace on this frame
                    frame.f_trace = self.trace_dispatch
                    sys.settrace(self.trace_dispatch)

                    # Semantic breakpoints (node/tool/state/transition)
                    # should stop immediately on entry; pure line
                    # breakpoints should keep running until matched.
                    if self._pending_break_on_entry:
                        self._set_stopinfo(None, None)
                    else:
                        self.set_continue()
                    self._pending_break_on_entry = False
                    return self.trace_dispatch

            # Keep the lightweight trace active for nested calls
            return _wait_for_user_code

        sys.settrace(_wait_for_user_code)

    def user_line(self, frame: FrameType) -> None:
        """Called when bdb stops at a line."""
        # If we've left user code (back in library), deactivate tracing
        # entirely so library code runs at full speed. Without this,
        # dispatch_line runs for every line of library code (thousands
        # of lines) making the debugger appear frozen.
        if self._is_library_frame(frame):
            self.set_continue()
            return

        self.event_queue.put(
            BreakpointHit(
                frame=frame,
                filename=frame.f_code.co_filename,
                lineno=frame.f_lineno,
                node=self._current_node,
                graph_state=self._graph_state.copy(),
            )
        )

        # Block the worker thread until the UI sends a command
        cmd = self._wait_for_command()
        self._dispatch_command(cmd, frame)

    def user_return(self, frame: FrameType, return_value: Any) -> None:
        """Called when a user-code function returns."""
        if self._is_library_frame(frame):
            return
        # After an explicit continue, don't pause again on function return.
        if self._last_debug_command == DebugCommand.CONTINUE:
            return
        frame.f_locals["__return__"] = return_value
        # Show the return value, then deactivate tracing since we're
        # about to re-enter library code.
        self.event_queue.put(
            BreakpointHit(
                frame=frame,
                filename=frame.f_code.co_filename,
                lineno=frame.f_lineno,
                node=self._current_node,
                graph_state=self._graph_state.copy(),
            )
        )
        cmd = self._wait_for_command()
        # After a function return, step_over and step_into would go
        # into library code. Treat them as continue to avoid freezing.
        if cmd in (DebugCommand.STEP_OVER, DebugCommand.STEP_INTO):
            self.set_continue()
        else:
            self._dispatch_command(cmd, frame)

    def user_exception(
        self,
        frame: FrameType,
        exc_info: tuple[type, BaseException, Any],
    ) -> None:
        """Called on an exception if we're debugging."""
        exc_type, exc_value, _ = exc_info
        frame.f_locals["__exception__"] = exc_type, exc_value
        self.user_line(frame)

    # ------------------------------------------------------------------
    # LangGraph callback handler methods
    # ------------------------------------------------------------------

    # These properties tell LangChain not to raise on ignored events.
    @property
    def ignore_agent(self) -> bool:
        """Whether to ignore agent callbacks."""
        return False

    @property
    def ignore_chain(self) -> bool:
        """Whether to ignore chain callbacks."""
        return False

    @property
    def ignore_llm(self) -> bool:
        """Whether to ignore LLM callbacks."""
        return True  # Skip noisy token events

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain (graph node) starts."""
        if not name:
            return

        self._current_node = name
        self._node_stack.append((run_id, name))
        if run_id is not None:
            self._run_to_node[run_id] = name
        self._current_step += 1

        self.event_queue.put(
            NodeStartEvent(
                node=name,
                step=self._current_step,
                input=inputs,
            )
        )

        # Keep bdb line breakpoints in sync with manager breakpoints.
        self._sync_line_breakpoints()

        # If Python line breakpoints are active, run with tracing
        # enabled so bdb can stop at matching file:line locations.
        if self.bp_manager.has_line_breakpoints:
            self._activate_tracing(break_on_entry=False)

        # Check if we should break on this node
        if self.bp_manager.should_break_on_node(name):
            self._activate_tracing()

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain (graph node) ends."""
        node_name: str | None = None
        if run_id is not None:
            node_name = self._run_to_node.pop(run_id, None)
            for i in range(len(self._node_stack) - 1, -1, -1):
                stack_run_id, stack_node = self._node_stack[i]
                if stack_run_id == run_id:
                    node_name = node_name or stack_node
                    del self._node_stack[i]
                    break
        elif self._node_stack:
            _, node_name = self._node_stack.pop()

        if node_name is None:
            node_name = self._current_node

        self._current_node = (
            self._node_stack[-1][1] if self._node_stack else None
        )

        if node_name:
            result = outputs if isinstance(outputs, dict) else {}
            self.event_queue.put(
                NodeEndEvent(
                    node=node_name,
                    step=self._current_step,
                    result=result,
                )
            )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain (graph node) errors."""
        node_name: str | None = None
        if run_id is not None:
            node_name = self._run_to_node.pop(run_id, None)
            for i in range(len(self._node_stack) - 1, -1, -1):
                stack_run_id, stack_node = self._node_stack[i]
                if stack_run_id == run_id:
                    node_name = node_name or stack_node
                    del self._node_stack[i]
                    break
        elif self._node_stack:
            _, node_name = self._node_stack.pop()

        if node_name is None:
            node_name = self._current_node

        self._current_node = (
            self._node_stack[-1][1] if self._node_stack else None
        )

        if node_name:
            self.event_queue.put(
                NodeEndEvent(
                    node=node_name,
                    step=self._current_step,
                    error=str(error),
                )
            )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool is about to execute."""
        tool_name = name or serialized.get("name", "unknown")
        args = inputs or {}
        tool_call_id = (
            kwargs.get("tool_call_id")
            or args.get("tool_call_id")
            or (str(run_id) if run_id else "")
        )

        self.event_queue.put(
            ToolCallEvent(
                name=tool_name,
                args=args,
                tool_call_id=tool_call_id,
                node=self._current_node,
                step=self._current_step,
            )
        )

        # Check if we should break on this tool
        if self.bp_manager.should_break_on_tool(tool_name):
            self._activate_tracing()

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool finishes."""
        pass  # Tool results are captured via messages

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def request_break_on_next_user_frame(self) -> None:
        """Request a semantic pause at the next user-code frame."""
        self._activate_tracing()

    def _wait_for_command(self) -> DebugCommand:
        """Block the worker thread until the UI sends a command."""
        return self.command_queue.get()

    def _dispatch_command(
        self, cmd: DebugCommand, frame: FrameType
    ) -> None:
        """Execute a debug command from the UI."""
        self._last_debug_command = cmd
        if cmd == DebugCommand.CONTINUE:
            self.set_continue()
        elif cmd == DebugCommand.STEP_OVER:
            self.set_next(frame)
        elif cmd == DebugCommand.STEP_INTO:
            self.set_step()
        elif cmd == DebugCommand.STEP_OUT:
            self.set_return(frame)
        elif cmd == DebugCommand.QUIT:
            self.set_quit()

    def update_graph_state(self, state: dict[str, Any]) -> None:
        """Update the cached graph state (called from runner)."""
        self._graph_state = state

    def reset_step(self) -> None:
        """Reset step counter for a new run."""
        self._current_step = 0
        self._current_node = None
        self._node_stack = []
        self._run_to_node = {}
        for filename in list(self.breaks.keys()):
            self.clear_all_file_breaks(filename)
        self._pending_break_on_entry = False
        self._break_on_next_user_frame = False
        self._synced_line_breaks = set()
        self._last_debug_command = DebugCommand.CONTINUE
