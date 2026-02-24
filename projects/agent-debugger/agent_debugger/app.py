"""DebuggerApp: main Textual TUI for adb."""

from __future__ import annotations

import json
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.reactive import reactive
from textual.widgets import (
    Collapsible,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    TabbedContent,
    TabPane,
)

from agent_debugger.breakpoints import BreakpointManager
from agent_debugger.events import (
    AgentErrorEvent,
    AgentResponseEvent,
    BreakpointHit,
    DebugCommand,
    NodeEndEvent,
    NodeStartEvent,
    RunFinishedEvent,
    StateUpdateEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from agent_debugger.extensions import (
    ChatOutputRenderer,
    StateMutationResult,
    StateMutator,
    StateRenderer,
    StoreRenderer,
    ToolRenderer,
)
from agent_debugger.panels.diff import DiffPanel
from agent_debugger.panels.logs import LogsPanel
from agent_debugger.panels.messages import MessagesPanel
from agent_debugger.panels.source import SourcePanel
from agent_debugger.panels.stack import StackPanel
from agent_debugger.panels.state import StatePanel
from agent_debugger.panels.store import StorePanel
from agent_debugger.panels.tools import ToolCallRecord, ToolCallsPanel
from agent_debugger.panels.variables import VariablesPanel
from agent_debugger.runner import AgentRunner


class DebugInput(Input):
    """Input subclass that intercepts backtick prefix sequences for tab navigation.

    Press ` then m/t/s/d/l to switch bottom tabs, even while typing.
    """

    def _on_key(self, event: Key) -> None:
        app = self.app
        if app._backtick_pending:
            app._backtick_pending = False
            tab_id = app._BACKTICK_TAB_MAP.get(event.key)
            if tab_id is not None:
                event.prevent_default()
                event.stop()
                app.action_focus_tab(tab_id)
                return
            # Not a tab key — insert the buffered backtick, then
            # let Input handle the current key normally.
            super()._on_key(Key("grave_accent", "`"))
            super()._on_key(event)
            return

        if event.key == "grave_accent":
            app._backtick_pending = True
            event.prevent_default()
            event.stop()
            return

        super()._on_key(event)


# Spinner frames for "Thinking..." animation
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

HISTORY_DIR = Path(".adb")
HISTORY_PATH = HISTORY_DIR / "history.json"


class ChatLog(RichLog):
    """Main chat pane with auto-wrapping text."""

    DEFAULT_CSS = """
    ChatLog {
        overflow-x: hidden;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize chat pane with responsive line-wrapping."""
        kwargs.setdefault("wrap", True)
        kwargs.setdefault("min_width", 1)
        super().__init__(**kwargs)

    def write(
        self,
        content: Any,
        width: int | None = None,
        expand: bool = False,
        shrink: bool = True,
        scroll_end: bool | None = None,
        animate: bool = False,
    ) -> ChatLog:
        """Write content, forcing wrap to container width.

        When ``wrap`` is enabled and no explicit width is given, this
        renders content at the current container width so long lines
        wrap instead of extending the virtual canvas.
        """
        if self.wrap and width is None and self._size_known:
            content_width = self.scrollable_content_region.width
            if content_width > 0:
                width = content_width
        return super().write(
            content,
            width=width,
            expand=expand,
            shrink=shrink,
            scroll_end=scroll_end,
            animate=animate,
        )


class DebuggerApp(App):
    """adb: Agent Debugger for LangChain/LangGraph."""

    CSS = """
    #main-container {
        layout: horizontal;
    }

    #left-pane {
        width: 2fr;
        height: 100%;
    }

    #right-pane {
        width: 1fr;
        height: 100%;
        border-left: solid $primary;
        overflow-y: auto;
    }

    #chat-log {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }

    #chat-input {
        dock: bottom;
        height: 3;
        margin: 1 0;
    }

    #spinner-label {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }

    #bottom-tabs {
        height: 14;
        dock: bottom;
        border-top: solid $primary;
    }

    .right-collapsible {
        height: auto;
        max-height: 100%;
        border: solid $secondary;
        margin-bottom: 1;
    }

    .right-collapsible.-collapsed {
        height: auto;
    }

    .stacked-panel {
        height: auto;
        max-height: 20;
        padding: 0 1 1 1;
        overflow-x: auto;
    }

    #state-collapsible .stacked-panel {
        max-height: 30;
    }

    #vars-collapsible .stacked-panel {
        max-height: 20;
    }

    #stack-collapsible .stacked-panel {
        max-height: 15;
    }
    """

    BINDINGS = [
        # pudb-style debug keys. These work at breakpoints because the
        # Input widget is disabled and focus moves to the ChatLog, which
        # doesn't consume single-char keys. Do NOT use priority=True --
        # that would block these letters from being typed in the Input.
        Binding("c", "continue_exec", "Continue", show=True),
        Binding("n", "step_over", "StepOver", show=True),
        Binding("s", "step_into", "StepInto", show=True),
        Binding("r", "step_out", "StepOut", show=True),
        # Tab navigation is handled via backtick prefix key sequence
        # (see on_key). Not shown in footer.
        # Other
        Binding("b", "toggle_bottom", "Bottom", show=True),
        Binding("f9", "toggle_breakpoint_prompt", "Break"),
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+c", "ctrl_c", "C-c C-c to quit", show=False),
        Binding("escape", "cancel", "Cancel"),
        Binding("q", "quit", "Quit"),
    ]

    show_bottom = reactive(True)

    def __init__(
        self,
        runner: AgentRunner,
        bp_manager: BreakpointManager,
        store_renderer: StoreRenderer | None = None,
        state_renderer: StateRenderer | None = None,
        output_renderer: ChatOutputRenderer | None = None,
        tool_renderer: ToolRenderer | None = None,
        state_mutator: StateMutator | None = None,
        state_mutation_provider: StateMutator | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.runner = runner
        self.bp_manager = bp_manager
        self.store_renderer = store_renderer
        self.state_renderer = state_renderer
        self.output_renderer = output_renderer
        self.tool_renderer = tool_renderer
        self.state_mutator = state_mutator or state_mutation_provider
        self.event_queue: Queue = runner.event_queue
        self.command_queue: Queue = runner.command_queue
        self._processing = False
        self._at_breakpoint = False
        self._last_ctrl_c: float = 0.0
        self._last_response_signature: str | None = None
        self._spinner_frame = 0
        self._spinner_timer = None
        self._input_history: list[str] = []
        self._history_index = -1
        self._history_buffer = ""
        self._previous_state: dict[str, Any] | None = None
        self._current_state: dict[str, Any] = {}
        self._suppress_breakpoint_chat_once = False
        self._last_store_had_data = False
        self._poll_timer = None
        self._turn_counter = 0
        self._backtick_pending = False

    def compose(self) -> ComposeResult:
        """Compose the IDE layout."""
        yield Header(show_clock=True)

        with Horizontal(id="main-container"):
            with Vertical(id="left-pane"):
                yield ChatLog(id="chat-log", highlight=True, markup=True)
                yield Label("", id="spinner-label")
                yield DebugInput(
                    placeholder="Enter message or /command...",
                    id="chat-input",
                )

            with Vertical(id="right-pane"):
                with Collapsible(
                    title="Store",
                    id="store-collapsible",
                    classes="right-collapsible",
                    collapsed=True,
                ):
                    yield VerticalScroll(
                        StorePanel(id="store-panel"),
                        id="store-scroll",
                        classes="stacked-panel",
                    )

                with Collapsible(
                    title="State",
                    id="state-collapsible",
                    classes="right-collapsible",
                    collapsed=False,
                ):
                    yield StatePanel(id="state-panel", classes="stacked-panel")

                with Collapsible(
                    title="Variables",
                    id="vars-collapsible",
                    classes="right-collapsible",
                    collapsed=True,
                ):
                    yield VariablesPanel(id="vars-panel", classes="stacked-panel")

                with Collapsible(
                    title="Stack",
                    id="stack-collapsible",
                    classes="right-collapsible",
                    collapsed=True,
                ):
                    yield StackPanel(id="stack-panel", classes="stacked-panel")

        with TabbedContent(id="bottom-tabs", initial="tools-tab"):
            with TabPane("Messages", id="messages-tab"):
                yield MessagesPanel(id="messages-panel", highlight=True, markup=True)
            with TabPane("Tools", id="tools-tab"):
                yield ToolCallsPanel(id="tools-panel", highlight=True, markup=True)
            with TabPane("Source", id="source-tab"):
                yield SourcePanel(id="source-panel", highlight=True, markup=True)
            with TabPane("Diff", id="diff-tab"):
                yield DiffPanel(id="diff-panel", highlight=True, markup=True)
            with TabPane("Breakpoints", id="bp-tab"):
                yield RichLog(id="bp-panel", highlight=True, markup=True)
            with TabPane("Logs", id="logs-tab"):
                yield LogsPanel(id="logs-panel", highlight=True, markup=True)

        yield Footer()

    async def on_mount(self) -> None:
        """Handle app mount."""
        self.title = "adb"
        self.sub_title = "Agent Debugger"
        self._load_history()

        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.write(Text("adb: Agent Debugger", style="bold cyan"))
        chat_log.write(Text("[Type /help for available commands]", style="dim"))
        chat_log.write(Text("=" * 40, style="dim"))
        chat_log.write("")

        self._log("adb started", "info")

        # Start polling the event queue
        self._poll_timer = self.set_interval(0.05, self._poll_events)

        self.query_one("#chat-input", Input).focus()

    @on(Input.Submitted)
    async def handle_input(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        user_input = event.value.strip()
        if not user_input:
            return

        event.input.value = ""

        # Save to history (skip quit/debug control commands)
        _skip_history = {
            "/c",
            "/n",
            "/s",
            "/r",
            "/q",
            "/quit",
            "c",
            "n",
            "s",
            "r",
            "q",
        }
        _breakpoint_debug_commands = {
            "/continue",
            "/next",
            "/step",
            "/return",
        }
        normalized_input = user_input.lower()
        skip_history = normalized_input in _skip_history or (
            self._at_breakpoint and normalized_input in _breakpoint_debug_commands
        )
        if not skip_history and (not self._input_history or self._input_history[-1] != user_input):
            self._input_history.append(user_input)
            self._save_history()
        self._history_index = -1

        if user_input.startswith("/"):
            await self._handle_command(user_input)
        else:
            if user_input == "q":
                self.action_quit()
                return
            if self._processing:
                self._log("Agent is busy, please wait.", "warning")
                return
            self._invoke_agent(user_input)

    @on(Key)
    def handle_key(self, event: Key) -> None:
        """Handle arrow key history navigation."""
        # Arrow key history navigation
        focused = self.focused
        if not isinstance(focused, Input):
            return

        if event.key == "up":
            if not self._input_history:
                return
            if self._history_index == -1:
                self._history_buffer = focused.value
                self._history_index = len(self._input_history) - 1
            elif self._history_index > 0:
                self._history_index -= 1
            focused.value = self._input_history[self._history_index]
            focused.cursor_position = len(focused.value)
            event.prevent_default()
        elif event.key == "down":
            if self._history_index == -1:
                return
            if self._history_index < len(self._input_history) - 1:
                self._history_index += 1
                focused.value = self._input_history[self._history_index]
            else:
                self._history_index = -1
                focused.value = self._history_buffer
            focused.cursor_position = len(focused.value)
            event.prevent_default()

    # ------------------------------------------------------------------
    # Agent invocation
    # ------------------------------------------------------------------

    @work(exclusive=True)
    async def _invoke_agent(self, message: str) -> None:
        """Invoke the agent with a message."""
        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.write(Text(f"$ {message}", style="green"))

        self._turn_counter += 1
        self._processing = True
        self._previous_state = self._current_state.copy()
        self._last_response_signature = None  # Reset dedup for new turn
        self._suppress_breakpoint_chat_once = False
        self._start_spinner()
        self._log(f"Invoking agent: {message[:50]}...", "info")

        # Run in worker thread
        self.runner.invoke(message)

    def _invoke_replay(self, *, pause_on_entry: bool) -> bool:
        """Invoke graph replay from the current replay cursor."""
        if pause_on_entry:
            self.runner.arm_pause_on_replay_start()

        self._processing = True
        self._previous_state = self._current_state.copy()
        self._last_response_signature = None
        self._suppress_breakpoint_chat_once = False
        self._start_spinner()

        started = self.runner.invoke_replay()
        if not started:
            self._processing = False
            self._stop_spinner()
            self._resume_input()
            return False
        self._log("Replay execution started.", "info")
        return True

    def _clear_breakpoint_context(self) -> None:
        """Clear panels that only make sense for an active paused frame."""
        self._at_breakpoint = False
        self._suppress_breakpoint_chat_once = False
        self.query_one("#source-panel", SourcePanel).clear_source()
        self.query_one("#vars-panel", VariablesPanel).clear_frame()
        self.query_one("#stack-panel", StackPanel).clear_frame()
        self.query_one("#state-panel", StatePanel).set_current_node(None)
        try:
            self.query_one("#vars-collapsible", Collapsible).collapsed = True
            self.query_one("#stack-collapsible", Collapsible).collapsed = True
        except Exception as e:
            self._log(f"Failed to collapse breakpoint panels: {e}", "warning")
        self._resume_input()

    def _apply_replay_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Apply a replay snapshot to UI panels without executing the graph."""
        self._clear_breakpoint_context()
        self._previous_state = self._current_state.copy()

        values = snapshot.get("values", {})
        if not isinstance(values, dict):
            values = {}
        self._current_state = values
        self._update_state_panel(values)

        messages = values.get("messages", [])
        if isinstance(messages, list):
            self.query_one("#messages-panel", MessagesPanel).update_messages(messages)

        self.query_one("#diff-panel", DiffPanel).update_diff(self._previous_state, values)
        store_items, store_source, store_error = self.runner.get_store_snapshot()
        self._update_store_panel(
            StateUpdateEvent(
                values=values,
                store_items=store_items,
                store_source=store_source,
                store_error=store_error,
                checkpoint_id=(
                    str(snapshot.get("checkpoint_id"))
                    if snapshot.get("checkpoint_id") is not None
                    else None
                ),
                checkpoint_config=(
                    snapshot.get("checkpoint_config")
                    if isinstance(snapshot.get("checkpoint_config"), dict)
                    else None
                ),
                checkpoint_step=(
                    snapshot.get("checkpoint_step")
                    if isinstance(snapshot.get("checkpoint_step"), int)
                    else None
                ),
                next_nodes=(
                    [str(node) for node in snapshot.get("next_nodes", [])]
                    if isinstance(snapshot.get("next_nodes"), list)
                    else []
                ),
            )
        )
        self._maybe_auto_expand_store(store_items)

    # ------------------------------------------------------------------
    # Event polling
    # ------------------------------------------------------------------

    def _poll_events(self) -> None:
        """Poll the event queue for debug events from the worker."""
        try:
            while True:
                event = self.event_queue.get_nowait()
                try:
                    self._handle_event(event)
                except Exception as e:
                    self._log(
                        f"Failed to handle {type(event).__name__}: {e}",
                        "error",
                    )
        except Empty:
            pass

        # Recovery path: if the worker has exited but RunFinished was not
        # processed (e.g., due a transient event handler failure), ensure
        # the UI does not remain stuck in "Thinking...".
        if self._processing and not self._at_breakpoint and not self.runner.is_running:
            self._stop_spinner()
            self._processing = False
            self._resume_input()
            self._log(
                "Recovered stale processing state after worker exit.",
                "warning",
            )

    def _handle_event(self, event: Any) -> None:
        """Handle a single debug event."""
        if isinstance(event, NodeStartEvent):
            state_panel = self.query_one("#state-panel", StatePanel)
            state_panel.set_current_node(event.node)
            self._log(
                f"Node started: {event.node} (step {event.step})",
                "debug",
            )

        elif isinstance(event, NodeEndEvent):
            if event.error:
                self._log(f"Node error: {event.node}: {event.error}", "error")
            else:
                self._log(
                    f"Node ended: {event.node} (step {event.step})",
                    "debug",
                )

        elif isinstance(event, ToolCallEvent):
            tools_panel = self.query_one("#tools-panel", ToolCallsPanel)
            tools_panel.add_tool_call(
                ToolCallRecord(
                    name=event.name,
                    args=event.args,
                    tool_call_id=event.tool_call_id,
                    node=event.node,
                    step=event.step,
                    turn=self._turn_counter or None,
                )
            )
            self._maybe_render_tools_panel()
            self._log(f"Tool call: {event.name}", "debug")

        elif isinstance(event, ToolResultEvent):
            tools_panel = self.query_one("#tools-panel", ToolCallsPanel)
            tools_panel.update_result(
                event.tool_call_id,
                result=event.result,
                error=event.error,
                duration_ms=event.duration_ms,
            )
            self._maybe_render_tools_panel()

        elif isinstance(event, StateUpdateEvent):
            self._current_state = event.values
            self._update_state_panel(event.values)

            # Update messages panel
            messages = event.values.get("messages", [])
            if messages:
                msg_panel = self.query_one("#messages-panel", MessagesPanel)
                msg_panel.update_messages(messages)

            # Update diff panel
            diff_panel = self.query_one("#diff-panel", DiffPanel)
            diff_panel.update_diff(self._previous_state, event.values)
            self._update_store_panel(event)
            self._maybe_auto_expand_store(event.store_items)

        elif isinstance(event, BreakpointHit):
            try:
                self._handle_breakpoint(event)
            except Exception as e:
                # If breakpoint rendering fails, the worker thread is
                # blocked waiting for a debug command. Auto-continue so
                # runs cannot deadlock in a perpetual "waiting" state.
                self._at_breakpoint = False
                self._suppress_breakpoint_chat_once = False
                self._replace_pending_command(DebugCommand.CONTINUE)
                if self._processing:
                    self._start_spinner()
                self._resume_input()
                self._log(
                    (f"Failed to handle breakpoint UI; auto-continued execution: {e}"),
                    "error",
                )

        elif isinstance(event, AgentResponseEvent):
            # Deduplicate: values and updates modes can both emit the
            # same response.
            signature = self._response_signature(event)
            if signature != self._last_response_signature:
                self._last_response_signature = signature
                chat_log = self.query_one("#chat-log", ChatLog)
                chat_log.write("")
                if not self._render_custom_chat_output(event, chat_log):
                    if event.text:
                        chat_log.write(Text(event.text, style="cyan"))

        elif isinstance(event, AgentErrorEvent):
            chat_log = self.query_one("#chat-log", ChatLog)
            chat_log.write("")
            chat_log.write(Text(f"Error: {event.message}", style="bold red"))
            self._log(f"Agent error: {event.message}", "error")

        elif isinstance(event, RunFinishedEvent):
            self._stop_spinner()
            self._processing = False
            self._at_breakpoint = False
            self._resume_input()
            self._suppress_breakpoint_chat_once = False
            self._log("Run finished", "info")

    def _handle_breakpoint(self, event: BreakpointHit) -> None:
        """Handle a breakpoint hit -- update UI panels."""
        self._at_breakpoint = True
        self._stop_spinner()

        # Disable the input widget so debug keys (c/n/s/r) are handled
        # by the App's BINDINGS instead of being consumed as text input.
        # Focus the ChatLog -- it doesn't consume single-char keys, so
        # bindings for c/n/s/r will fire correctly.
        try:
            inp = self.query_one("#chat-input", Input)
            inp.disabled = True
            inp.placeholder = "Debugger: c=continue  n=next  s=step  r=return"
            chat_log = self.query_one("#chat-log", ChatLog)
            chat_log.focus()
        except Exception as e:
            self._log(f"Failed to enter breakpoint input mode: {e}", "warning")

        # Show source in bottom tabs
        source_panel = self.query_one("#source-panel", SourcePanel)
        bp_lines = set()
        for bp in self.bp_manager.breakpoints:
            if bp.filename == event.filename and bp.lineno is not None:
                bp_lines.add(bp.lineno)
        source_panel.show_source(event.filename, event.lineno, bp_lines)

        # Auto-switch to source tab
        tabs = self.query_one("#bottom-tabs", TabbedContent)
        tabs.active = "source-tab"

        # Show variables
        vars_panel = self.query_one("#vars-panel", VariablesPanel)
        vars_panel.update_frame(event.frame)

        # Expand variables collapsible
        try:
            vars_coll = self.query_one("#vars-collapsible", Collapsible)
            vars_coll.collapsed = False
        except Exception as e:
            self._log(f"Failed to expand variables panel: {e}", "warning")

        # Show stack
        stack_panel = self.query_one("#stack-panel", StackPanel)
        stack_panel.update_frame(event.frame)

        # Expand stack collapsible
        try:
            stack_coll = self.query_one("#stack-collapsible", Collapsible)
            stack_coll.collapsed = False
        except Exception as e:
            self._log(f"Failed to expand stack panel: {e}", "warning")

        # Update state with graph context
        if event.graph_state:
            self._update_state_panel(event.graph_state)
        self._ensure_store_visible_at_breakpoint()

        node_info = f" in {event.node}" if event.node else ""
        self._log(
            f"Breakpoint hit: {event.filename}:{event.lineno}{node_info}",
            "info",
        )

        if self._suppress_breakpoint_chat_once:
            self._suppress_breakpoint_chat_once = False
        else:
            chat_log = self.query_one("#chat-log", ChatLog)
            chat_log.write(
                Text(
                    f"⏸ Breakpoint: {event.filename}:{event.lineno}{node_info}",
                    style="bold yellow",
                )
            )

    def _response_signature(self, event: AgentResponseEvent) -> str:
        """Build a stable signature for response deduplication."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        return json.dumps(
            {"text": event.text, "payload": payload},
            sort_keys=True,
            default=str,
        )

    def _render_custom_chat_output(self, event: AgentResponseEvent, chat_log: ChatLog) -> bool:
        """Render chat output through an optional renderer."""
        renderer = self.output_renderer
        if renderer is None:
            return False

        payload = event.payload if isinstance(event.payload, dict) else {}
        try:
            if not renderer.can_render(payload):
                return False
            model = renderer.render_chat_output(
                payload=payload,
                state=self._current_state,
                messages=self._current_state.get("messages", []),
            )
        except Exception as e:
            self._log(f"Output renderer failed: {e}", "warning")
            return False

        if model is None:
            return False

        lines = getattr(model, "lines", None)
        if isinstance(lines, str):
            lines = [lines]
        if not isinstance(lines, list):
            return False
        if not any(str(line).strip() for line in lines):
            return False

        for line in lines:
            text_line = str(line)
            try:
                chat_log.write(Text.from_markup(text_line))
            except Exception:
                # Renderer markup should not suppress chat output.
                chat_log.write(Text(text_line))
        return True

    def _update_state_panel(self, state: dict[str, Any]) -> None:
        """Update state panel via optional state renderer."""
        panel = self.query_one("#state-panel", StatePanel)
        snapshot = {"state": state}
        renderer = self.state_renderer
        if renderer is not None:
            try:
                model = renderer.render_state(snapshot)
            except Exception as e:
                self._log(f"State renderer failed: {e}", "warning")
            else:
                lines = self._normalize_lines(model)
                if lines is not None:
                    panel.update_custom_lines(lines, state=state)
                    return

        panel.update_state(state)

    def _maybe_render_tools_panel(self) -> None:
        """Update tools panel through optional custom renderer."""
        panel = self.query_one("#tools-panel", ToolCallsPanel)
        renderer = self.tool_renderer
        if renderer is None:
            panel.clear_custom_lines()
            return

        records = panel.records
        # Sort most-recent turn first; within a turn, most-recent call first.
        ordered = sorted(
            enumerate(records),
            key=lambda item: (
                -(item[1].turn if item[1].turn is not None else -1),
                -item[0],
            ),
        )
        snapshot = {
            "tool_calls": [
                {
                    "name": rec.name,
                    "args": rec.args,
                    "tool_call_id": rec.tool_call_id,
                    "result": rec.result,
                    "error": rec.error,
                    "duration_ms": rec.duration_ms,
                    "node": rec.node,
                    "step": rec.step,
                    "turn": rec.turn,
                }
                for _, rec in ordered
            ],
            "state": self._current_state,
            "turn": self._turn_counter,
        }
        try:
            model = renderer.render_tools(snapshot)
        except Exception as e:
            self._log(f"Tool renderer failed: {e}", "warning")
            panel.clear_custom_lines()
            return

        lines = self._normalize_lines(model)
        if lines is None:
            panel.clear_custom_lines()
            return
        panel.update_custom_lines(lines)

    def _update_store_panel(self, event: StateUpdateEvent) -> None:
        """Update the backend store panel via optional renderers."""
        panel = self.query_one("#store-panel", StorePanel)
        snapshot = {
            "state": event.values,
            "store_items": event.store_items,
            "store_source": event.store_source,
            "store_error": event.store_error,
        }

        if self.store_renderer is not None:
            try:
                model = self.store_renderer.render_store(snapshot)
            except Exception as e:
                self._log(f"Store renderer failed: {e}", "warning")
            else:
                lines = self._normalize_lines(model)
                if lines is not None:
                    panel.update_custom_lines(
                        lines,
                        source=event.store_source,
                        error=event.store_error,
                    )
                    return

        panel.update_store(
            event.store_items,
            source=event.store_source,
            error=event.store_error,
        )

    def _normalize_lines(self, model: Any) -> list[str] | None:
        """Coerce renderer output to string lines."""
        if model is None:
            return None
        lines = getattr(model, "lines", None)
        if isinstance(lines, str):
            lines = [lines]
        if isinstance(lines, list):
            return [str(line) for line in lines]
        return None

    def _has_visible_store(self, store_items: dict[str, dict[str, Any]]) -> bool:
        """Whether backend store currently has entries."""
        if not store_items:
            return False
        return any(bool(entries) for entries in store_items.values())

    def _maybe_auto_expand_store(self, store_items: dict[str, dict[str, Any]]) -> None:
        """Auto-expand Store when backend store data first appears."""
        has_store = self._has_visible_store(store_items)
        if has_store and not self._last_store_had_data:
            try:
                store_collapsible = self.query_one("#store-collapsible", Collapsible)
                store_collapsible.collapsed = False
            except Exception as e:
                self._log(f"Failed to auto-expand store panel: {e}", "warning")
        self._last_store_had_data = has_store

    def _ensure_store_visible_at_breakpoint(self) -> None:
        """Keep Store visible while stepping through breakpoint stops."""
        panel = self.query_one("#store-panel", StorePanel)
        has_store = bool(panel.items) or bool(panel.custom_lines)
        if not has_store:
            return
        try:
            store_collapsible = self.query_one("#store-collapsible", Collapsible)
            store_collapsible.collapsed = False
        except Exception as e:
            self._log(f"Failed to keep store panel visible: {e}", "warning")

    def _clear_local_context(self) -> None:
        """Clear local chat/panels/snapshot metadata."""
        self.query_one("#chat-log", ChatLog).clear()
        self.query_one("#messages-panel", MessagesPanel).clear()
        self.query_one("#tools-panel", ToolCallsPanel).clear_records()
        self.query_one("#diff-panel", DiffPanel).clear()
        self.query_one("#state-panel", StatePanel).set_current_node(None)
        self.query_one("#state-panel", StatePanel).update_state({})
        self.query_one("#store-panel", StorePanel).update_store({})
        self._previous_state = None
        self._current_state = {}
        self._last_response_signature = None
        self._last_store_had_data = False
        self._turn_counter = 0
        self._stop_spinner()
        self._processing = False

    def _apply_state_mutation(self, mutation: str, args: list[str]) -> StateMutationResult:
        """Apply a generic state mutation via mutator when configured."""
        if self.state_mutator is None:
            return StateMutationResult(
                applied=False,
                message=("State mutator is not configured; only local clear was applied."),
            )

        try:
            result = self.state_mutator.mutate_state(
                mutation=mutation,
                args=args,
                current_state=self._current_state,
                runner=self.runner,
            )
        except Exception as e:
            return StateMutationResult(
                applied=False,
                message=f"State mutation failed: {e}",
            )

        if result is None:
            return StateMutationResult(
                applied=False,
                message=f"Mutation '{mutation}' was not handled.",
            )
        return result

    def _available_graph_nodes(self) -> set[str] | None:
        """Return known graph node names when discoverable."""
        names: set[str] = set()
        graph = getattr(self.runner, "graph", None)
        if graph is None:
            return None

        nodes_attr = getattr(graph, "nodes", None)
        if isinstance(nodes_attr, dict):
            names.update(str(name) for name in nodes_attr.keys())

        if not names and hasattr(graph, "get_graph"):
            try:
                ui_graph = graph.get_graph()
            except Exception:
                ui_graph = None
            graph_nodes = getattr(ui_graph, "nodes", None)
            if isinstance(graph_nodes, dict):
                names.update(str(name) for name in graph_nodes.keys())

        names.discard("__start__")
        names.discard("__end__")
        return names or None

    # ------------------------------------------------------------------
    # Debug commands
    # ------------------------------------------------------------------

    def _update_input_placeholder(self) -> None:
        """Update the input placeholder based on current state."""
        try:
            inp = self.query_one("#chat-input", Input)
            if inp.disabled:
                return  # At breakpoint, placeholder is set by _handle_breakpoint
            if self.bp_manager.breakpoints:
                inp.placeholder = "Enter message, /command, or when breakpoints active: c/n/s/r"
            else:
                inp.placeholder = "Enter message or /command..."
        except Exception as e:
            self._log(f"Failed to update input placeholder: {e}", "warning")

    def _resume_input(self) -> None:
        """Re-enable the chat input after leaving a breakpoint."""
        try:
            inp = self.query_one("#chat-input", Input)
            inp.disabled = False
            inp.focus()
            self._update_input_placeholder()
        except Exception as e:
            self._log(f"Failed to resume input: {e}", "warning")

    def _signal_worker_quit(self) -> None:
        """Signal the worker thread to quit if it is blocked on commands."""
        try:
            self.command_queue.put_nowait(DebugCommand.QUIT)
        except Exception as e:
            self._log(f"Failed to enqueue quit command: {e}", "warning")

    def _replace_pending_command(self, cmd: DebugCommand) -> None:
        """Drop queued debug commands and enqueue a single replacement."""
        cleared = 0
        try:
            while True:
                self.command_queue.get_nowait()
                cleared += 1
        except Empty:
            pass

        self.command_queue.put(cmd)
        if cleared:
            self._log(
                f"Cleared {cleared} queued debug command(s) before {cmd.name}.",
                "debug",
            )

    def action_continue_exec(self) -> None:
        """Continue execution (c)."""
        if not self._at_breakpoint:
            return
        self._at_breakpoint = False
        self._suppress_breakpoint_chat_once = False
        self._replace_pending_command(DebugCommand.CONTINUE)
        self._start_spinner()
        self._resume_input()

        # Collapse debug panels
        try:
            self.query_one("#vars-collapsible", Collapsible).collapsed = True
            self.query_one("#stack-collapsible", Collapsible).collapsed = True
        except Exception as e:
            self._log(f"Failed to collapse debug panels: {e}", "warning")

        # Clear source
        source_panel = self.query_one("#source-panel", SourcePanel)
        source_panel.clear_source()

        self._log("Continuing execution", "info")

    def action_step_over(self) -> None:
        """Step over (n - next line, same as pudb)."""
        if not self._at_breakpoint:
            return
        self._suppress_breakpoint_chat_once = True
        self._replace_pending_command(DebugCommand.STEP_OVER)
        self._log("Step over", "debug")

    def action_step_into(self) -> None:
        """Step into (s - step, same as pudb)."""
        if not self._at_breakpoint:
            return
        self._suppress_breakpoint_chat_once = True
        self._replace_pending_command(DebugCommand.STEP_INTO)
        self._log("Step into", "debug")

    def action_step_out(self) -> None:
        """Step out / finish (r - return, same as pudb)."""
        if not self._at_breakpoint:
            return
        self._suppress_breakpoint_chat_once = True
        self._replace_pending_command(DebugCommand.STEP_OUT)
        self._log("Step out", "debug")

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    async def _handle_command(self, cmd: str) -> None:
        """Handle a slash command."""
        chat_log = self.query_one("#chat-log", ChatLog)
        parts = cmd.split(maxsplit=2)
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        if command == "/help":
            self._show_help()

        elif command in ("/c", "/continue"):
            if self._at_breakpoint:
                self.action_continue_exec()
            else:
                chat_log.write(Text("Not at a breakpoint.", style="yellow"))

        elif command in ("/n", "/next"):
            if self._at_breakpoint:
                self.action_step_over()
            else:
                chat_log.write(Text("Not at a breakpoint.", style="yellow"))

        elif command in ("/s", "/step"):
            if self._at_breakpoint:
                self.action_step_into()
            else:
                chat_log.write(Text("Not at a breakpoint.", style="yellow"))

        elif command in ("/r", "/return"):
            if self._at_breakpoint:
                self.action_step_out()
            else:
                chat_log.write(Text("Not at a breakpoint.", style="yellow"))

        elif command in ("/q", "/quit"):
            self.action_quit()

        elif command == "/break" or command == "/b":
            if not args:
                chat_log.write(
                    Text(
                        "Usage: /break node|tool|state|line <name>",
                        style="yellow",
                    )
                )
                return
            bp_type = args[0].lower()
            name = args[1] if len(args) > 1 else ""

            if bp_type == "node" and name:
                available_nodes = self._available_graph_nodes()
                if available_nodes is not None and name not in available_nodes:
                    available = ", ".join(sorted(available_nodes))
                    chat_log.write(
                        Text(
                            f"Error: node '{name}' does not exist. Available nodes: {available}",
                            style="red",
                        )
                    )
                    self._log(
                        f"Invalid node breakpoint: '{name}'",
                        "warning",
                    )
                    return
                bp = self.bp_manager.add_node(name)
                chat_log.write(Text(f"Breakpoint set: {bp}", style="green"))
                if self._processing:
                    chat_log.write(
                        Text(
                            "Auto-seek skipped while agent is running.",
                            style="yellow",
                        )
                    )
                else:
                    supported, reason = self.runner.supports_replay()
                    if supported:
                        ok, message, snapshot = self.runner.seek_nearest_to_node(name)
                        if ok and snapshot is not None:
                            self._apply_replay_snapshot(snapshot)
                            chat_log.write(Text(message, style="green"))
                            if self._invoke_replay(pause_on_entry=True):
                                chat_log.write(
                                    Text(
                                        "Replay resumed. Stepping will activate at breakpoint.",
                                        style="green",
                                    )
                                )
                            else:
                                chat_log.write(Text("Failed to start replay.", style="red"))
                        else:
                            chat_log.write(Text(message, style="yellow"))
                    else:
                        chat_log.write(Text(f"Replay unavailable: {reason}", style="dim"))
            elif bp_type == "tool" and name:
                bp = self.bp_manager.add_tool(name)
                chat_log.write(Text(f"Breakpoint set: {bp}", style="green"))
            elif bp_type == "state" and name:
                bp = self.bp_manager.add_state(name)
                chat_log.write(Text(f"Breakpoint set: {bp}", style="green"))
            elif bp_type == "transition":
                bp = self.bp_manager.add_transition()
                chat_log.write(Text(f"Breakpoint set: {bp}", style="green"))
            elif bp_type == "line" and name:
                # Parse file:line format
                if ":" in name:
                    fpath, lstr = name.rsplit(":", 1)
                    try:
                        lineno = int(lstr)
                        bp = self.bp_manager.add_line(fpath, lineno)
                        chat_log.write(
                            Text(
                                f"Breakpoint set: {bp}",
                                style="green",
                            )
                        )
                    except ValueError:
                        chat_log.write(
                            Text(
                                "Invalid line number.",
                                style="red",
                            )
                        )
                else:
                    chat_log.write(
                        Text(
                            "Usage: /break line file.py:42",
                            style="yellow",
                        )
                    )
            else:
                chat_log.write(
                    Text(
                        "Usage: /break node|tool|state|line|transition <name>",
                        style="yellow",
                    )
                )
            self._refresh_breakpoints_panel()
            self._update_input_placeholder()

        elif command in ("/rewind", "/forward", "/foward"):
            if self._processing:
                chat_log.write(Text("Agent is busy, please wait.", style="yellow"))
                return
            if len(args) < 2 or args[0].lower() != "node":
                chat_log.write(
                    Text(
                        "Usage: /rewind node <name> or /forward node <name>",
                        style="yellow",
                    )
                )
                return

            node_name = args[1]
            available_nodes = self._available_graph_nodes()
            if available_nodes is not None and node_name not in available_nodes:
                available = ", ".join(sorted(available_nodes))
                chat_log.write(
                    Text(
                        f"Error: node '{node_name}' does not exist. Available nodes: {available}",
                        style="red",
                    )
                )
                return

            supported, reason = self.runner.supports_replay()
            if not supported:
                chat_log.write(Text(f"Replay unavailable: {reason}", style="yellow"))
                return

            if command == "/rewind":
                ok, message, snapshot = self.runner.seek_backward_to_node(node_name)
            else:
                ok, message, snapshot = self.runner.seek_forward_to_node(node_name)

            if not ok or snapshot is None:
                chat_log.write(Text(message, style="yellow"))
                return

            self._apply_replay_snapshot(snapshot)
            chat_log.write(Text(message, style="green"))

        elif command == "/breakpoints" or command == "/bp":
            self._refresh_breakpoints_panel()
            tabs = self.query_one("#bottom-tabs", TabbedContent)
            tabs.active = "bp-tab"

        elif command == "/messages":
            self.action_focus_tab("messages-tab")

        elif command == "/tools":
            self.action_focus_tab("tools-tab")

        elif command == "/source":
            self.action_focus_tab("source-tab")

        elif command == "/diff":
            self.action_focus_tab("diff-tab")

        elif command == "/logs":
            self.action_focus_tab("logs-tab")

        elif command == "/clear":
            if args and args[0] == "bp":
                count = self.bp_manager.clear()
                chat_log.write(
                    Text(
                        f"Cleared {count} breakpoint(s).",
                        style="green",
                    )
                )
                self._refresh_breakpoints_panel()
                self._update_input_placeholder()
            else:
                mutation = args[0] if args else "local"
                mutation_args = args[1:] if len(args) > 1 else []
                self._clear_local_context()
                chat_log.write(
                    Text(
                        "Cleared local chat/panels/snapshot metadata.",
                        style="green",
                    )
                )
                if mutation not in {"local", ""}:
                    result = self._apply_state_mutation(mutation, mutation_args)
                    if result.applied:
                        chat_log.write(
                            Text(
                                result.message or f"Applied mutation '{mutation}'.",
                                style="green",
                            )
                        )
                    elif result.message:
                        chat_log.write(Text(result.message, style="yellow"))

        elif command == "/state":
            self._update_state_panel(self._current_state)
            store_items, store_source, store_error = self.runner.get_store_snapshot()
            self._update_store_panel(
                StateUpdateEvent(
                    values=self._current_state,
                    store_items=store_items,
                    store_source=store_source,
                    store_error=store_error,
                )
            )
            chat_log.write(Text("State refreshed.", style="green"))

        elif command == "/store":
            store_items, store_source, store_error = self.runner.get_store_snapshot()
            self._update_store_panel(
                StateUpdateEvent(
                    values=self._current_state,
                    store_items=store_items,
                    store_source=store_source,
                    store_error=store_error,
                )
            )
            self._maybe_auto_expand_store(store_items)
            chat_log.write(Text("Store refreshed.", style="green"))

        elif command == "/theme":
            if self.dark:
                self.theme = "textual-light"
                self.dark = False
            else:
                self.theme = "textual-dark"
                self.dark = True

        elif command == "/history":
            count = 10
            if args:
                try:
                    count = int(args[0])
                except ValueError:
                    pass
            entries = self._input_history[-count:]
            start = len(self._input_history) - len(entries) + 1
            chat_log.write(Text("Recent history:", style="bold"))
            for idx, entry in enumerate(entries, start=start):
                chat_log.write(Text(f"  {idx}. {entry}", style="cyan"))

        else:
            chat_log.write(Text(f"Unknown command: {command}", style="red"))

    def _show_help(self) -> None:
        """Show help text."""
        chat_log = self.query_one("#chat-log", ChatLog)
        help_text = """\
[bold cyan]adb Commands:[/bold cyan]

[cyan]Breakpoints:[/cyan]
  /break node <name>    Set node breakpoint + auto-seek + replay
  /break tool <name>    Break when tool is called
  /break state <key>    Break when state key changes
  /break transition     Break on every node transition
  /break line file:42   Standard Python breakpoint
  /breakpoints          Show all breakpoints
  /rewind node <name>   Move replay cursor to prior node checkpoint
  /forward node <name>  Move replay cursor to later node checkpoint
  /foward node <name>   Alias for /forward
  /clear bp             Clear all breakpoints
  /clear [local]        Clear local chat/panels/snapshot metadata
  /clear <mutation>     Local clear + optional mutator mutation

[cyan]Debugging (when at breakpoint, pudb-style):[/cyan]
  c                     Continue execution
  n                     Step over (next line)
  s                     Step into
  r                     Step out (return / finish)
  /c /n /s /r           Slash aliases for debug commands

[cyan]Navigation:[/cyan]
  ` m                   Messages tab
  ` t                   Tools tab
  ` s                   Source tab
  ` d                   Diff tab
  ` b                   Breakpoints tab
  ` l                   Logs tab
  /messages /tools      Switch bottom tab
  /source /diff /logs   Switch bottom tab
  b                     Toggle bottom panel

[cyan]Other:[/cyan]
  /state                Refresh state panel
  /store                Refresh store panel from backend
  /theme                Toggle light/dark
  /history [n]          Show input history
  /q, /quit, q, Ctl-c Ctl-c   Quit
  /help                 Show this help
"""
        chat_log.write(Text.from_markup(help_text))

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _refresh_breakpoints_panel(self) -> None:
        """Refresh the breakpoints panel."""
        panel = self.query_one("#bp-panel", RichLog)
        panel.clear()
        bps = self.bp_manager.breakpoints
        if not bps:
            panel.write(Text("No breakpoints set.", style="dim"))
            return
        for i, bp in enumerate(bps):
            status = "●" if bp.enabled else "○"
            hits = f" (hits: {bp.hits})" if bp.hits else ""
            panel.write(
                Text(
                    f"  {status} [{i}] {bp}{hits}",
                    style="green" if bp.enabled else "dim",
                )
            )

    def _start_spinner(self) -> None:
        """Start the thinking spinner."""
        self._spinner_frame = 0
        if self._spinner_timer is None:
            self._spinner_timer = self.set_interval(0.1, self._update_spinner)

    def _update_spinner(self) -> None:
        """Update the spinner animation."""
        self._spinner_frame = (self._spinner_frame + 1) % len(SPINNER_FRAMES)
        try:
            label = self.query_one("#spinner-label", Label)
            label.update(f"{SPINNER_FRAMES[self._spinner_frame]} Thinking...")
        except Exception as e:
            self._log(f"Failed to update spinner: {e}", "debug")

    def _stop_spinner(self) -> None:
        """Stop the spinner."""
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None
        try:
            label = self.query_one("#spinner-label", Label)
            label.update("")
        except Exception as e:
            self._log(f"Failed to clear spinner: {e}", "debug")

    def _log(self, message: str, level: str = "info") -> None:
        """Add a log entry."""
        try:
            logs = self.query_one("#logs-panel", LogsPanel)
            logs.log(message, level)
        except Exception as e:
            import sys

            print(f"[adb] log panel unavailable: {e}", file=sys.stderr)  # noqa: T201

    def _load_history(self) -> None:
        """Load input history from disk."""
        try:
            if not HISTORY_PATH.exists():
                return
            with open(HISTORY_PATH) as f:
                data = json.load(f)
            if isinstance(data, list):
                self._input_history = [str(x) for x in data if x]
        except Exception as e:
            self._log(f"Failed to load command history: {e}", "warning")

    def _save_history(self) -> None:
        """Save input history to disk."""
        try:
            HISTORY_DIR.mkdir(exist_ok=True)
            with open(HISTORY_PATH, "w") as f:
                json.dump(self._input_history[-500:], f, indent=2)
        except Exception as e:
            self._log(f"Failed to save command history: {e}", "warning")

    # ------------------------------------------------------------------
    # Backtick prefix key sequence for tab navigation
    # ------------------------------------------------------------------

    _BACKTICK_TAB_MAP: dict[str, str] = {
        "m": "messages-tab",
        "t": "tools-tab",
        "s": "source-tab",
        "d": "diff-tab",
        "b": "bp-tab",
        "l": "logs-tab",
    }

    def on_key(self, event: Key) -> None:
        """Handle backtick prefix key sequence for tab navigation.

        Press ` then m/t/s/d/l to switch tabs. Works regardless of
        which widget has focus.
        """
        if self._backtick_pending:
            self._backtick_pending = False
            tab_id = self._BACKTICK_TAB_MAP.get(event.key)
            if tab_id is not None:
                event.prevent_default()
                event.stop()
                self.action_focus_tab(tab_id)
            return

        if event.key == "grave_accent":
            self._backtick_pending = True
            event.prevent_default()
            event.stop()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_focus_tab(self, tab_id: str) -> None:
        """Switch to a specific bottom tab."""
        try:
            tabs = self.query_one("#bottom-tabs", TabbedContent)
            tabs.active = tab_id
        except Exception as e:
            self._log(f"Failed to focus tab '{tab_id}': {e}", "warning")

    def action_toggle_bottom(self) -> None:
        """Toggle the bottom panel visibility."""
        self.show_bottom = not self.show_bottom

    def watch_show_bottom(self, show: bool) -> None:
        """React to show_bottom changes."""
        try:
            tabs = self.query_one("#bottom-tabs", TabbedContent)
            tabs.display = show
        except Exception as e:
            self._log(f"Failed to toggle bottom panel visibility: {e}", "warning")

    def action_clear_chat(self) -> None:
        """Clear the chat log."""
        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.clear()

    def action_cancel(self) -> None:
        """Cancel / dismiss."""
        pass

    def action_ctrl_c(self) -> None:
        """Handle Ctrl-C. Double-tap (C-c C-c) to quit."""
        import time

        now = time.monotonic()
        if now - self._last_ctrl_c < 1.0:
            # Double Ctrl-C within 1 second -> quit
            self.action_quit()
        else:
            self._last_ctrl_c = now
            self._log("Press C-c again to quit", "warning")
            try:
                chat_log = self.query_one("#chat-log", ChatLog)
                chat_log.write(Text("Press C-c again to quit.", style="yellow"))
            except Exception as e:
                self._log(f"Failed to write Ctrl-C hint: {e}", "warning")

    def action_quit(self) -> None:
        """Quit the app and unblock paused debug threads."""
        self._signal_worker_quit()
        self.exit()

    def action_toggle_breakpoint_prompt(self) -> None:
        """Focus input with /break prefix."""
        inp = self.query_one("#chat-input", Input)
        inp.value = "/break "
        inp.focus()
        inp.cursor_position = len(inp.value)
