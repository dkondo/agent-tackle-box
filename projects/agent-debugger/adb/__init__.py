"""adb: Agent Debugger for LangChain/LangGraph."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adb.tracer import AgentTracer

_tracer: AgentTracer | None = None


def set_trace(paused: bool = True) -> None:
    """Drop into the agent debugger at this point.

    Usage:
        import adb; adb.set_trace()

    Requires adb to be running (launched via ``adb run``).
    """
    global _tracer
    if _tracer is None:
        from adb.tracer import AgentTracer

        _tracer = AgentTracer.get_current()
        if _tracer is None:
            raise RuntimeError(
                "No adb debugger is running. "
                "Launch your agent with: adb run <script.py>"
            )

    frame = sys._getframe().f_back
    _tracer.set_trace(frame, paused=paused)
