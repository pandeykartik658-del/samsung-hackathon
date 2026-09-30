# /mnt/project-files/theme5/theme5/coordinator.py
"""Coordination layer (guide §1): call registry, cancellation tokens,
generation counters, idempotency for state-modifying tools, floor state.

Pure bookkeeping, no I/O and no awaits: the engine calls it synchronously so
cancellations land in the same handler as the event that caused them.

Generation scheme
-----------------
`Coordinator.generation` counts plans. Every re-plan starts a new generation;
calls whose arguments are still valid are *promoted* to it, the rest are
cancelled. A result is acted on only if its call is live and carries the
current generation; anything else is dropped and logged (never acted on).

Idempotency-key scheme
----------------------
key = sha256("<tool>|<canonical args>")[:16], canonical args = JSON with
sorted keys, trimmed lower-cased strings (tools.canonical_args). For
state-modifying tools the key is reserved on issue, committed on an ok
result, and released only on a definite failure (not a timeout, whose
outcome is unknown) or a cancel before any result (ASSUMPTION U10/U15: cancel-before-result == not executed). A second
issue of a reserved or committed key is blocked and logged.
"""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .protocol import IdGen, ToolResult, ToolSpec
from .state import SessionState
from .tools import canonical_args
from .trace import TraceLog


def idempotency_key(tool: str, args: dict[str, Any]) -> str:
    return hashlib.sha256(f"{tool}|{canonical_args(args)}".encode()).hexdigest()[:16]


# ================================================================ floor state


class FloorState(str, Enum):
    IDLE = "idle"
    USER_SPEAKING = "user_speaking"
    AGENT_SPEAKING = "agent_speaking"
    WAITING_ON_TOOLS = "waiting_on_tools"


# signal -> next state; "*" applies from any state. Speech has no end signal in
# the guide, so agent_speaking lasts until the next signal (ASSUMPTION U08).
FLOOR_TRANSITIONS: dict[tuple[str, str], FloorState] = {
    ("*", "user_partial"): FloorState.USER_SPEAKING,
    ("*", "interrupt"): FloorState.USER_SPEAKING,
    ("*", "agent_speak"): FloorState.AGENT_SPEAKING,
    ("*", "call_started"): FloorState.WAITING_ON_TOOLS,
    ("*", "final"): FloorState.IDLE,
}


class Floor:
    """Tracks who holds the floor. `user_eot` and `calls_idle` resolve to
    WAITING_ON_TOOLS or IDLE depending on in-flight work."""

    def __init__(self) -> None:
        self.state = FloorState.IDLE
        self.history: list[tuple[float, FloorState, str]] = []

    def signal(self, sig: str, t: float, calls_inflight: int) -> FloorState:
        if sig in ("user_eot", "calls_idle", "call_done"):
            new = FloorState.WAITING_ON_TOOLS if calls_inflight else FloorState.IDLE
        else:
            new = FLOOR_TRANSITIONS.get((self.state.value, sig)) or FLOOR_TRANSITIONS.get(("*", sig), self.state)
        if new is not self.state:
            self.history.append((t, new, sig))
            self.state = new
        return self.state

    @property
    def barge_in(self) -> bool:
        """True if the user took the floor while we held it."""
        return len(self.history) >= 2 and self.history[-1][1] is FloorState.USER_SPEAKING and \
            self.history[-2][1] in (FloorState.AGENT_SPEAKING, FloorState.WAITING_ON_TOOLS)


# ================================================================== calls


class CancelToken:
    """Cancellation token shared by a call record and its watcher task."""

    def __init__(self) -> None:
        self.reason: str | None = None
        self._event = asyncio.Event() if _has_loop() else None

    @property
    def cancelled(self) -> bool:
        return self.reason is not None

    def cancel(self, reason: str) -> None:
        if self.reason is None:
            self.reason = reason
            if self._event is not None:
                self._event.set()

    async def wait(self) -> str:
        if self._event is None:
            self._event = asyncio.Event()
            if self.reason is not None:
                self._event.set()
        await self._event.wait()
        return self.reason or ""


def _has_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


@dataclass
class Call:
    call_id: str
    tool: str
    args: dict[str, Any]
    state_modifying: bool
    generation: int
    purpose: str = "goal"  # "goal" | "prereq"
    intent: str | None = None
    attempt: int = 1
    issued_t: float = 0.0
    status: str = "inflight"  # inflight | done | failed | cancelled
    token: CancelToken = field(default_factory=CancelToken)
    idem_key: str = ""
    key: str = ""  # tool|canonical args (dedupe of identical in-flight reads too)

    def __post_init__(self) -> None:
        self.key = f"{self.tool}|{canonical_args(self.args)}"
        self.idem_key = idempotency_key(self.tool, self.args)


# verdicts returned by Coordinator.resolve
OK = "ok"
FAILED = "failed"
STALE_GENERATION = "stale_generation"
LATE_AFTER_CANCEL = "late_after_cancel"
UNKNOWN_CALL = "unknown_call"
DUPLICATE_RESULT = "duplicate_result"


class Coordinator:
    def __init__(self, ids: IdGen | None = None, trace: TraceLog | None = None,
                 clock: Callable[[], float] | None = None) -> None:
        self.ids = ids or IdGen()
        self.trace = trace if trace is not None else TraceLog()
        self.clock = clock or (lambda: 0.0)
        self.state = SessionState()
        self.floor = Floor()
        self.generation = 0
        self.calls: dict[str, Call] = {}
        self.committed: dict[str, Any] = {}  # idem key -> result
        self.reserved: set[str] = set()  # idem keys of in-flight writes

    # ---------------------------------------------------------- views
    @property
    def inflight(self) -> dict[str, Call]:
        return {cid: c for cid, c in self.calls.items() if c.status == "inflight"}

    def find_inflight(self, key: str) -> Call | None:
        return next((c for c in self.calls.values() if c.status == "inflight" and c.key == key), None)

    def is_stale(self, call_id: str) -> bool:
        c = self.calls.get(call_id)
        return c is None or c.status != "inflight" or c.generation != self.generation

    # ---------------------------------------------------------- generations
    def next_generation(self) -> int:
        self.generation += 1
        return self.generation

    def promote(self, call_id: str) -> None:
        """Keep a still-valid in-flight call alive into the current generation."""
        c = self.calls.get(call_id)
        if c is not None and c.status == "inflight":
            c.generation = self.generation

    def reconcile(self, still_wanted: Callable[[Call], bool]) -> tuple[list[Call], list[Call]]:
        """Diff a new plan against in-flight calls: start a new generation,
        promote calls the plan still wants, return (kept, invalid). The
        caller cancels `invalid` (the engine does so before any speech)."""
        self.next_generation()
        kept, invalid = [], []
        for c in list(self.inflight.values()):
            if still_wanted(c):
                self.promote(c.call_id)
                kept.append(c)
            else:
                invalid.append(c)
        self.trace.note("reconcile", self.clock(), generation=self.generation,
                        kept=[c.call_id for c in kept], invalid=[c.call_id for c in invalid])
        return kept, invalid

    # ---------------------------------------------------------- issuing
    def check_issue(self, tool: ToolSpec, args: dict[str, Any]) -> str | None:
        """None if the call may be issued, else the reason it is blocked."""
        if tool.state_modifying:
            k = idempotency_key(tool.name, args)
            if k in self.committed:
                return "duplicate_write_committed"
            if k in self.reserved:
                return "duplicate_write_pending"
        if self.find_inflight(f"{tool.name}|{canonical_args(args)}") is not None:
            return "identical_call_in_flight"
        return None

    def issue(self, tool: ToolSpec, args: dict[str, Any], purpose: str = "goal",
              intent: str | None = None, attempt: int = 1) -> Call | None:
        """Register a call, or return None (and log) if it would be a duplicate."""
        blocked = self.check_issue(tool, args)
        if blocked is not None:
            self.trace.note("blocked_call", self.clock(), tool=tool.name, args=dict(args), reason=blocked,
                            idempotency_key=idempotency_key(tool.name, args))
            return None
        c = Call(self.ids("call"), tool.name, dict(args), tool.state_modifying, self.generation,
                 purpose, intent, attempt, self.clock())
        self.calls[c.call_id] = c
        if c.state_modifying:
            self.reserved.add(c.idem_key)
        return c

    # ---------------------------------------------------------- cancelling
    def cancel(self, call_id: str, reason: str = "superseded") -> Call | None:
        """Mark cancelled (terminal: a later result is still ignored). Returns
        the call if it was in flight, else None (nothing to emit)."""
        c = self.calls.get(call_id)
        if c is None or c.status != "inflight":
            self.trace.note("cancel_ignored", self.clock(), call_id=call_id, reason=reason,
                            status=None if c is None else c.status)
            return None
        c.status = "cancelled"
        c.token.cancel(reason)
        if c.state_modifying:
            self.reserved.discard(c.idem_key)
        return c

    # ---------------------------------------------------------- results
    def resolve(self, tr: ToolResult) -> tuple[Call | None, str]:
        """Classify a tool result. Only OK / FAILED may be acted on."""
        c = self.calls.get(tr.call_id)
        t = self.clock()
        if c is None:
            self.trace.note("dropped_result", t, call_id=tr.call_id, reason=UNKNOWN_CALL)
            return None, UNKNOWN_CALL
        if c.status == "cancelled":
            self.trace.note("dropped_result", t, call_id=c.call_id, tool=c.tool, reason=LATE_AFTER_CANCEL,
                            cancel_reason=c.token.reason)
            return c, LATE_AFTER_CANCEL
        if c.status in ("done", "failed"):
            self.trace.note("dropped_result", t, call_id=c.call_id, tool=c.tool, reason=DUPLICATE_RESULT)
            return c, DUPLICATE_RESULT
        if c.generation != self.generation:
            c.status = "cancelled"
            c.token.cancel(STALE_GENERATION)
            if c.state_modifying:
                self.reserved.discard(c.idem_key)
                if tr.ok:  # it executed: these args must never be issued again
                    self.committed[c.idem_key] = tr.result
            self.trace.note("dropped_result", t, call_id=c.call_id, tool=c.tool, reason=STALE_GENERATION,
                            call_generation=c.generation, current_generation=self.generation)
            return c, STALE_GENERATION
        if tr.ok:
            c.status = "done"
            if c.state_modifying:
                self.reserved.discard(c.idem_key)
                self.committed[c.idem_key] = tr.result
            return c, OK
        c.status = "failed"
        if c.state_modifying and not tr.outcome_unknown:
            self.reserved.discard(c.idem_key)  # definite failure: same args may be sent again
        return c, FAILED

    def committed_result(self, tool: str, args: dict[str, Any]) -> tuple[bool, Any]:
        k = idempotency_key(tool, args)
        return (k in self.committed), self.committed.get(k)
