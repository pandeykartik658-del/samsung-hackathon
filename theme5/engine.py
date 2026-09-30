# /mnt/project-files/theme5/theme5/engine.py
"""Event loop: two asyncio queues (events in, actions out), every action
timestamped by the session clock, validated and serialised in one place.

The engine owns the mechanics (ids, clock, coordinator, slow path, per-call
watcher tasks, trace). Policy lives in a handler (agent.Agent) that decides
what to say and which tools to call.
"""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Callable, Protocol

from . import protocol as P
from .clock import Clock, EventClock
from .coordinator import FAILED, OK, Call, Coordinator
from .events import Action, Cancel, FinalResponse, ToolCall, action_fields
from .protocol import Event, IdGen, ToolResult, ToolSpec, WireFormat
from .slowpath import SlowPath
from .trace import TraceLog
from .watchdog import Watchdog

log = logging.getLogger("theme5.engine")


class Handler(Protocol):
    async def on_event(self, ev: Event) -> None:
        """Every non-result event, in arrival order."""

    def on_tool_result(self, call: Call, result: ToolResult) -> None:
        """Only results the coordinator accepted (live call, current generation)."""


class Engine:
    def __init__(self, handler: Handler, clock: Clock | None = None,
                 snapshot_fn: Callable[[], dict[str, Any]] | None = None) -> None:
        self.handler = handler
        self.ids = IdGen()
        self.wire = WireFormat()
        self.clock: Clock = clock or EventClock()
        self.trace = TraceLog()
        self.coord = Coordinator(self.ids, self.trace, self.clock)
        self.slow = SlowPath(self.trace, self.clock)
        self.snapshot_fn = snapshot_fn or self.coord.state.snapshot
        self._out: asyncio.Queue[dict[str, Any]] | None = None
        self._watchers: dict[str, asyncio.Task[None]] = {}
        self.watchdog = Watchdog(self)
        self.turns = 0  # user turns seen (end-of-turn text/audio, interrupts)
        self.answered = 0  # value of `turns` at the last final_response

    @property
    def turns_unanswered(self) -> bool:
        return self.answered < self.turns

    # ------------------------------------------------------------------ loop
    async def run(self, inbox: asyncio.Queue[Any], outbox: asyncio.Queue[dict[str, Any]]) -> None:
        self._out = outbox
        self.watchdog.start()
        try:
            while True:
                raw = await inbox.get()
                try:
                    if raw is None:
                        break
                    try:
                        ev = self.parse(raw)
                    except Exception as exc:  # noqa: BLE001 - a malformed event is skipped, never fatal
                        self.trace.note("malformed_event", self.clock(), error=repr(exc), raw=repr(raw)[:200])
                        continue
                    if ev.type == P.EV_END:
                        break
                    await self.dispatch(ev)
                finally:
                    inbox.task_done()
            await self._bounded_drain()
            if P.CANCEL_PENDING_ON_END:
                for cid in list(self.coord.inflight):
                    self.cancel_call(cid, "session_end")
        finally:
            self.watchdog.stop()
            self.slow.cancel_all()
            for t in self._watchers.values():
                t.cancel()
            self._watchers.clear()

    async def _bounded_drain(self) -> None:
        """Let slow-path work finish after session end, but never past the
        watchdog point: then the watchdog closes the turn and the rest is
        cancelled."""
        if not self.slow.pending:
            return
        try:
            await asyncio.wait_for(self.slow.drain(), timeout=self.watchdog.until_fire_s())
        except asyncio.TimeoutError:
            self.trace.note("drain_timeout", self.clock(), slow_pending=self.slow.pending)
            self.watchdog.fire("drain_timeout")
            self.slow.cancel_all()

    def parse(self, raw: Any) -> Event:
        if isinstance(raw, Event):
            return raw
        if isinstance(raw, dict):
            self.wire.observe(raw)
        return P.parse_event(raw, self.wire.unit)

    async def dispatch(self, ev: Event) -> None:
        if isinstance(self.clock, EventClock):
            self.clock.observe(ev)
        self._floor_on_event(ev)
        if self._is_user_turn(ev):
            self.turns += 1
        try:
            if ev.type == P.EV_TOOL_RESULT:
                self.on_result(P.parse_tool_result(ev))
            elif ev.type == P.EV_UNKNOWN:
                self.trace.note("ignored_event", ev.t, raw_type=ev.raw_type)
            else:
                await self.handler.on_event(ev)
        except Exception as exc:  # noqa: BLE001 - one bad event must not end the session (U24)
            log.exception("event handling failed: %s", ev)
            self.trace.note("handler_error", self.clock(), error=repr(exc), event_type=ev.type)

    async def drain(self) -> None:
        await self.slow.drain()

    # ------------------------------------------------------------------ actions
    def emit(self, action: Action) -> dict[str, Any] | None:
        """Stamp, attach snapshot, validate, trace, put on the wire. Every
        final response carries a valid snapshot (the current one if the
        policy did not supply its own)."""
        typ, fields = action_fields(action)
        snap = action.snapshot if isinstance(action, FinalResponse) and action.snapshot else self._snapshot()
        a = P.make_action(typ, self.clock(), snapshot=snap, ids=self.ids, **fields)
        errs = P.validate_action(a)
        if errs and typ == P.ACT_FINAL:  # a final must go out: retry once with a JSON-clean snapshot
            clean = _json_clean(snap)
            if not (isinstance(clean, dict) and "intent" in clean and isinstance(clean.get("slots"), dict)):
                clean = P.snapshot_payload(None, {})
            a = P.make_action(typ, self.clock(), snapshot=clean, ids=self.ids, **_json_clean(fields))
            errs = P.validate_action(a)
        if errs:  # never put a malformed payload on the wire
            self.trace.note("dropped_invalid_action", a["t"], errors=errs, action=repr(a))
            return None
        if typ == P.ACT_FINAL:
            self.answered = self.turns
        self.trace.append(a)
        if self._out is not None:
            self._out.put_nowait(P.to_wire(a, self.wire))
        sig = {P.ACT_SPEAK: "agent_speak", P.ACT_CLARIFY: "agent_speak", P.ACT_FINAL: "final",
               P.ACT_TOOL_CALL: "call_started"}.get(typ)
        if sig:
            self.coord.floor.signal(sig, a["t"], len(self.coord.inflight))
        return a

    def _snapshot(self) -> dict[str, Any]:
        """The policy's snapshot; if it raises, the coordinator's raw state,
        and failing that an empty but valid snapshot."""
        for fn in (self.snapshot_fn, self.coord.state.snapshot):
            try:
                snap = fn()
                if isinstance(snap, dict):
                    return snap
            except Exception:  # noqa: BLE001
                log.exception("snapshot failed")
        return P.snapshot_payload(None, {})

    # ------------------------------------------------------------------ calls
    def can_call(self, tool: ToolSpec, args: dict[str, Any]) -> str | None:
        return self.coord.check_issue(tool, args)

    def start_call(self, tool: ToolSpec, args: dict[str, Any], purpose: str = "goal",
                   intent: str | None = None, attempt: int = 1) -> Call | None:
        """Issue a non-blocking call: register (duplicates blocked + logged),
        emit tool_call, start its watcher task."""
        c = self.coord.issue(tool, args, purpose, intent, attempt)
        if c is None:
            return None
        self.emit(ToolCall(c.call_id, c.tool, c.args, c.generation, c.idem_key if c.state_modifying else None))
        self.trace.note("call_meta", self.clock(), call_id=c.call_id, generation=c.generation,
                        idempotency_key=c.idem_key, state_modifying=c.state_modifying)
        self._watchers[c.call_id] = asyncio.ensure_future(self._watch(c))
        return c

    def cancel_call(self, call_id: str, reason: str = "superseded") -> bool:
        """Mark cancelled and emit the cancel action synchronously (same clock
        tick as the triggering event, i.e. inside any grace period)."""
        c = self.coord.cancel(call_id, reason)
        if c is None:
            return False
        self.emit(Cancel(call_id, reason))
        w = self._watchers.pop(call_id, None)
        if w is not None:
            w.cancel()
        return True

    def reconcile(self, still_wanted: Callable[[Call], bool], reason: str = "superseded") -> tuple[list[Call], list[Call]]:
        """New plan generation: keep calls the plan still wants, cancel the rest
        immediately (same clock tick, before any speech)."""
        kept, invalid = self.coord.reconcile(still_wanted)
        for c in invalid:
            self.cancel_call(c.call_id, reason)
        return kept, invalid

    async def _watch(self, c: Call) -> None:
        """Per-call task. Ends when the call is resolved or cancelled; with
        TOOL_TIMEOUT_MS set it synthesises a retryable timeout (ASSUMPTION)."""
        try:
            if P.TOOL_TIMEOUT_MS is None:
                await c.token.wait()
                return
            await asyncio.wait_for(c.token.wait(), P.TOOL_TIMEOUT_MS / 1000.0)
        except asyncio.TimeoutError:
            if c.status == "inflight":
                self.trace.note("call_timeout", self.clock(), call_id=c.call_id)
                self.on_result(ToolResult(c.call_id, ok=False, error="timeout", retryable=not c.state_modifying))
        except asyncio.CancelledError:
            pass

    def on_result(self, tr: ToolResult) -> None:
        call, verdict = self.coord.resolve(tr)
        if call is not None:
            w = self._watchers.pop(call.call_id, None)
            if w is not None:
                w.cancel()
        if verdict not in (OK, FAILED) or call is None:
            return  # dropped and logged by the coordinator; never acted on
        self.coord.floor.signal("call_done", self.clock(), len(self.coord.inflight))
        self.handler.on_tool_result(call, tr)

    # ------------------------------------------------------------------ floor
    @staticmethod
    def _is_user_turn(ev: Event) -> bool:
        if ev.type == P.EV_EOT:
            return True
        if ev.type == P.EV_AUDIO:  # same end-of-turn rule as the agent's audio handler
            return P.is_end_of_turn(ev) or "end_of_turn" not in ev.payload
        if ev.type == P.EV_INTERRUPT:
            return bool(P.text_of(ev))
        return ev.type == P.EV_TEXT and P.is_end_of_turn(ev)

    def _floor_on_event(self, ev: Event) -> None:
        f, n = self.coord.floor, len(self.coord.inflight)
        if ev.type == P.EV_INTERRUPT:
            f.signal("interrupt", ev.t, n)
        elif ev.type in (P.EV_TEXT, P.EV_AUDIO, P.EV_EOT):
            f.signal("user_eot" if P.is_end_of_turn(ev) or ev.type == P.EV_EOT else "user_partial", ev.t, n)


def _json_clean(v: Any) -> Any:
    """Coerce to JSON-safe values: non-finite floats -> None, unknown objects -> str."""
    if isinstance(v, dict):
        return {str(k): _json_clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_json_clean(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return str(v)
