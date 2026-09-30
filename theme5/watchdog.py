# /mnt/project-files/theme5/theme5/watchdog.py
"""Global scenario watchdog (guide §6: 120 s wall-clock cap per scenario).

Started by Engine.run(). At WATCHDOG_AT_S after session start it checks
whether the session is left hanging: a user turn with no final_response
since, a tool call still in flight, or slow-path work still running. If so it
cancels every in-flight call and slow task and emits one safe final_response
that carries the current (always valid) State Snapshot and says honestly what
did not finish. A session that is already answered and idle is left alone.

Time base: the running loop's clock, so the sim's virtual clock in replay and
monotonic wall time in the real kit. ASSUMPTION (SPEC U06): the kit's 120 s
cap is counted from session start, which we take to be our run() entry
(protocol.WATCHDOG_AT_S). The engine also bounds its end-of-session drain by
the same deadline so session_end can never hang on a stuck slow task.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from . import protocol as P
from .events import FinalResponse

if TYPE_CHECKING:
    from .coordinator import Call
    from .engine import Engine

log = logging.getLogger("theme5.watchdog")


def _human(tool: str) -> str:
    return tool.replace("_", " ").replace("-", " ").strip()


def _join(names: list[str]) -> str:
    names = list(dict.fromkeys(names))
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def safe_text(writes: list["Call"], reads: list["Call"], slow_pending: int) -> str:
    """Honest closing line: never claims a pending write happened or did not."""
    if writes:
        return (f"I've run out of time, so I stopped the {_join([_human(c.tool) for c in writes])} request before it "
                f"was confirmed. It may not have gone through, so please check before trying again.")
    if reads:
        return (f"I've run out of time before the {_join([_human(c.tool) for c in reads])} check finished, "
                f"so I'm stopping here. Please ask again if you still need it.")
    if slow_pending:
        return "I've run out of time before I could finish processing that, so I'm stopping here."
    return "I've run out of time on this request, so I'm stopping here."


class Watchdog:
    def __init__(self, engine: "Engine", at_s: float = P.WATCHDOG_AT_S,
                 cap_s: float = P.SCENARIO_WALL_CAP_S) -> None:
        self.engine = engine
        self.at_s = at_s
        self.cap_s = cap_s
        self.fired = False
        self.fired_action: dict[str, Any] | None = None
        self._t0: float | None = None
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------ clock
    def _now(self) -> float:
        return asyncio.get_running_loop().time()

    def elapsed_s(self) -> float:
        return 0.0 if self._t0 is None else self._now() - self._t0

    def remaining_s(self) -> float:
        """Seconds left before the hard cap (for slow-path budgets)."""
        return max(0.0, self.cap_s - self.elapsed_s())

    def until_fire_s(self) -> float:
        return max(0.0, self.at_s - self.elapsed_s())

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._task is not None:
            return
        self._t0 = self._now()
        self._task = asyncio.ensure_future(self._run())

    def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def _run(self) -> None:
        try:
            await asyncio.sleep(self.at_s)
        except asyncio.CancelledError:
            return
        self.fire("time_nearly_up")

    # ------------------------------------------------------------------ firing
    def needs_final(self) -> bool:
        e = self.engine
        return e.turns_unanswered or bool(e.coord.inflight) or e.slow.pending > 0

    def fire(self, reason: str = "time_nearly_up") -> dict[str, Any] | None:
        """One shot. Returns the emitted final_response, or None if nothing
        was hanging. Never raises."""
        if self.fired:
            return None
        self.fired = True
        e = self.engine
        try:
            hanging = self.needs_final()
            e.trace.note("watchdog_fired", e.clock(), reason=reason, elapsed_s=round(self.elapsed_s(), 3),
                         inflight=list(e.coord.inflight), slow_pending=e.slow.pending, emitted_final=hanging)
            if not hanging:
                return None
            inflight = list(e.coord.inflight.values())
            slow = e.slow.pending
            for c in inflight:
                e.cancel_call(c.call_id, "watchdog")
            e.slow.cancel_all()
            text = safe_text([c for c in inflight if c.state_modifying],
                             [c for c in inflight if not c.state_modifying], slow)
            self.fired_action = e.emit(FinalResponse(text))
            return self.fired_action
        except Exception:  # noqa: BLE001 - the safety net must not become the crash
            log.exception("watchdog failed")
            return None
