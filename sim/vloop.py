# sim/vloop.py
"""Virtual-time asyncio event loop.

`loop.time()` returns a virtual clock. Whenever the loop would block waiting
for the next timer and no I/O is ready, the clock jumps straight to that timer.
Every `asyncio.sleep`, `call_later`, `wait_for` timeout etc. therefore runs on
virtual time, so a scenario replays deterministically and far faster than real
time, without the agent having to know it is being simulated.

Options
- charge_compute: add the real wall time spent inside callbacks to the virtual
  clock, so slow agent code shows up as latency (non-deterministic by nature).
- wall_cap_s: hard real-time cap (guide section 6: 120 s per scenario). When
  exceeded, `on_wall_cap` is called once from inside the loop.

Executor threads (run_in_executor) are tracked: while any are running, the loop
waits in real time instead of jumping, so off-loop work is not skipped over.
"""
from __future__ import annotations

import asyncio
import selectors
import time
from typing import Callable, Optional


class _VirtualSelector:
    def __init__(self, loop: "VirtualTimeLoop", real: selectors.BaseSelector):
        self._loop = loop
        self._real = real

    def select(self, timeout: Optional[float] = None):
        loop = self._loop
        loop._charge()
        loop._check_wall()
        events = self._real.select(0)
        if events or timeout == 0:
            loop._mark()
            return events
        if loop._executor_jobs > 0:
            # Off-loop work in progress: wait in real time, advance by elapsed.
            wait = 0.005 if timeout is None else min(timeout, 0.005)
            start = time.perf_counter()
            events = self._real.select(wait)
            loop._vt += time.perf_counter() - start
            loop._mark()
            return events
        if timeout is None:
            # Nothing scheduled at all: only I/O or threads can wake us.
            events = self._real.select(0.01)
            loop._mark()
            return events
        loop._advance_to_next_timer(timeout)
        loop._mark()
        return []

    def __getattr__(self, name):
        return getattr(self._real, name)


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    def __init__(self, charge_compute: bool = False, wall_cap_s: Optional[float] = None,
                 on_wall_cap: Optional[Callable[[], None]] = None):
        super().__init__()
        self._vt = 0.0
        self._selector = _VirtualSelector(self, self._selector)
        self.charge_compute = charge_compute
        self.wall_cap_s = wall_cap_s
        self.on_wall_cap = on_wall_cap
        self.wall_cap_hit = False
        self._wall_start = time.perf_counter()
        self._last_mark = time.perf_counter()
        self._executor_jobs = 0

    # -- clock ---------------------------------------------------------------
    def time(self) -> float:
        return self._vt

    def _mark(self) -> None:
        self._last_mark = time.perf_counter()

    def _charge(self) -> None:
        if self.charge_compute:
            self._vt += max(0.0, time.perf_counter() - self._last_mark)

    def _advance_to_next_timer(self, timeout: float) -> None:
        target = self._vt + timeout
        # Land exactly on the earliest timer to avoid float drift.
        sched = [h for h in self._scheduled if not h._cancelled]
        if sched:
            target = max(self._vt, min(h._when for h in sched))
        self._vt = max(self._vt, target)

    def _check_wall(self) -> None:
        if self.wall_cap_s is None or self.wall_cap_hit:
            return
        if time.perf_counter() - self._wall_start > self.wall_cap_s:
            self.wall_cap_hit = True
            if self.on_wall_cap is not None:
                self.call_soon(self.on_wall_cap)

    def wall_elapsed(self) -> float:
        return time.perf_counter() - self._wall_start

    # -- executor tracking ---------------------------------------------------
    def run_in_executor(self, executor, func, *args):
        fut = super().run_in_executor(executor, func, *args)
        self._executor_jobs += 1

        def _done(_):
            self._executor_jobs -= 1

        fut.add_done_callback(_done)
        return fut


def run_virtual(coro, charge_compute: bool = False, wall_cap_s: Optional[float] = None):
    """Run `coro` to completion on a fresh VirtualTimeLoop. Returns its result."""
    loop = VirtualTimeLoop(charge_compute=charge_compute, wall_cap_s=wall_cap_s)
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()
