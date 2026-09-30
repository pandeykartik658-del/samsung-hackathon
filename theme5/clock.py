# /mnt/project-files/theme5/theme5/clock.py
"""Clocks. All agent timestamps are milliseconds on the harness timeline.

Decisions never read wall time (SPEC U06); wall time is only used to measure
our own processing latency for traces.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .protocol import Event


class Clock(Protocol):
    def __call__(self) -> float:
        """Current time in ms on the harness timeline."""


class VirtualClock:
    """Deterministic clock driven by the harness (local replay, tests)."""

    def __init__(self, start_ms: float = 0.0) -> None:
        self.now = float(start_ms)

    def __call__(self) -> float:
        return self.now

    def set(self, t_ms: float) -> None:
        if t_ms < self.now:
            raise ValueError(f"virtual time cannot go backwards ({t_ms} < {self.now})")
        self.now = float(t_ms)

    def advance(self, dt_ms: float) -> float:
        self.set(self.now + dt_ms)
        return self.now


class EventClock:
    """Clock for the real kit: latest event timestamp plus local elapsed time
    since that event arrived. ASSUMPTION (U06): the harness stamps actions on
    receipt, so ours are informative; they stay monotonic either way."""

    def __init__(self) -> None:
        self._base = 0.0
        self._at = time.perf_counter()

    def observe(self, ev: "Event") -> None:
        if ev.t >= self._base:
            self._base = ev.t
            self._at = time.perf_counter()

    def __call__(self) -> float:
        return self._base + (time.perf_counter() - self._at) * 1000.0


def wall_clock_ms() -> Clock:
    t0 = time.perf_counter()
    return lambda: (time.perf_counter() - t0) * 1000.0


class Stopwatch:
    """Wall-time measurement of our own processing (trace only)."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0
