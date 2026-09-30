# /mnt/project-files/theme5/theme5/slowpath.py
"""Slow path: background work that must never block the fast path
(ASR, vision, optional LLM). Work that finishes within a small budget is
awaited inline so utterances keep event order; slower work is acknowledged
by the caller and continues as a tracked task."""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Awaitable, Callable, TypeVar

from .trace import TraceLog

log = logging.getLogger("theme5.slowpath")
T = TypeVar("T")

INLINE_BUDGET_S = 0.25  # ASSUMPTION: beyond this, acknowledge and continue in background


_HOLD_RE = re.compile(
    r"(?:\b(?:u+m+|u+h+|e+r+m*|h+m+|no\s*,?\s*wait|wait|actually|sorry|i\s+mean|or|and|to|from|on|for|at|in|the|a|an)"
    r"|[,\-])\s*[.!?]?\s*$", re.I)


def speculation_ok(partial_text: str) -> bool:
    """Speculating on a partial turn is safe only when the user is not
    mid-phrase: no trailing hesitation, repair marker, connective or comma
    ("...to Mumbai, uh, no wait" must not launch a Mumbai search)."""
    t = partial_text.strip()
    return bool(t) and not _HOLD_RE.search(t)


class SlowPath:
    def __init__(self, trace: TraceLog | None = None, clock: Callable[[], float] | None = None) -> None:
        self.trace = trace if trace is not None else TraceLog()
        self.clock = clock or (lambda: 0.0)
        self._tasks: set[asyncio.Task[Any]] = set()

    @property
    def pending(self) -> int:
        return len(self._tasks)

    def spawn(self, coro: Awaitable[T], name: str = "slow") -> "asyncio.Task[T]":
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._done(name))
        return task

    def _done(self, name: str) -> Callable[["asyncio.Task[Any]"], None]:
        def cb(t: "asyncio.Task[Any]") -> None:
            self._tasks.discard(t)
            if not t.cancelled() and t.exception() is not None:
                self.trace.note("slow_task_failed", self.clock(), task=name, error=repr(t.exception()))
        return cb

    async def inline_or_background(self, coro: Awaitable[T], budget_s: float = INLINE_BUDGET_S,
                                   name: str = "slow") -> tuple[bool, "asyncio.Future[T]"]:
        """Run `coro`; wait at most `budget_s`. Returns (finished_inline, future).
        If not finished, the task is tracked and keeps running."""
        fut = asyncio.ensure_future(coro)
        done, _ = await asyncio.wait({fut}, timeout=budget_s)
        if done:
            return True, fut
        self._tasks.add(fut)
        fut.add_done_callback(self._done(name))
        return False, fut

    async def drain(self) -> None:
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    def cancel_all(self) -> int:
        n = 0
        for t in list(self._tasks):
            if not t.done():
                t.cancel()
                n += 1
        return n
