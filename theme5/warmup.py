# /mnt/project-files/theme5/theme5/warmup.py
"""Process-wide warm-up: run one throwaway mini-session through a real Agent
so the first real user turn does not pay for lazy regex compilation, first-call
code paths and import side effects.

Profiling (bench/profile_suite.py) showed the first utterance of a process
took ~7 ms on local disk and up to ~440 ms on the shared network mount, versus
~0.7 ms warm. `warm_up()` is called from Agent.setup() (the kit's 300 s
warm-up hook) and once per process from Agent construction, so it also helps
harnesses that never call setup(). It never raises and runs at most once.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

log = logging.getLogger("theme5.warmup")

_done = False
_lock = threading.Lock()
last_ms: float | None = None  # wall time the warm-up took (for traces / tests)

_MANIFEST = {"type": "tool_manifest", "t": 0, "tools": [
    {"name": "flight_search", "description": "Search flights between two airports on a date.", "read_only": True,
     "parameters": {"type": "object", "required": ["origin", "destination", "date"], "properties": {
         "origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"},
         "passengers": {"type": "integer"}}}},
    {"name": "book_flight", "description": "Book a seat on a flight. State-modifying.", "read_only": False,
     "parameters": {"type": "object", "required": ["flight_id", "passenger_name"], "properties": {
         "flight_id": {"type": "string"}, "passenger_name": {"type": "string"}}}},
    {"name": "manual_lookup", "description": "Look up a device manual section.", "read_only": True,
     "parameters": {"type": "object", "required": ["device_model", "query"], "properties": {
         "device_model": {"type": "string"}, "query": {"type": "string"}}}},
]}

_EVENTS: list[dict[str, Any]] = [
    _MANIFEST,
    {"type": "text_chunk", "t": 10, "text": "Search flights from Delhi to", "end_of_turn": False},
    {"type": "text_chunk", "t": 20, "text": "Search flights from Delhi to Mumbai tomorrow for 2 people.",
     "end_of_turn": True},
    {"type": "interrupt", "t": 30, "text": "Uh, no wait, make it Goa on 5 October."},
    {"type": "tool_result", "t": 40, "call_id": "call-0002", "ok": True,
     "result": {"flights": [{"flight_id": "AI-101", "price": 4500}, {"flight_id": "6E-22", "price": 3900}]}},
    {"type": "text_chunk", "t": 50, "text": "Book the cheapest one for Ananya Das.", "end_of_turn": True},
    {"type": "tool_result", "t": 60, "call_id": "call-0003", "ok": False, "status": "error",
     "error": "unavailable", "retryable": True},
    {"type": "audio_clip", "t": 70, "transcript": "Actually cancel everything.", "confidence": 0.9,
     "end_of_turn": True},
    {"type": "video_frame", "t": 80, "labels": [{"label": "washing_machine", "model": "WW90T", "confidence": 0.9,
                                                 "text": "4C"}]},
    {"type": "text_chunk", "t": 90, "text": "What does this error mean and how do I fix it?", "end_of_turn": True},
    {"type": "mystery_event", "t": 95},
    {"type": "end_session", "t": 100},
]


async def _session() -> None:
    from .agent import Agent  # local: agent imports this module

    agent = Agent(warm=False)
    inbox: asyncio.Queue[Any] = asyncio.Queue()
    outbox: asyncio.Queue[Any] = asyncio.Queue()
    for ev in _EVENTS:
        inbox.put_nowait(ev)
    try:
        await asyncio.wait_for(agent.run(inbox, outbox), 2.0)
    except asyncio.TimeoutError:
        pass


def _run_private_loop() -> None:
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_session())
    finally:
        loop.close()


def warm_up() -> float | None:
    """Run the mini-session once per process. Safe from sync code and from
    inside a running loop (then it runs on a short-lived thread). Returns the
    wall ms it took, or None if it already ran."""
    global _done, last_ms
    with _lock:
        if _done:
            return None
        _done = True
    t0 = time.perf_counter()
    try:
        try:
            asyncio.get_running_loop()
            inside_loop = True
        except RuntimeError:
            inside_loop = False
        if inside_loop:
            th = threading.Thread(target=_run_private_loop, name="theme5-warmup", daemon=True)
            th.start()
            th.join(5.0)
        else:
            _run_private_loop()
    except Exception:  # noqa: BLE001 - warm-up is an optimisation, never a failure
        log.debug("warm-up failed", exc_info=True)
    last_ms = (time.perf_counter() - t0) * 1000.0
    return last_ms
