# /mnt/project-files/theme5/tests/test_hardening.py
"""Hardening: protocol fuzzing (the agent must never crash), the global
watchdog, the bounded end-of-session drain, and the process warm-up."""
from __future__ import annotations

import asyncio
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim import harness, scenario  # noqa: E402
from sim.adapter import load_agent_factory, load_codec  # noqa: E402
from sim.vloop import run_virtual  # noqa: E402
from theme5 import protocol as P  # noqa: E402
from theme5 import warmup  # noqa: E402
from theme5.agent import Agent  # noqa: E402
from theme5.engine import _json_clean  # noqa: E402
from theme5.watchdog import Watchdog, safe_text  # noqa: E402

MANIFEST = {"type": "tool_manifest", "t": 0, "tools": [
    {"name": "flight_search", "description": "Search flights between two airports on a date.", "read_only": True,
     "parameters": {"type": "object", "required": ["origin", "destination", "date"], "properties": {
         "origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}}}},
    {"name": "book_flight", "description": "Book a seat on a flight. State-modifying.", "read_only": False,
     "parameters": {"type": "object", "required": ["flight_id", "passenger_name"], "properties": {
         "flight_id": {"type": "string"}, "passenger_name": {"type": "string"}}}},
], "date": "2026-09-29"}

SEARCH = {"type": "text_chunk", "t": 100, "text": "Search flights from Delhi to Mumbai tomorrow.", "end_of_turn": True}
BOOK = {"type": "text_chunk", "t": 100, "text": "Book AI-5828 for Ananya Das.", "end_of_turn": True}


# ---------------------------------------------------------------- helpers
def _actions(agent: Agent) -> list[dict[str, Any]]:
    return agent.trace.actions()


def _assert_all_valid(agent: Agent) -> None:
    for a in _actions(agent):
        assert P.validate_action(a) == [], a
        json.dumps(a, allow_nan=False)
        snap = a.get("state_snapshot")
        assert isinstance(snap, dict) and "intent" in snap and isinstance(snap["slots"], dict)


async def _session(events: list[Any], agent: Agent | None = None, settle: float = 0.05) -> Agent:
    agent = agent or Agent(warm=False)
    inbox: asyncio.Queue[Any] = asyncio.Queue()
    outbox: asyncio.Queue[Any] = asyncio.Queue()
    task = asyncio.ensure_future(agent.run(inbox, outbox))
    for ev in events:
        inbox.put_nowait(ev)
        await asyncio.sleep(0)
        assert not task.done() or task.exception() is None, f"agent died on {ev!r}"
    await asyncio.sleep(settle)
    inbox.put_nowait(None)
    await asyncio.wait_for(task, 5)
    assert task.exception() is None
    return agent


# ---------------------------------------------------------------- fuzz: parser
GARBAGE: list[Any] = [
    1, 3.5, True, "x", "{not json", "[1,2]", "null", [1], (), b"bytes", b"\xff\xfe", bytearray(b"{}"), object(),
    {}, {"type": None}, {"type": 5}, {"type": ["text"]}, {"type": {"a": 1}},
    {"type": "text", "t": "abc"}, {"type": "text", "t": {"a": 1}}, {"type": "text", "t": [1]},
    {"type": "text", "t": float("nan"), "text": "hi"}, {"type": "text", "t": float("inf")},
    {"type": "text", "t": -float("inf")}, {"type": "text", "t": 1e400}, {"type": "text", "t": True},
    {"type": "text", "t": "2026-13-45T99:99:99Z"}, {"type": "text", "t": 10**400},
]


@pytest.mark.parametrize("raw", GARBAGE, ids=lambda r: repr(r)[:30])
def test_parse_event_never_raises(raw: Any) -> None:
    ev = P.parse_event(raw)
    assert isinstance(ev.t, float) and math.isfinite(ev.t)
    assert isinstance(ev.type, str)


def test_non_objects_become_unknown_events() -> None:
    for raw in (1, "[1,2]", "{not json", b"\xff", [1]):
        assert P.parse_event(raw).type == P.EV_UNKNOWN


@pytest.mark.parametrize("v", [None, "abc", {"a": 1}, [1], float("nan"), float("inf"), 1e400, True, "", " "])
def test_to_ms_is_finite_and_total(v: Any) -> None:
    out = P.to_ms(v)
    assert isinstance(out, float) and math.isfinite(out)


def test_to_ms_keeps_valid_inputs() -> None:
    assert P.to_ms(1.5, "s") == 1500.0 and P.to_ms("250") == 250.0
    assert P.to_ms("1970-01-01T00:00:01Z") == 1000.0


def test_validate_action_rejects_nan() -> None:
    a = P.make_action(P.ACT_SPEAK, 1.0, snapshot=P.snapshot_payload(None, {}), text="hi", kind=P.SPEAK_ACK)
    assert P.validate_action(a) == []
    a["t"] = float("nan")
    assert "not JSON-serialisable" in P.validate_action(a)


# ---------------------------------------------------------------- fuzz: agent
def _mutations(rng: random.Random) -> list[Any]:
    types = ["text_chunk", "text", "audio_clip", "video_frame", "interrupt", "tool_result", "tool_manifest",
             "end_of_turn", "mystery", "", None, 7]
    values: list[Any] = [None, "", "x", 0, -1, 1e18, float("nan"), True, [], {}, [1, "a"], {"k": [None]},
                         "book it", "a " * 5000, "\u0000퟿\U0001f600", "<EOT>"]
    keys = ["t", "ts", "text", "transcript", "end_of_turn", "call_id", "result", "error", "status", "ok",
            "labels", "confidence", "tools", "payload", "data", "id", "path", "retryable", "cumulative"]
    out: list[Any] = []
    for _ in range(400):
        roll = rng.random()
        if roll < 0.1:
            out.append(rng.choice(GARBAGE))
            continue
        ev: dict[str, Any] = {}
        if rng.random() < 0.9:
            ev["type"] = rng.choice(types)
        for k in rng.sample(keys, rng.randint(0, 6)):
            ev[k] = rng.choice(values)
        if rng.random() < 0.2:
            ev["labels"] = [rng.choice([{"confidence": rng.choice(values), "label": rng.choice(values)}, "tv", 3])]
        if rng.random() < 0.1:
            ev["tools"] = [rng.choice([1, None, {"name": None}, {"name": "x", "parameters": "bad"},
                                       MANIFEST["tools"][0]])]
        out.append(ev)
    return out


@pytest.mark.parametrize("seed", range(6))
def test_agent_survives_fuzzed_event_stream(seed: int) -> None:
    rng = random.Random(seed)

    async def main() -> Agent:
        return await _session([MANIFEST] + _mutations(rng) + [SEARCH])

    agent = asyncio.run(main())
    _assert_all_valid(agent)
    # still functional after the garbage: the valid search at the end was handled
    assert any(a["type"] == P.ACT_TOOL_CALL and a.get("tool") == "flight_search" for a in _actions(agent)) \
        or any(a["type"] in (P.ACT_CLARIFY, P.ACT_FINAL) for a in _actions(agent))


def test_every_malformed_kind_is_skipped_not_fatal() -> None:
    async def main() -> Agent:
        return await _session([MANIFEST] + GARBAGE + [
            {"type": "tool_result"}, {"type": "tool_result", "call_id": None, "error": {"x": 1}},
            {"type": "tool_result", "call_id": "nope", "ok": True},
            {"type": "audio_clip", "confidence": "high", "transcript": "book"},
            {"type": "video_frame", "labels": [{"confidence": "x"}]},
            {"type": "tool_manifest", "tools": "nope"}, {"type": "tool_manifest", "tools": [1, None, {"name": None}]},
            {"type": "interrupt", "t": -5}, SEARCH])

    agent = asyncio.run(main())
    _assert_all_valid(agent)
    assert any(a["type"] == P.ACT_TOOL_CALL for a in _actions(agent))


def test_out_of_order_timestamps_keep_action_times_monotonic() -> None:
    evs = [MANIFEST,
           {"type": "text_chunk", "t": 5000, "text": "Search flights from Delhi to Mumbai tomorrow.", "end_of_turn": True},
           {"type": "text_chunk", "t": 10, "text": "No wait, make it Goa.", "end_of_turn": True},
           {"type": "interrupt", "t": 1, "text": "Actually Chennai."},
           {"type": "tool_result", "t": -100, "call_id": "call-0001", "ok": True, "result": {"flights": []}}]
    agent = asyncio.run(_session(evs))
    ts = [a["t"] for a in _actions(agent)]
    assert ts and ts == sorted(ts) and min(ts) >= 5000
    _assert_all_valid(agent)


def test_huge_utterance_does_not_block_the_loop() -> None:
    import time

    big = {"type": "text_chunk", "t": 1, "text": "flights from Delhi to Mumbai " * 4000, "end_of_turn": True}
    t0 = time.perf_counter()
    agent = asyncio.run(_session([MANIFEST, big]))
    assert time.perf_counter() - t0 < 5.0
    _assert_all_valid(agent)


def test_snapshot_failure_still_yields_valid_final() -> None:
    async def main() -> Agent:
        agent = Agent(warm=False)

        def boom() -> dict[str, Any]:
            raise RuntimeError("snapshot bug")

        agent.engine.snapshot_fn = boom
        agent.state.slots["weird"] = float("nan")
        return await _session([MANIFEST, {"type": "text_chunk", "t": 1, "text": "cancel everything",
                                          "end_of_turn": True}], agent)

    agent = asyncio.run(main())
    finals = [a for a in _actions(agent) if a["type"] == P.ACT_FINAL]
    assert finals
    _assert_all_valid(agent)


def test_json_clean() -> None:
    assert _json_clean({"a": float("nan"), 1: (1, {2}), "o": object()})["a"] is None
    assert json.dumps(_json_clean({"x": [float("inf"), b"b"]}), allow_nan=False)


# ---------------------------------------------------------------- watchdog
async def _run_until(agent: Agent, events: list[tuple[float, Any]], until_s: float) -> None:
    """Deliver (t_s, event) on the virtual clock, then end the session at until_s."""
    inbox: asyncio.Queue[Any] = asyncio.Queue()
    outbox: asyncio.Queue[Any] = asyncio.Queue()
    task = asyncio.ensure_future(agent.run(inbox, outbox))
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    for t_s, ev in events:
        await asyncio.sleep(max(0.0, t0 + t_s - loop.time()))
        inbox.put_nowait(ev)
    await asyncio.sleep(max(0.0, t0 + until_s - loop.time()))
    inbox.put_nowait(None)
    await task


def test_watchdog_closes_hanging_read_with_safe_final() -> None:
    agent = Agent(warm=False)
    run_virtual(_run_until(agent, [(0, MANIFEST), (0.1, SEARCH)], until_s=119))
    acts = _actions(agent)
    call = next(a for a in acts if a["type"] == P.ACT_TOOL_CALL)
    cancel = next(a for a in acts if a["type"] == P.ACT_CANCEL)
    final = [a for a in acts if a["type"] == P.ACT_FINAL]
    assert cancel["call_id"] == call["call_id"] and cancel["reason"] == "watchdog"
    assert len(final) == 1 and "run out of time" in final[0]["text"]
    assert final[0]["state_snapshot"]["slots"].get("destination") == "Mumbai"
    assert agent.engine.watchdog.fired and agent.trace.of("watchdog_fired")
    _assert_all_valid(agent)


def test_watchdog_is_honest_about_pending_write() -> None:
    agent = Agent(warm=False)
    run_virtual(_run_until(agent, [(0, MANIFEST), (0.1, BOOK)], until_s=119))
    final = [a for a in _actions(agent) if a["type"] == P.ACT_FINAL]
    assert len(final) == 1
    assert "may not have gone through" in final[0]["text"] and "book flight" in final[0]["text"]


def test_watchdog_stays_quiet_when_answered_and_idle() -> None:
    agent = Agent(warm=False)
    result = {"type": "tool_result", "call_id": "call-0001", "ok": True,
              "result": {"flights": [{"flight_id": "AI-1", "price": 100}]}}
    run_virtual(_run_until(agent, [(0, MANIFEST), (0.1, SEARCH), (1.0, result)], until_s=119))
    finals = [a for a in _actions(agent) if a["type"] == P.ACT_FINAL]
    assert len(finals) == 1 and "run out of time" not in finals[0]["text"]
    notes = agent.trace.of("watchdog_fired")
    assert notes and notes[0]["emitted_final"] is False


def test_watchdog_not_reached_in_short_session() -> None:
    agent = Agent(warm=False)
    run_virtual(_run_until(agent, [(0, MANIFEST), (0.1, SEARCH)], until_s=20))
    assert not agent.engine.watchdog.fired
    assert not [a for a in _actions(agent) if a["type"] == P.ACT_FINAL]


def test_watchdog_fires_once() -> None:
    async def main() -> Agent:
        agent = Agent(warm=False)
        agent.engine.watchdog.start()
        agent.engine.turns = 1
        first = agent.engine.watchdog.fire()
        second = agent.engine.watchdog.fire()
        agent.engine.watchdog.stop()
        assert first is not None and first["type"] == P.ACT_FINAL and second is None
        return agent

    asyncio.run(main())


def test_watchdog_budget_helpers() -> None:
    async def main() -> None:
        agent = Agent(warm=False)
        wd = Watchdog(agent.engine, at_s=10, cap_s=12)
        wd.start()
        await asyncio.sleep(4)
        assert 5.9 <= wd.until_fire_s() <= 6.0 and 7.9 <= wd.remaining_s() <= 8.0
        wd.stop()

    run_virtual(main())


def test_safe_text_variants() -> None:
    class C:
        def __init__(self, tool: str, w: bool) -> None:
            self.tool, self.state_modifying = tool, w

    assert "may not have gone through" in safe_text([C("book_flight", True)], [], 0)
    assert "flight search check" in safe_text([], [C("flight_search", False)], 0)
    assert "processing" in safe_text([], [], 2)
    assert "run out of time" in safe_text([], [], 0)


class _StuckPerception:
    """Transcription that never finishes (a hung ASR backend)."""

    async def transcribe(self, ev: Any) -> tuple[str | None, float]:
        await asyncio.sleep(10_000)
        return None, 0.0

    async def describe(self, ev: Any) -> tuple[str | None, float]:
        return None, 0.0


def test_session_end_drain_is_bounded_by_watchdog() -> None:
    async def main() -> tuple[Agent, float]:
        agent = Agent(warm=False, perception=_StuckPerception())
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await _run_until(agent, [(0, MANIFEST), (0.1, {"type": "audio_clip", "t": 100, "path": "x.wav",
                                                        "end_of_turn": True})], until_s=1)
        return agent, loop.time() - t0

    agent, took = run_virtual(main())
    assert took <= P.WATCHDOG_AT_S + 1  # without the bound, drain waits 10 000 s
    assert agent.trace.of("drain_timeout")
    finals = [a for a in _actions(agent) if a["type"] == P.ACT_FINAL]
    assert len(finals) == 1 and "processing" in finals[0]["text"]
    assert agent.engine.slow.pending == 0


# ---------------------------------------------------------------- end to end on the harness
def test_harness_hanging_tool_ends_with_watchdog_final() -> None:
    sc = scenario.from_dict({
        "id": "hx_hanging_tool", "modality": "text", "duration_ms": 119_000,
        "tools": {"enabled": ["flight_search"], "config": {"flight_search": {"latency_ms": 500_000}}},
        "events": [{"id": "e1", "t_ms": 100, "type": "text_chunk", "end_of_turn": True,
                    "text": "Search flights from Delhi to Mumbai tomorrow."}],
        "expected": {"required_calls": [], "forbidden_calls": [], "final_snapshot": {}},
    })
    res = harness.run_scenario(sc, load_agent_factory("theme5.agent:Agent"), load_codec(None, "theme5.agent:Agent"))
    outs = [r for r in res.records if r["dir"] == "out"]
    finals = [r for r in outs if r["kind"] in ("final", "final_response")]
    assert res.agent_error is None and res.end_reason == "quiescent"
    assert len(finals) == 1 and abs(finals[0]["t_ms"] - P.WATCHDOG_AT_S * 1000) < 50
    assert any(r["kind"] == "tool_cancelled" for r in res.records)
    assert not [r for r in outs if r["kind"] == "invalid"]


# ---------------------------------------------------------------- warm-up
def test_warm_up_runs_once_and_never_raises() -> None:
    warmup.warm_up()  # may or may not be the first call in this process
    assert warmup._done and warmup.last_ms is not None
    assert warmup.warm_up() is None


def test_warm_up_inside_running_loop() -> None:
    async def main() -> None:
        warmup._done = False
        took = warmup.warm_up()
        assert took is not None and took < 5000

    asyncio.run(main())


def test_setup_calls_warm_up() -> None:
    warmup._done = False
    asyncio.run(Agent(warm=False).setup())
    assert warmup._done
