# /mnt/project-files/theme5/tests/test_clock_trace_slowpath_events.py
import asyncio
import json

import pytest

from theme5 import events as E
from theme5.clock import EventClock, Stopwatch, VirtualClock
from theme5.coordinator import CancelToken, Floor, FloorState, idempotency_key
from theme5.protocol import parse_event
from theme5.slowpath import SlowPath
from theme5.trace import TraceLog


def test_virtual_clock_monotonic():
    c = VirtualClock()
    c.set(10)
    assert c() == 10 and c.advance(5) == 15
    with pytest.raises(ValueError):
        c.set(1)


def test_event_clock_tracks_latest_event():
    c = EventClock()
    c.observe(parse_event({"type": "text", "t": 500}))
    assert 500 <= c() < 600
    c.observe(parse_event({"type": "text", "t": 100}))  # out-of-order event never moves time back
    assert c() >= 500
    assert Stopwatch().ms() >= 0


def test_trace_log(tmp_path):
    t = TraceLog()
    t.note("x", 1, a=1)
    t.append({"type": "speak", "t": 2})
    assert t.of("x")[0]["a"] == 1 and t.actions() == [{"type": "speak", "t": 2}]
    t.dump(tmp_path / "t.jsonl")
    assert [json.loads(l)["type"] for l in (tmp_path / "t.jsonl").read_text().splitlines()] == ["_x", "speak"]


def test_slowpath_inline_and_background():
    async def go():
        sp = SlowPath()

        async def fast():
            return 1

        async def slow():
            await asyncio.sleep(0.05)
            return 2

        ok1, f1 = await sp.inline_or_background(fast(), 0.2)
        ok2, f2 = await sp.inline_or_background(slow(), 0.001)
        pending = sp.pending
        await sp.drain()

        async def boom():
            raise RuntimeError("x")

        sp.spawn(boom(), name="boom")
        await sp.drain()
        sp.spawn(asyncio.sleep(10))
        cancelled = sp.cancel_all()
        return ok1, f1.result(), ok2, pending, f2.result(), sp.trace, cancelled

    ok1, r1, ok2, pending, r2, trace, cancelled = asyncio.run(go())
    assert ok1 and r1 == 1 and not ok2 and pending == 1 and r2 == 2
    assert trace.of("slow_task_failed") and cancelled == 1


def test_typed_events_cover_guide_inputs():
    cases = {
        "tool_manifest": E.ToolManifest, "text": E.TextChunk, "end_of_turn": E.EndOfTurn, "audio": E.AudioClip,
        "frame": E.VideoFrame, "interrupt": E.Interruption, "tool_result": E.ToolResultEvent,
        "end_session": E.SessionEnd, "bogus": E.UnknownEvent,
    }
    for typ, cls in cases.items():
        assert isinstance(E.typed(parse_event({"type": typ, "t": 1, "tools": []})), cls), typ
    a = E.typed(parse_event({"type": "audio", "t": 0, "clip_id": "c", "transcript": "hi", "end_of_turn": False}))
    assert a.ref == "c" and a.transcript == "hi" and not a.end_of_turn


def test_action_fields_cover_guide_outputs():
    assert E.action_fields(E.Speak("x", "filler"))[0] == "speak"
    typ, f = E.action_fields(E.ToolCall("call-1", "t", {"a": 1}, 3, "k"))
    assert typ == "tool_call" and f["call_id"] == "call-1" and "generation" not in f  # EMIT_CALL_META off
    assert E.action_fields(E.Cancel("call-1"))[1]["reason"] == "superseded"
    assert E.action_fields(E.Clarify("which?", "slot"))[1]["slot"] == "slot"
    assert E.action_fields(E.FinalResponse("done"))[0] == "final_response"
    assert E.StateSnapshot("i", {"a": None, "b": 1}).as_dict() == {"intent": "i", "slots": {"b": 1}}
    with pytest.raises(TypeError):
        E.action_fields("nope")


def test_idempotency_key_normalises_args():
    assert idempotency_key("book", {"a": " X ", "b": 1}) == idempotency_key("book", {"b": 1, "a": "x"})
    assert idempotency_key("book", {"a": "x"}) != idempotency_key("book", {"a": "y"})
    assert len(idempotency_key("book", {})) == 16


def test_cancel_token_and_floor_signals():
    async def go():
        tok = CancelToken()
        waiter = asyncio.ensure_future(tok.wait())
        tok.cancel("r1")
        tok.cancel("r2")  # first reason wins
        return await waiter, tok.cancelled

    assert asyncio.run(go()) == ("r1", True)
    f = Floor()
    assert f.signal("user_eot", 0, calls_inflight=2) is FloorState.WAITING_ON_TOOLS
    assert f.signal("final", 1, 0) is FloorState.IDLE
