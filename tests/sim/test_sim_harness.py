# tests/sim/test_sim_harness.py
import asyncio
import json

import pytest

from sim.harness import HarnessConfig, run_scenario
from sim.scenario import from_dict
from sim.trace import read_jsonl
from sim_agents import ScriptAgent, call, cancel, factory, final, say


def make(events, tools=None, expected=None, duration_ms=10000, **kw):
    d = {"id": "t", "modality": "text", "duration_ms": duration_ms, "events": events,
         "session": {"reference_date": "2026-09-29"}, "tools": tools or {},
         "expected": expected or {"required_calls": [], "forbidden_calls": [], "final_snapshot": None}}
    d.update(kw)
    return from_dict(d)


FS = dict(origin="DEL", destination="BOM", date="2026-10-12")
TXT = {"id": "e1", "t_ms": 100, "type": "text_chunk", "text": "hi", "end_of_turn": True}


def kinds(res, direction=None):
    return [(r["dir"], r["kind"]) for r in res.records if direction is None or r["dir"] == direction]


def test_manifest_first_with_date_and_read_only():
    res = run_scenario(make([TXT]), factory({"e1": [final("ok", None)]}))
    first_in = next(r for r in res.records if r["dir"] == "in")
    assert first_in["kind"] == "tool_manifest" and first_in["t_ms"] == 0
    assert first_in["data"]["date"] == "2026-09-29"
    assert all("read_only" in t for t in first_in["data"]["tools"])
    assert res.end_reason == "quiescent"


def test_events_replayed_at_exact_virtual_times_in_script_order():
    evs = [{"id": "a", "t_ms": 1500, "type": "interrupt", "reason": "barge_in"},
           {"id": "b", "t_ms": 1500, "type": "text_chunk", "text": "x", "end_of_turn": True},
           {"id": "c", "t_ms": 2750, "type": "text_chunk", "text": "y", "end_of_turn": True}]
    res = run_scenario(make(evs), factory({"c": [final("done", None)]}))
    got = [(r["data"]["event_id"], r["t_ms"], r["data"]["t"]) for r in res.records
           if r["dir"] == "in" and r["data"].get("event_id") in ("a", "b", "c")]
    assert got == [("a", 1500, 1500), ("b", 1500, 1500), ("c", 2750, 2750)]


def test_tool_call_latency_and_result():
    tools = {"config": {"flight_search": {"latency_ms": 1234}}}
    res = run_scenario(make([TXT], tools), factory({
        "e1": [call("c1", "flight_search", **FS)], "result:c1:ok": [final("done", "search_flights")]}))
    r = next(r for r in res.records if r["kind"] == "tool_result")
    assert r["t_ms"] == 1334 and r["data"]["ok"] is True and r["data"]["status"] == "ok"
    assert "flights" in r["data"]["result"]


def test_cancel_stops_call_and_acks_and_no_commit():
    tools = {"config": {"book_flight": {"latency_ms": 3000}}}
    ev2 = {"id": "e2", "t_ms": 1000, "type": "interrupt", "reason": "barge_in"}
    res = run_scenario(make([TXT, ev2], tools), factory({
        "e1": [call("b1", "book_flight", flight_id="AI-1", passenger_name="A")],
        "e2": [cancel("b1"), final("cancelled", None)]}))
    assert res.state["bookings"] == []
    results = [r["data"] for r in res.records if r["kind"] == "tool_result"]
    assert [x["status"] for x in results] == ["cancelled"]
    assert ("sys", "tool_cancelled") in kinds(res)
    assert ("sys", "write_committed") not in kinds(res)


def test_cancel_ack_can_be_disabled_and_late_cancel_ignored():
    ev2 = {"id": "e2", "t_ms": 5000, "type": "text_chunk", "text": "x", "end_of_turn": True}
    cfg = HarnessConfig(emit_cancel_ack=False)
    res = run_scenario(make([TXT, ev2], {"config": {"flight_search": {"latency_ms": 100}}}), factory({
        "e1": [call("c1", "flight_search", **FS)], "e2": [cancel("c1"), cancel("nope"), final("x", None)]}), config=cfg)
    ign = [r["data"]["reason"] for r in res.records if r["kind"] == "cancel_ignored"]
    assert ign == ["already_completed", "unknown"]


def test_write_commits_on_completion():
    res = run_scenario(make([TXT]), factory({
        "e1": [call("b1", "book_flight", flight_id="AI-1", passenger_name="A")],
        "result:b1:ok": [final("booked", "book_flight")]}))
    assert len(res.state["bookings"]) == 1
    assert ("sys", "write_committed") in kinds(res)


def test_timeout_fault_delivers_timeout_status():
    tools = {"config": {"flight_search": {"faults": [{"call_index": 1, "kind": "timeout", "timeout_ms": 2500}]}}}
    res = run_scenario(make([TXT], tools), factory({
        "e1": [call("c1", "flight_search", **FS)], "result:c1:timeout": [final("sorry", None)]}))
    r = next(r for r in res.records if r["kind"] == "tool_result")
    assert r["t_ms"] == 2600 and r["data"]["status"] == "timeout" and r["data"]["retryable"] is True


def test_invalid_actions_are_logged_not_executed():
    res = run_scenario(make([TXT]), factory({"e1": [
        {"type": "dance"}, {"type": "tool_call", "tool": "flight_search"}, say(""), final("ok", None)]}))
    inv = [r for r in res.records if r["kind"] == "invalid"]
    assert len(inv) == 3
    assert not any(r["kind"] == "tool_started" for r in res.records)


def test_protocol_py_action_names_are_accepted():
    act = {"type": "final_response", "action_id": "a1", "t": 100, "text": "done",
           "state_snapshot": {"intent": None, "slots": {}}}
    res = run_scenario(make([TXT]), factory({"e1": [act]}))
    out = [r for r in res.records if r["dir"] == "out"]
    assert out[0]["kind"] == "final" and out[0]["data"]["snapshot"] == {"intent": None, "slots": {}}
    assert res.end_reason == "quiescent"


def test_duplicate_call_id_rejected():
    res = run_scenario(make([TXT]), factory({"e1": [call("c1", "flight_search", **FS), call("c1", "flight_search", **FS)],
                                              "result:c1:ok": [final("x", None)]}))
    assert any(r["kind"] == "duplicate_call_id" for r in res.records)


def test_deadline_when_agent_silent():
    res = run_scenario(make([TXT], duration_ms=3000), factory({}))
    assert res.end_reason == "deadline"
    assert res.records[-1]["kind"] == "scenario_end"
    assert any(r["kind"] == "session_end" for r in res.records)


def test_agent_crash_is_recorded():
    async def boom(inbox, outbox):
        await inbox.get()
        raise RuntimeError("kaput")

    res = run_scenario(make([TXT]), lambda: __import__("sim.adapter", fromlist=["x"]).resolve(boom))
    assert res.end_reason == "agent_error" and "kaput" in res.agent_error


def test_clock_is_passed_to_agent_factory():
    seen = {}

    class ClockAgent:
        def __init__(self, clock=None):
            self.clock = clock

        async def run(self, inbox, outbox):
            while True:
                ev = await inbox.get()
                if ev.get("event_id") == "e1":
                    await asyncio.sleep(0.25)
                    seen["t"] = self.clock()
                    outbox.put_nowait(final("x", None))
                if ev["type"] == "session_end":
                    return

    run_scenario(make([TXT]), lambda clock=None: ClockAgent(clock))
    assert seen["t"] == pytest.approx(350)


def test_strip_oracle():
    ev = {"id": "e1", "t_ms": 0, "type": "audio_clip", "clip_id": "c", "duration_ms": 500, "end_of_turn": True,
          "transcript": "secret"}
    res = run_scenario(make([ev]), factory({"e1": [final("x", None)]}), config=HarnessConfig(strip_oracle=True))
    audio = next(r for r in res.records if r["kind"] == "audio_clip")
    assert "transcript" not in audio["data"]


def test_trace_jsonl_roundtrip_and_determinism(tmp_path):
    tools = {"config": {"flight_search": {"latency_ms": 800}}}
    script = {"e1": [say("looking", "progress"), call("c1", "flight_search", **FS)],
              "result:c1:ok": [final("found", "search_flights", **FS)]}
    a = run_scenario(make([TXT], tools), factory(script), trace_path=tmp_path / "a.jsonl")
    b = run_scenario(make([TXT], tools), factory(script), trace_path=tmp_path / "b.jsonl")
    assert (tmp_path / "a.jsonl").read_text() == (tmp_path / "b.jsonl").read_text()
    recs = read_jsonl(tmp_path / "a.jsonl")
    assert recs == json.loads(json.dumps(a.records))
    assert [r["seq"] for r in recs] == list(range(len(recs)))
    assert {"seq", "t_ms", "dir", "kind", "data"} <= set(recs[0])


def test_asset_paths_resolved_before_replay(tmp_path, monkeypatch):
    ev = {"id": "e1", "t_ms": 0, "type": "audio_clip", "clip_id": "c", "path": "a.wav", "duration_ms": 500,
          "end_of_turn": True, "transcript": "hi"}
    sc = make([ev])
    sc.base_dir = tmp_path
    from sim import harness as H
    h = H.Harness(sc, factory({"e1": [final("x", None)]}))
    import pathlib
    def boom(self, *a, **k):
        raise AssertionError("resolve() called during replay")
    monkeypatch.setattr(pathlib.Path, "resolve", boom)
    res = h.run()
    audio = next(r for r in res.records if r["kind"] == "audio_clip")
    assert audio["data"]["path"] == str(tmp_path / "a.wav")
