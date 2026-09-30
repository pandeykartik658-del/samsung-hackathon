# /mnt/project-files/theme5/tests/test_fastpath.py
"""Fast path tests. Scenarios run on a virtual clock (ms) and are logged in the
sim/trace.py record format; assertions inspect the trace only."""
from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from theme5.fastpath import (
    Classification, FastPath, FastPathConfig, FastPathTicker, InputKind, audit_false_completion, audit_fillers,
    audit_latency, audit_unbacked_narration, claims_completion, classify, describe_call,
)
from theme5.protocol import (
    ACT_CANCEL, ACT_CLARIFY, ACT_FINAL, ACT_SPEAK, ACT_TOOL_CALL, SPEAK_ACK, SPEAK_PROGRESS, ToolResult,
    make_action, validate_action,
)

MANIFEST = {"tools": [
    {"name": "search_flights", "read_only": True,
     "parameters": {"type": "object", "properties": {"origin": {}, "destination": {}, "date": {}}, "required": ["destination"]}},
    {"name": "book_flight", "state_modifying": True,
     "parameters": {"type": "object", "properties": {"flight_id": {}}, "required": ["flight_id"]}},
]}


# --------------------------------------------------------------------------- scripted driver
@dataclass
class Step:
    t: float
    op: str  # text | interrupt | call | cancel | result | clarify | final | say
    args: dict[str, Any] = field(default_factory=dict)


def text(t, s, eot=True, **kw):
    return Step(t, "text", {"text": s, "eot": eot, **kw})


def call(t, cid, tool, **args):
    return Step(t, "call", {"call_id": cid, "tool": tool, "args": args})


def result(t, cid, ok=True, res=None):
    return Step(t, "result", {"call_id": cid, "ok": ok, "result": res})


class Run:
    def __init__(self, steps: list[Step], cfg: FastPathConfig | None = None, until: float | None = None):
        self.fp = FastPath(cfg)
        self.records: list[dict[str, Any]] = []
        self._seq = itertools.count()
        self._log(0.0, "in", "tool_manifest", dict(MANIFEST, type="tool_manifest", t=0.0))
        steps = sorted(steps, key=lambda s: s.t)
        # steps sharing a timestamp are one event handler: flush once at its end
        for t, group in itertools.groupby(steps, key=lambda s: s.t):
            self._ticks_until(t)
            for s in group:
                self._do(s)
            self._out(self.fp.flush(t))
        self._ticks_until(until if until is not None else (steps[-1].t if steps else 0.0) + 30_000)

    def _log(self, t, d, kind, data):
        self.records.append({"seq": next(self._seq), "t_ms": round(t, 3), "dir": d, "kind": kind, "data": data})

    def _out(self, actions):
        for a in actions:
            assert validate_action(a) == [], a
            self._log(a["t"], "out", a["type"], a)

    def _emit(self, a):
        self._log(a["t"], "out", a["type"], a)
        self._out(self.fp.observe(a))

    def _ticks_until(self, t_end: float):
        while True:
            d = self.fp.next_deadline()
            if d is None or d > t_end:
                return
            acts = self.fp.tick(d)
            self._out(acts)
            if not acts and self.fp.next_deadline() == d:
                return

    def _do(self, s: Step):
        a, t = s.args, s.t
        if s.op == "text":
            self._log(t, "in", "text", {"type": "text", "t": t, "text": a["text"], "end_of_turn": a["eot"]})
            self.fp.on_user_text(a["text"], t, a["eot"], awaiting_clarification=a.get("awaiting", False),
                                 has_active_task=a.get("active", False))
        elif s.op == "interrupt":
            self._log(t, "in", "interrupt", {"type": "interrupt", "t": t})
            self.fp.on_interrupt(t)
        elif s.op == "call":
            self._emit(make_action(ACT_TOOL_CALL, t, call_id=a["call_id"], tool=a["tool"], args=a["args"]))
        elif s.op == "cancel":
            self._emit(make_action(ACT_CANCEL, t, call_id=a["call_id"]))
        elif s.op == "result":
            self._log(t, "in", "tool_result", {"type": "tool_result", "t": t, **a})
            self._out(self.fp.on_result(ToolResult(a["call_id"], a["ok"], a.get("result")), t))
        elif s.op == "clarify":
            self._emit(make_action(ACT_CLARIFY, t, text=a["text"], slot=None))
        elif s.op == "final":
            self._emit(make_action(ACT_FINAL, t, snapshot={"intent": None, "slots": {}}, text=a["text"]))
        elif s.op == "say":
            self._emit(make_action(ACT_SPEAK, t, text=a["text"], kind=a.get("kind", SPEAK_ACK)))

    # views
    def spoken(self, source_fp_only: bool = False) -> list[tuple[float, str, str]]:
        return [(r["t_ms"], r["data"].get("kind"), r["data"]["text"]) for r in self.records
                if r["dir"] == "out" and r["kind"] == ACT_SPEAK]

    def texts(self) -> list[str]:
        return [x[2] for x in self.spoken()]


def assert_clean(run: Run, budget: float = 300.0):
    assert audit_false_completion(run.records) == []
    assert audit_unbacked_narration(run.records) == []
    assert audit_fillers(run.records, run.fp.cfg) == []
    assert audit_latency(run.records, budget) == []


# --------------------------------------------------------------------------- classification
@pytest.mark.parametrize("s,kw,kind", [
    ("uh-huh", {}, InputKind.BACKCHANNEL),
    ("mm hmm", {"has_active_task": True}, InputKind.BACKCHANNEL),
    ("umm", {"awaiting_clarification": True}, InputKind.BACKCHANNEL),
    ("okay", {"has_active_task": True}, InputKind.BACKCHANNEL),
    ("yeah I see", {}, InputKind.BACKCHANNEL),
    ("Book a flight to Mumbai on Friday", {}, InputKind.NEW_REQUEST),
    ("okay so book a flight to Goa", {}, InputKind.NEW_REQUEST),
    ("Actually make it Pune", {"has_active_task": True}, InputKind.CORRECTION),
    ("no, to Chennai instead", {"has_active_task": True}, InputKind.CORRECTION),
    ("Mumbai", {"has_active_task": True}, InputKind.CORRECTION),
    ("Never mind", {"has_active_task": True}, InputKind.CANCELLATION),
    ("cancel it", {"has_active_task": True}, InputKind.CANCELLATION),
    ("forget it", {}, InputKind.CANCELLATION),
    ("cancel my booking", {}, InputKind.NEW_REQUEST),
    ("Friday", {"awaiting_clarification": True}, InputKind.CLARIFICATION_ANSWER),
    ("yes", {"awaiting_clarification": True}, InputKind.CLARIFICATION_ANSWER),
    ("okay", {"awaiting_clarification": True}, InputKind.CLARIFICATION_ANSWER),
    ("never mind", {"awaiting_clarification": True}, InputKind.CANCELLATION),
    ("", {}, InputKind.EMPTY),
])
def test_classify(s, kw, kind):
    assert classify(s, **kw).kind is kind


def test_self_repair_in_fresh_request_is_new_request():
    c = classify("fly to Delhi, no, Mumbai tomorrow")
    assert c.kind is InputKind.NEW_REQUEST and c.self_repair


def test_backchannel_does_not_take_floor():
    assert not classify("uh-huh").takes_floor
    assert classify("book a flight").takes_floor


def test_cumulative_and_incremental_chunks_agree():
    a, b = FastPath(), FastPath()
    for chunk in ("book a", "flight to", "Mumbai"):
        a.on_user_text(chunk, 0, False)
    for chunk in ("book a", "book a flight to", "book a flight to Mumbai"):
        b.on_user_text(chunk, 0, False)
    ca, cb = a.on_user_text("", 10, True), b.on_user_text("", 10, True)
    assert ca == cb and ca.kind is InputKind.NEW_REQUEST and "Mumbai" in ca.text


def test_classifier_is_fast():
    samples = ["uh-huh", "book a flight to Mumbai on Friday for two", "actually make it Pune", "never mind"] * 250
    t0 = time.perf_counter()
    for s in samples:
        classify(s, has_active_task=True)
    assert (time.perf_counter() - t0) / len(samples) < 0.002  # < 2 ms real CPU each, far under budget


# --------------------------------------------------------------------------- narration text
@pytest.mark.parametrize("tool,args,want", [
    ("search_flights", {"destination": "Mumbai"}, "checking flights to Mumbai"),
    ("search_flights", {"origin": "delhi", "destination": "mumbai", "date": "2026-10-02"},
     "checking flights from Delhi to Mumbai on 2026-10-02"),
    ("book_flight", {"flight_id": "AI101"}, "booking a flight"),
    ("lookup_manual", {"device": "tv"}, "looking up the manual for tv"),
    ("createTicket", {"issue": "the washer is leaking water all over the floor"}, "creating a ticket"),
    ("navigate", {"destination": "Airport"}, "working out the route to Airport"),
    ("find_hotel", {"city": "goa"}, "looking for a hotel in Goa"),
    ("frobnicate", {}, "working on the frobnicate"),
])
def test_describe_call(tool, args, want):
    assert describe_call(tool, args) == want


def test_describe_call_never_claims_completion():
    assert describe_call("search_flights", {"destination": "done"}) == "working on that"


@pytest.mark.parametrize("s,claim", [
    ("Your flight is booked.", True), ("Done, booking succeeded.", True), ("Here's what I found.", True),
    ("I've created the ticket.", True), ("Checking flights to Mumbai.", False), ("Booking a flight.", False),
    ("That didn't go through, so I'm booking a flight again.", False), ("It wasn't booked.", False),
    ("Okay, I've stopped checking flights to Delhi.", False),
])
def test_claims_completion(s, claim):
    assert claims_completion(s) is claim


# --------------------------------------------------------------------------- scenarios
def test_request_narrates_the_issued_call_immediately():
    r = Run([text(1000, "find flights to Mumbai"), call(1000, "c1", "search_flights", destination="Mumbai"),
             result(1800, "c1", res=[{"flight_id": "AI1"}]), Step(1800, "final", {"text": "Here's what I found: AI1."})])
    assert r.spoken()[0] == (1000.0, SPEAK_ACK, "Checking flights to Mumbai.")
    assert_clean(r)


def test_no_narration_without_a_call():
    # engine could not call (missing slot) and clarifies instead
    r = Run([text(1000, "find me a flight"), Step(1020, "clarify", {"text": "Where to?"})])
    assert not any("hecking" in t for t in r.texts())
    assert r.texts() == []  # clarify at 20 ms beat the 150 ms generic-ack deadline
    assert_clean(r)


def test_generic_ack_only_when_nothing_concrete_by_deadline():
    r = Run([text(1000, "find me a flight"), Step(1600, "clarify", {"text": "Where to?"})])
    assert r.spoken() == [(1150.0, SPEAK_ACK, "Sure, one moment.")]
    assert not claims_completion(r.texts()[0])
    assert_clean(r)


def test_long_call_progress_is_rate_limited_and_never_repeats():
    cfg = FastPathConfig()
    r = Run([text(0, "find flights to Mumbai"), call(0, "c1", "search_flights", destination="Mumbai"),
             result(25_000, "c1", res=[])], cfg=cfg)
    prog = [x for x in r.spoken() if x[1] == SPEAK_PROGRESS]
    assert 1 <= len(prog) <= len(cfg.progress_after_ms)
    assert prog[0] == (2000.0, SPEAK_PROGRESS, "Still checking flights to Mumbai.")
    assert len(set(r.texts())) == len(r.texts())
    assert_clean(r)


def test_parallel_calls_share_one_filler_budget():
    cfg = FastPathConfig()
    steps = [text(0, "find flights to Mumbai, Pune and Goa")]
    steps += [call(0, f"c{i}", "search_flights", destination=d) for i, d in enumerate(("Mumbai", "Pune", "Goa"))]
    steps += [result(20_000, f"c{i}") for i in range(3)]
    r = Run(steps, cfg=cfg)
    fillers = [x for x in r.spoken() if x[1] == SPEAK_PROGRESS]
    assert len(fillers) <= cfg.max_fillers_per_turn
    gaps = [b[0] - a[0] for a, b in zip(fillers, fillers[1:])]
    assert all(g >= cfg.filler_min_gap_ms for g in gaps)
    assert_clean(r)


def test_filler_audit_catches_spam():
    r = Run([text(0, "find flights to Mumbai"), call(0, "c1", "search_flights", destination="Mumbai"),
             Step(500, "say", {"text": "One moment.", "kind": "filler"}),
             Step(900, "say", {"text": "One moment.", "kind": "filler"}), result(1000, "c1")])
    why = {v["why"].split(" ")[0] for v in audit_fillers(r.records)}
    assert {"filler", "repeated"} <= why


def test_backchannel_is_silent_and_changes_nothing():
    base = [text(0, "find flights to Mumbai"), call(0, "c1", "search_flights", destination="Mumbai"),
            result(9000, "c1")]
    quiet = Run(base)
    chatty = Run(base + [text(1200, "uh-huh"), text(2500, "mm hmm"), text(4000, "okay")])
    assert quiet.spoken() == chatty.spoken()
    assert_clean(chatty)


def test_no_speech_while_user_holds_the_floor():
    # speculative call on a partial transcript: narrate only at end of turn
    r = Run([text(0, "find flights to Mumbai", eot=False), call(10, "c1", "search_flights", destination="Mumbai"),
             text(3000, "on Friday", eot=False), text(3400, "", eot=True), result(8000, "c1")])
    first = r.spoken()[0]
    assert first == (3400.0, SPEAK_ACK, "Checking flights to Mumbai.")
    assert_clean(r)


def test_correction_cancels_then_narrates_the_new_call():
    r = Run([text(0, "find flights to Delhi"), call(0, "c1", "search_flights", destination="Delhi"),
             Step(900, "interrupt"), text(1000, "actually make it Mumbai", active=True),
             Step(1000, "cancel", {"call_id": "c1"}), call(1000, "c2", "search_flights", destination="Mumbai"),
             result(1100, "c1"),  # late stale result of the cancelled call
             result(2500, "c2", res=[{"flight_id": "AI2"}])])
    assert (1000.0, SPEAK_ACK, "Okay, checking flights to Mumbai instead.") in r.spoken()
    assert not any("Delhi" in t for t in r.texts()[1:])
    assert_clean(r)


def test_cancellation_ack_names_what_was_stopped():
    r = Run([text(0, "find flights to Delhi"), call(0, "c1", "search_flights", destination="Delhi"),
             Step(1500, "interrupt"), text(1600, "never mind", active=True), Step(1600, "cancel", {"call_id": "c1"})])
    assert r.spoken()[-1] == (1600.0, SPEAK_ACK, "Okay, I've stopped checking flights to Delhi.")
    assert not any(x[1] == SPEAK_PROGRESS for x in r.spoken())  # no "still checking" after cancel
    assert_clean(r)


def test_retry_narration_only_after_a_real_failure():
    r = Run([text(0, "book flight AI101"), call(0, "b1", "book_flight", flight_id="AI101"),
             result(800, "b1", ok=False), call(800, "b2", "book_flight", flight_id="AI101"),
             result(1500, "b2", res={"pnr": "X1"}), Step(1500, "final", {"text": "Done, your flight is booked. PNR X1."})])
    assert "That didn't go through, so I'm booking a flight again." in r.texts()
    assert_clean(r)


def test_completion_audit_flags_claim_before_result():
    r = Run([text(0, "book flight AI101"), call(0, "b1", "book_flight", flight_id="AI101"),
             Step(300, "say", {"text": "Your flight is booked.", "kind": "info"}), result(900, "b1")])
    bad = audit_false_completion(r.records)
    assert len(bad) == 1 and bad[0]["t_ms"] == 300.0


def test_completion_audit_ignores_results_of_cancelled_calls():
    r = Run([text(0, "book flight AI101"), call(0, "b1", "book_flight", flight_id="AI101"),
             Step(100, "cancel", {"call_id": "b1"}), result(200, "b1", res={"pnr": "X"}),
             Step(300, "final", {"text": "Your flight is booked."})])
    assert len(audit_false_completion(r.records)) == 1


def test_completion_audit_needs_a_write_for_write_claims():
    r = Run([text(0, "find flights to Mumbai"), call(0, "c1", "search_flights", destination="Mumbai"),
             result(500, "c1", res=[{"flight_id": "AI1"}]), Step(500, "final", {"text": "Booked AI1 for you."})])
    assert len(audit_false_completion(r.records)) == 1


def test_unbacked_narration_audit_catches_fake_progress():
    r = Run([text(0, "find flights to Mumbai"), Step(50, "say", {"text": "Checking flights to Mumbai.", "kind": "progress"})])
    assert len(audit_unbacked_narration(r.records)) == 1


def test_fast_path_never_claims_completion_across_scenarios():
    scenarios = [
        [text(0, "find flights to Mumbai"), call(0, "c1", "search_flights", destination="Mumbai"), result(15_000, "c1")],
        [text(0, "book it"), call(0, "b1", "book_flight", flight_id="Z9"), result(400, "b1", ok=False),
         call(400, "b2", "book_flight", flight_id="Z9"), result(12_000, "b2")],
        [text(0, "never mind")], [text(0, "Friday", awaiting=True)], [text(0, "hello there")],
    ]
    for steps in scenarios:
        r = Run(steps)
        assert not any(claims_completion(t) for t in r.texts()), r.texts()
        assert_clean(r)


def test_latency_budget_every_turn_on_virtual_clock():
    cfg = FastPathConfig()
    r = Run([
        text(0, "find flights to Delhi"), call(0, "c1", "search_flights", destination="Delhi"),
        text(2000, "uh-huh"),
        Step(3000, "interrupt"), text(3200, "no, Mumbai", active=True), Step(3200, "cancel", {"call_id": "c1"}),
        call(3230, "c2", "search_flights", destination="Mumbai"),  # slow path took 30 ms
        result(5000, "c2"),
        text(9000, "what about hotels"),  # engine does nothing: generic ack at deadline
        text(12000, "forget it", active=True),
    ], cfg=cfg)
    assert audit_latency(r.records, cfg.latency_budget_ms) == []
    worst = audit_latency(r.records, 0.0)
    assert max(v["latency_ms"] for v in worst) <= cfg.ack_deadline_ms
    assert_clean(r)


def test_latency_audit_flags_slow_and_missing_responses():
    recs = [
        {"seq": 0, "t_ms": 0.0, "dir": "in", "kind": "text", "data": {"text": "find flights", "end_of_turn": True}},
        {"seq": 1, "t_ms": 900.0, "dir": "out", "kind": "speak", "data": {"text": "Okay.", "kind": "ack"}},
        {"seq": 2, "t_ms": 1000.0, "dir": "in", "kind": "text", "data": {"text": "to Pune", "end_of_turn": True}},
    ]
    bad = audit_latency(recs, 300.0)
    assert bad[0]["latency_ms"] == 900.0 and bad[1]["why"] == "no response"


def test_media_ack_is_immediate():
    fp = FastPath()
    acts = fp.on_media("frame", 42.0)
    assert len(acts) == 1 and acts[0]["t"] == 42.0 and acts[0]["kind"] == SPEAK_ACK
    assert validate_action(acts[0]) == []


def test_observe_ignores_own_actions_and_counts_engine_fillers():
    fp = FastPath()
    fp.on_user_text("find flights to Mumbai", 0, True)
    own = fp.observe(make_action(ACT_TOOL_CALL, 0, call_id="c1", tool="search_flights", args={"destination": "Mumbai"}))
    assert fp.observe(own[0]) == []
    fp.observe(make_action(ACT_SPEAK, 100, text="One moment.", kind="filler"))
    assert fp.limiter.turn_count == 1 and fp.limiter.last_t == 100


def test_trace_is_deterministic():
    steps = [text(0, "find flights to Delhi"), call(0, "c1", "search_flights", destination="Delhi"),
             text(2500, "actually Mumbai", active=True), Step(2500, "cancel", {"call_id": "c1"}),
             call(2500, "c2", "search_flights", destination="Mumbai"), result(9000, "c2")]
    strip = lambda recs: [(r["t_ms"], r["dir"], r["kind"], r["data"].get("text")) for r in recs]
    assert strip(Run(steps).records) == strip(Run(steps).records)


# --------------------------------------------------------------------------- ticker on the virtual loop
def test_ticker_narrates_on_virtual_time():
    vloop = pytest.importorskip("sim.vloop")

    async def main():
        loop = asyncio.get_running_loop()
        now = lambda: loop.time() * 1000.0
        said: list[tuple[float, str]] = []
        fp = FastPath()
        ticker = FastPathTicker(fp, lambda a: said.append((a["t"], a["text"])), now)
        ticker.start()
        fp.on_user_text("find flights to Mumbai", now(), True)
        for a in fp.observe(make_action(ACT_TOOL_CALL, now(), call_id="c1", tool="search_flights",
                                        args={"destination": "Mumbai"})):
            said.append((a["t"], a["text"]))
        ticker.poke()
        await asyncio.sleep(10.0)
        fp.on_result(ToolResult("c1", True, []), now())
        ticker.poke()
        await asyncio.sleep(10.0)
        await ticker.stop()
        return said

    said = vloop.run_virtual(main())
    assert said[0] == (0.0, "Checking flights to Mumbai.")
    assert said[1] == (2000.0, "Still checking flights to Mumbai.")
    assert said[2][0] == 6000.0
    assert len(said) == 3  # nothing after the result
