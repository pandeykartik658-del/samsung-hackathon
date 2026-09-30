# tests/bench/test_scorer.py
"""Scorer tests on hand-built traces in the sim/trace.py record format."""
from __future__ import annotations

import json

import pytest

from bench import scorer

MANIFEST = [
    {"name": "flight_search", "side_effect": "read",
     "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"},
                                                      "date": {"type": "string"}}, "required": ["origin", "destination", "date"]}},
    {"name": "book_flight", "side_effect": "write",
     "parameters": {"type": "object", "properties": {"flight_id": {"type": "string"}, "passenger_name": {"type": "string"}},
                    "required": ["flight_id", "passenger_name"]}},
]

SCENARIO = {
    "id": "t_search_book", "modality": "text", "cancel_grace_ms": 300,
    "events": [{"id": "u1", "t_ms": 100, "type": "text_chunk"}, {"id": "u2", "t_ms": 1500, "type": "text_chunk"}],
    "expected": {
        "required_calls": [{"tool": "flight_search", "args": {"destination": "BOM"}},
                           {"tool": "book_flight", "args": {"flight_id": "AI-101"}}],
        "forbidden_calls": [{"tool": "book_flight", "before_event": "u2"}],
        "final_snapshot": {"intent": "book_flight", "slots": {"destination": {"any_of": ["BOM", "Mumbai"]}, "flight_id": "AI-101"}},
        "max_writes": {"book_flight": 1},
    },
}


class T:
    """Tiny trace builder."""

    def __init__(self):
        self.recs = []

    def add(self, t, d, rec_kind, **data):
        self.recs.append({"seq": len(self.recs), "t_ms": float(t), "dir": d, "kind": rec_kind, "data": data})
        return self

    def user(self, t, text, eid, eot=True):
        return self.add(t, "in", "text_chunk", type="text_chunk", event_id=eid, text=text, end_of_turn=eot)

    def say(self, t, text, kind="ack"):
        return self.add(t, "out", "speak", type="speak", text=text, kind=kind)

    def call(self, t, cid, tool, **args):
        return self.add(t, "out", "tool_call", type="tool_call", call_id=cid, tool=tool, args=args)

    def result(self, t, cid, status="ok", **res):
        return self.add(t, "in", "tool_result", type="tool_result", call_id=cid, status=status, result=res)

    def cancel(self, t, cid):
        return self.add(t, "out", "cancel", type="cancel", call_id=cid)

    def final(self, t, text, intent, **slots):
        return self.add(t, "out", "final", type="final", text=text, snapshot={"intent": intent, "slots": slots})


def good_trace():
    t = T().add(0, "in", "tool_manifest", type="tool_manifest", tools=MANIFEST)
    t.user(100, "find me a flight to Mumbai tomorrow", "u1")
    t.say(150, "Searching flights from Delhi to Mumbai for tomorrow.")
    t.call(160, "c1", "flight_search", origin="DEL", destination="BOM", date="2026-09-30")
    t.result(1360, "c1", flights=[{"flight_id": "AI-101"}, {"flight_id": "6E-202"}])
    t.say(1370, "I found AI-101 at 9am and 6E-202 at noon.", kind="info")
    t.user(1500, "book the first one", "u2")
    t.say(1560, "Booking AI-101 now.")
    t.call(1570, "c2", "book_flight", flight_id="AI-101", passenger_name="Asha")
    t.result(2470, "c2", booking_ref="QX7P2K", status_="confirmed")
    t.add(2470, "sys", "write_committed", tool="book_flight", call_id="c2")
    t.final(2500, "Booked AI-101, reference QX7P2K.", "book_flight", destination="BOM", flight_id="AI-101")
    return t.recs


def test_perfect_trace_scores_high():
    s = scorer.score_trace(good_trace(), SCENARIO)
    assert s.TC == pytest.approx(40.0)
    assert s.SP == pytest.approx(10.0)
    assert s.LAT == pytest.approx(15.0)
    assert s.IR == pytest.approx(35.0)
    assert s.QM > 1.0
    assert s.total == 100.0
    assert s.weight == 1.0


def test_multimodal_weight():
    sc = dict(SCENARIO, modality="audio")
    assert scorer.score_trace(good_trace(), sc).weight == 1.5
    agg = scorer.suite_score([scorer.score_trace(good_trace(), SCENARIO), scorer.score_trace([], sc)])
    # (100*1 + x*1.5)/2.5
    assert agg["weighted_mean"] < agg["mean"]


def test_duplicate_booking_zeroes_nodup_and_counts_max_writes():
    recs = good_trace()
    t = T()
    t.recs = recs
    t.call(2600, "c3", "book_flight", flight_id="AI-101", passenger_name="asha ")
    t.result(3500, "c3", booking_ref="ZZ9")
    t.add(3500, "sys", "write_committed", tool="book_flight", call_id="c3")
    s = scorer.score_trace(t.recs, SCENARIO)
    assert s.components["sp_nodup"] == 0.0
    assert any("duplicate write" in n for n in s.notes)
    assert any("committed/issued 2 times" in n for n in s.notes)
    assert s.SP <= 6.0


def test_retry_after_error_is_not_duplicate_but_after_timeout_is():
    base = T().add(0, "in", "tool_manifest", type="tool_manifest", tools=MANIFEST)
    base.call(10, "c1", "book_flight", flight_id="X1", passenger_name="A")
    base.result(500, "c1", status="error")
    base.call(600, "c2", "book_flight", flight_id="X1", passenger_name="A")
    v = scorer.build_view(base.recs)
    assert scorer.duplicate_writes(v, {}) == []
    tb = T().add(0, "in", "tool_manifest", type="tool_manifest", tools=MANIFEST)
    tb.call(10, "c1", "book_flight", flight_id="X1", passenger_name="A")
    tb.result(5000, "c1", status="timeout")
    tb.call(5100, "c2", "book_flight", flight_id="X1", passenger_name="A")
    assert len(scorer.duplicate_writes(scorer.build_view(tb.recs), {})) == 1


def test_unknown_tool_defaults_to_state_modifying():
    assert scorer.is_write_tool(None)
    assert scorer.is_write_tool({"name": "x"})
    assert not scorer.is_write_tool({"name": "x", "annotations": {"readOnlyHint": True}})
    assert not scorer.is_write_tool({"name": "x", "read_only": True})
    assert scorer.is_write_tool({"name": "x", "side_effect": "write"})


def test_false_completion_claim_penalised():
    t = T().add(0, "in", "tool_manifest", type="tool_manifest", tools=MANIFEST)
    t.user(100, "book AI-101 for Asha", "u1")
    t.say(120, "Your flight is booked!")
    t.call(130, "c1", "book_flight", flight_id="AI-101", passenger_name="Asha")
    t.result(900, "c1", status="error")
    t.final(950, "Your flight AI-101 is booked.", "book_flight", flight_id="AI-101")
    sc = {"id": "x", "modality": "text", "expected": {"required_calls": [], "forbidden_calls": [], "final_snapshot": None}}
    s = scorer.score_trace(t.recs, sc)
    assert s.components["tc_grounding"] == 0.0
    assert s.QM <= 0.85
    assert sum("false completion" in n for n in s.notes) == 2


def test_negated_claim_is_not_false_claim():
    t = T().add(0, "in", "tool_manifest", type="tool_manifest", tools=MANIFEST)
    t.final(10, "The booking could not be confirmed; nothing was booked.", None)
    assert scorer._false_claims(scorer.build_view(t.recs)) == []


def test_interrupt_cancel_promptness_and_stale_rerun():
    sc = {
        "id": "dest_change", "modality": "text", "cancel_grace_ms": 300,
        "events": [{"id": "u1", "t_ms": 100}, {"id": "u2", "t_ms": 800}],
        "expected": {"required_calls": [{"tool": "flight_search", "args": {"destination": "GOI"}}],
                     "forbidden_calls": [], "final_snapshot": {"intent": "search_flights", "slots": {"destination": "GOI"}},
                     "must_cancel": [{"anchor_event": "u2", "tool": "flight_search"}]},
    }
    def trace(cancel_at):
        t = T().add(0, "in", "tool_manifest", type="tool_manifest", tools=MANIFEST)
        t.user(100, "flights to Mumbai", "u1")
        t.say(130, "Searching Mumbai flights.")
        t.call(140, "c1", "flight_search", origin="DEL", destination="BOM", date="2026-09-30")
        t.user(800, "actually Goa", "u2")
        if cancel_at is not None:
            t.cancel(cancel_at, "c1")
        t.say(max(cancel_at or 0, 800) + 5, "Switching to Goa.")
        t.call(820 if cancel_at is None else cancel_at + 10, "c2", "flight_search", origin="DEL", destination="GOI", date="2026-09-30")
        t.result(1900, "c1", flights=[{"flight_id": "AI-555"}])
        t.result(2100, "c2", flights=[{"flight_id": "6E-777"}])
        t.final(2150, "6E-777 flies to Goa at 10am.", "search_flights", destination="GOI")
        return t.recs
    fast = scorer.score_trace(trace(801), sc)
    slow = scorer.score_trace(trace(1500), sc)
    never = scorer.score_trace(trace(None), sc)
    assert fast.components["ir_cancel"] == 1.0
    assert 0 < slow.components["ir_cancel"] < 1.0
    assert never.components["ir_cancel"] == 0.0
    assert fast.IR > slow.IR > never.IR


def test_stale_result_in_final_is_caught():
    sc = {"id": "s", "modality": "text", "expected": {"required_calls": [], "forbidden_calls": [],
                                                      "final_snapshot": {"intent": "search_flights", "slots": {"destination": "GOI"}}}}
    t = T().add(0, "in", "tool_manifest", type="tool_manifest", tools=MANIFEST)
    t.user(100, "flights to Mumbai", "u1")
    t.call(140, "c1", "flight_search", origin="DEL", destination="BOM", date="d")
    t.user(300, "no, Goa", "u2")
    t.cancel(301, "c1")
    t.call(310, "c2", "flight_search", origin="DEL", destination="GOI", date="d")
    t.result(900, "c1", flights=[{"flight_id": "AI-555"}])
    t.result(1000, "c2", flights=[{"flight_id": "6E-777"}])
    t.final(1100, "Take AI-555.", "search_flights", destination="GOI")
    s = scorer.score_trace(t.recs, sc)
    assert any("stale value" in n for n in s.notes)
    assert s.components["ir_stale"] <= 0.5


def test_latency_curve_and_merge():
    assert scorer.lat_curve(200) == 1.0
    assert scorer.lat_curve(1000) == pytest.approx(0.5)
    assert scorer.lat_curve(3000) == 0.0
    assert scorer.lat_curve(None) == 0.0
    t = T()
    t.add(0, "in", "video_frame", type="video_frame", event_id="f1")
    t.user(400, "what is this error?", "u1")  # frame merged into this boundary
    t.say(650, "Let me look at that.", kind="filler")  # filler does not stop the clock
    t.say(900, "Looking up the E21 code for this washer.")
    s = scorer.ScenarioScore("x", "visual", 0, 0, 0, 0, 1, 0, 1.5)
    scorer.score_lat(scorer.build_view(t.recs), s)
    assert s.latencies_ms == [500.0]


def test_invalid_actions_and_missing_final_cost_sp():
    t = T().add(0, "in", "tool_manifest", type="tool_manifest", tools=MANIFEST)
    t.user(100, "hi", "u1")
    t.add(120, "out", "invalid", raw="garbage")
    t.add(130, "out", "speak", type="speak", text="")
    t.add(140, "sys", "agent_error", error="boom")
    s = scorer.score_trace(t.recs, {"id": "x", "modality": "text", "expected": {}})
    assert s.components["sp_valid"] == 0.0
    assert s.components["tc_snapshot"] == 0.0 and s.components["tc_grounding"] == 0.0


def test_protocol_py_shape_is_accepted():
    a, errs = scorer.normalise_action({"type": "final_response", "text": "ok", "state_snapshot": {"intent": None, "slots": {}}})
    assert errs == [] and a["type"] == "final"
    a, errs = scorer.normalise_action({"type": "tool_call", "call_id": "c", "name": "t", "arguments": json.dumps({"x": 1})})
    assert errs == [] and a["args"] == {"x": 1}


def test_matchers():
    assert scorer.value_matches({"any_of": ["BOM", "Mumbai"]}, "mumbai")
    assert scorer.value_matches({"regex": "^E2\\d$"}, "e21")
    assert scorer.value_matches(2, "2")
    assert not scorer.value_matches("GOI", "BOM")
    assert scorer.args_contradict({"destination": "GOI"}, {"destination": "BOM", "x": 1})
    assert not scorer.args_contradict({"destination": "GOI"}, {"origin": "DEL"})


def test_forbidden_call_before_event():
    recs = good_trace()
    t = T()
    t.recs = recs
    t.call(1200, "early", "book_flight", flight_id="AI-101", passenger_name="Asha")
    s = scorer.score_trace(sorted(t.recs, key=lambda r: r["t_ms"]), SCENARIO)
    assert any("forbidden" in n for n in s.notes)
    assert s.components["tc_exec"] <= 0.75


def test_excerpt_and_table_render():
    recs = good_trace()
    s = scorer.score_trace(recs, SCENARIO)
    lines = scorer.excerpt(recs, [3])
    assert any(line.startswith("*") for line in lines)
    assert "SUITE" in scorer.format_table([s])


def test_cli_assumptions(capsys):
    assert scorer.main(["--assumptions"]) == 0
    out = capsys.readouterr().out
    assert "A01_total" in out and "A50_qm" in out


def test_same_tick_call_is_response_not_stale_and_reask_same_tick():
    """A->B->A: call issued on the interruption's tick answers it; re-ask on the cancel tick counts."""
    sc = {"id": "aba", "modality": "text", "cancel_grace_ms": 300,
          "expected": {"required_calls": [], "forbidden_calls": [],
                       "final_snapshot": {"intent": "search_flights", "slots": {"destination": "BOM"}}}}
    t = T().add(0, "in", "tool_manifest", type="tool_manifest", tools=MANIFEST)
    t.user(0, "flights to Mumbai", "u1")
    t.call(0, "c1", "flight_search", origin="DEL", destination="BOM", date="d")
    t.user(800, "no, Pune", "u2")
    t.cancel(800, "c1")
    t.call(800, "c2", "flight_search", origin="DEL", destination="PNQ", date="d")
    t.user(1500, "no, Mumbai was right", "u3")
    t.cancel(1500, "c2")
    t.call(1500, "c3", "flight_search", origin="DEL", destination="BOM", date="d")
    t.result(2500, "c3", flights=[{"flight_id": "AI-1"}])
    t.final(2510, "AI-1 flies to Mumbai.", "search_flights", destination="BOM")
    s = scorer.score_trace(t.recs, sc)
    assert s.components["ir_cancel"] == 1.0
    assert s.components["ir_stale"] == 1.0
    assert s.IR == pytest.approx(35.0)
