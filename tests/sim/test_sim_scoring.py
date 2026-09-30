# tests/sim/test_sim_scoring.py
from sim import scoring


def rec(seq, t, d, kind, data):
    return {"seq": seq, "t_ms": t, "dir": d, "kind": kind, "data": data}


def base_records(cancel_at=None, rerun=False, final_slots=None, dup=False):
    r = [rec(0, 0, "in", "text_chunk", {"type": "text_chunk", "event_id": "e1", "end_of_turn": True}),
         rec(1, 50, "out", "speak", {"type": "speak", "kind": "ack", "text": "ok"}),
         rec(2, 60, "out", "tool_call", {"type": "tool_call", "call_id": "c1", "tool": "flight_search", "args": {"destination": "DEL"}}),
         rec(3, 60, "sys", "tool_started", {"call_id": "c1", "tool": "flight_search", "args": {"destination": "DEL"}, "side_effect": "read"}),
         rec(4, 1000, "in", "interrupt", {"type": "interrupt", "event_id": "e2"})]
    if cancel_at is not None:
        r += [rec(5, cancel_at, "out", "cancel", {"type": "cancel", "call_id": "c1"}),
              rec(6, cancel_at, "sys", "tool_cancelled", {"call_id": "c1", "tool": "flight_search"})]
    else:
        r += [rec(5, 3000, "sys", "tool_completed", {"call_id": "c1", "tool": "flight_search", "status": "ok"})]
    args2 = {"destination": "DEL" if rerun else "HYD"}
    r += [rec(10, 1100, "out", "tool_call", {"type": "tool_call", "call_id": "c2", "tool": "flight_search", "args": args2}),
          rec(11, 1100, "sys", "tool_started", {"call_id": "c2", "tool": "flight_search", "args": args2, "side_effect": "read"}),
          rec(12, 2000, "sys", "tool_completed", {"call_id": "c2", "tool": "flight_search", "status": "ok"})]
    for i in range(2 if dup else 1):
        cid = f"b{i}"
        r += [rec(20 + 3 * i, 2100 + i, "sys", "tool_started", {"call_id": cid, "tool": "book_flight", "args": {"flight_id": "X"}, "side_effect": "write"}),
              rec(21 + 3 * i, 2500 + i, "sys", "tool_completed", {"call_id": cid, "tool": "book_flight", "status": "ok"})]
    r.append(rec(30, 3100, "out", "final", {"type": "final", "text": "Booked 6E-6122",
                                             "snapshot": {"intent": "book_flight", "slots": final_slots or {"destination": "HYD"}}}))
    return sorted(r, key=lambda x: (x["t_ms"], x["seq"]))


EXP = {
    "required_calls": [{"tool": "flight_search", "args": {"destination": "HYD"}}, {"tool": "book_flight"}],
    "ordered_calls": True,
    "forbidden_calls": [{"tool": "flight_search", "args": {"destination": "DEL"}, "after_event": "e2"}],
    "must_cancel": [{"tool": "flight_search", "args": {"destination": "DEL"}, "anchor_event": "e2"}],
    "max_writes": {"book_flight": 1},
    "final_snapshot": {"intent": "book_flight", "slots": {"destination": {"any_of": ["HYD", "Hyderabad"]}}},
    "final_mentions": [{"any_of": ["6E-6122"]}],
    "final_must_not_mention": ["AI-804"],
}


def test_perfect_trace_scores_100():
    card = scoring.score(base_records(cancel_at=1000), EXP, "x", "text", 300)
    assert card.passed, [c for c in card.checks if not c.passed]
    assert (card.scores["task"], card.scores["interruption"], card.scores["safety"]) == (40, 35, 10)


def test_late_cancel_fails_must_cancel():
    card = scoring.score(base_records(cancel_at=1500), EXP, cancel_grace_ms=300)
    bad = {c.name for c in card.checks if not c.passed}
    assert bad == {"must_cancel[0] flight_search"}
    assert card.scores["interruption"] < 35


def test_missing_cancel_and_stale_rerun():
    card = scoring.score(base_records(cancel_at=None, rerun=True), EXP)
    bad = {c.name for c in card.checks if not c.passed}
    assert {"must_cancel[0] flight_search", "forbidden[0] flight_search", "required[0] flight_search"} <= bad


def test_duplicate_write_detected():
    card = scoring.score(base_records(cancel_at=1000, dup=True), EXP)
    bad = {c.name for c in card.checks if not c.passed}
    assert "no duplicate state-changing calls" in bad and "max_writes book_flight<=1" in bad
    assert card.scores["safety"] < 10


def test_snapshot_mismatch_and_grounding():
    exp = dict(EXP, final_mentions=["QP-1345"])
    card = scoring.score(base_records(cancel_at=1000, final_slots={"destination": "DEL"}), exp)
    bad = {c.name for c in card.checks if not c.passed}
    assert {"final snapshot matches", "final text grounded"} <= bad


def test_latency_curve():
    assert scoring._lat_score(100) == 1.0
    assert scoring._lat_score(2000) == 0.0
    assert scoring._lat_score(None) == 0.0
    assert abs(scoring._lat_score(1150) - 0.5) < 1e-9


def test_latency_measured_per_user_turn():
    card = scoring.score(base_records(cancel_at=1000), EXP)
    # e1 answered by speak at 50 ms; interrupt e2 at 1000 answered only by the final at 3100
    assert card.latencies_ms == [50, 2100]


def test_clarify_window():
    recs = [rec(0, 0, "in", "text_chunk", {"type": "text_chunk", "event_id": "e1", "end_of_turn": True}),
            rec(1, 10, "out", "clarify", {"type": "clarify", "text": "from where?"}),
            rec(2, 5000, "in", "text_chunk", {"type": "text_chunk", "event_id": "e2", "end_of_turn": True})]
    exp = {"required_calls": [], "forbidden_calls": [], "final_snapshot": None,
           "clarify": {"required": True, "after_event": "e1", "before_event": "e2"}}
    ok = {c.name: c.passed for c in scoring.score(recs, exp).checks}
    assert ok["clarification asked"]
    late = [recs[0], recs[2], rec(3, 5100, "out", "clarify", {"type": "clarify", "text": "?"})]
    ok = {c.name: c.passed for c in scoring.score(late, exp).checks}
    assert not ok["clarification asked"]


def test_invalid_actions_and_protocol_errors_hit_safety():
    recs = [rec(0, 0, "out", "invalid", {"errors": ["x"]}),
            rec(1, 0, "sys", "tool_started", {"call_id": "z", "tool": "nope", "args": {}, "rejected": "unknown_tool"}),
            rec(2, 0, "sys", "cancel_ignored", {"call_id": "q", "reason": "unknown"})]
    card = scoring.score(recs, {"required_calls": [], "forbidden_calls": [], "final_snapshot": None})
    assert card.scores["safety"] == 5.0


def test_aggregate_weights_multimodal():
    a = scoring.ScoreCard("a", "text", total=100)
    b = scoring.ScoreCard("b", "audio", total=50)
    agg = scoring.aggregate([a, b])
    assert agg["mean_total"] == 75 and agg["weighted_total"] == 70
