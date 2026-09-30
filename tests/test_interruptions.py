# /mnt/project-files/theme5/tests/test_interruptions.py
"""Adversarial interruption timing, run through the simulator (sim/, read-only).

One test per case; each prints its trace (pytest -s) and saves it under
runs/adversarial/. Assertions read the harness trace, never agent internals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.harness import run_scenario  # noqa: E402
from sim.scenario import from_dict  # noqa: E402
from sim.scoring import score  # noqa: E402
from theme5.agent import Agent  # noqa: E402

OUT = ROOT / "runs" / "adversarial"


def flights(dest: str, code: str, *ids: tuple[str, str, int]):
    return {"match": {"destination": {"any_of": [dest, code]}},
            "response": {"flights": [{"flight_id": i, "origin": "DEL", "destination": code, "date": "2026-10-12",
                                      "depart": d, "price_inr": p} for i, d, p in ids]}}


FIX = [flights("Mumbai", "BOM", ("AI-101", "06:00", 5000)),
       flights("Chennai", "MAA", ("6E-201", "07:30", 4200)),
       flights("Pune", "PNQ", ("UK-301", "08:15", 3900)),
       flights("Goa", "GOI", ("SG-401", "09:45", 4100)),
       flights("Kolkata", "CCU", ("AI-501", "11:20", 5600))]


def text(eid, t, s, eot=True):
    return {"id": eid, "t_ms": t, "type": "text_chunk", "text": s, "end_of_turn": eot}


def interrupt(eid, t):
    return {"id": eid, "t_ms": t, "type": "interrupt", "reason": "barge_in"}


def run(sid, events, expected, config, duration=12000):
    scn = from_dict({"id": sid, "modality": "text", "duration_ms": duration, "events": events,
                     "expected": {"required_calls": [], "forbidden_calls": [], "final_snapshot": {}, **expected},
                     "tools": {"enabled": ["flight_search", "book_flight"], "config": config}}, ROOT / "scenarios")
    res = run_scenario(scn, Agent)
    OUT.mkdir(parents=True, exist_ok=True)
    res.trace.write_jsonl(OUT / f"{sid}.jsonl")
    lines = [fmt(r) for r in res.trace.records if fmt(r)]
    (OUT / f"{sid}.txt").write_text("\n".join(lines) + "\n")
    print(f"\n==== {sid} ====")
    print("\n".join(lines))
    card = score(res.trace.records, scn.expected, sid, "text", scn.cancel_grace_ms)
    failed = [f"{c.name}: {c.detail}" for c in card.checks if not c.passed]
    assert res.agent_error is None
    return res.trace.records, failed


def fmt(r):
    d, k = r["data"], r["kind"]
    if r["dir"] == "in" and k in ("text_chunk", "interrupt"):
        return f"{r['t_ms']:>7.0f} IN   {k:<12} {d.get('text', '')}"
    if r["dir"] == "in" and k == "tool_result":
        return f"{r['t_ms']:>7.0f} IN   result       {d.get('call_id')} {d.get('status', 'ok')}"
    if r["dir"] == "out":
        extra = d.get("text") or f"{d.get('call_id')} {d.get('name', '')} {d.get('arguments', '')}".strip()
        return f"{r['t_ms']:>7.0f} OUT  {k:<12} {extra}"
    if r["dir"] == "sys" and k in ("tool_cancelled", "cancel_ignored", "write_committed"):
        return f"{r['t_ms']:>7.0f} SYS  {k:<12} {d.get('call_id')}"
    return ""


def outs(records, kind):
    return [r for r in records if r["dir"] == "out" and r["kind"] == kind]


def calls_to(records, dest):
    return [r for r in outs(records, "tool_call") if r["data"]["arguments"].get("destination") == dest]


def final_text(records):
    return outs(records, "final_response")[-1]["data"]["text"] if outs(records, "final_response") else \
        outs(records, "final")[-1]["data"]["text"]


def finals(records):
    return outs(records, "final_response") or outs(records, "final")


# --------------------------------------------------------------------------- cases


def test_correction_after_result_already_returned():
    rec, failed = run("adv1_correction_after_result", [
        text("e1", 0, "Find flights from Delhi to Mumbai on October 12th."),
        interrupt("e2", 2000),
        text("e3", 2000, "Actually, make it Chennai."),
    ], {"required_calls": [{"tool": "flight_search", "args": {"destination": {"any_of": ["Chennai", "MAA"]}}}],
        "forbidden_calls": [{"tool": "flight_search", "args": {"destination": {"any_of": ["Mumbai", "BOM"]}},
                             "after_event": "e3"}],
        "final_snapshot": {"slots": {"destination": {"any_of": ["Chennai", "MAA"]}, "origin": "Delhi"}},
        "final_mentions": ["6E-201"], "final_must_not_mention": ["AI-101"]},
        {"flight_search": {"latency_ms": 800, "fixtures": FIX}})
    assert not failed, failed
    assert not outs(rec, "cancel")  # nothing in flight to cancel: the Mumbai call had completed
    assert len(finals(rec)) == 2 and "AI-101" in finals(rec)[0]["data"]["text"]
    assert len(calls_to(rec, "Mumbai")) == 1 and len(calls_to(rec, "Chennai")) == 1


def test_two_interruptions_within_50ms():
    rec, failed = run("adv2_double_interrupt_40ms", [
        text("e1", 0, "Find flights from Delhi to Mumbai on October 12th."),
        interrupt("e2", 1000), text("e3", 1000, "no wait, to Pune"),
        interrupt("e4", 1040), text("e5", 1040, "sorry, Goa."),
    ], {"required_calls": [{"tool": "flight_search", "args": {"destination": {"any_of": ["Goa", "GOI"]}}}],
        "must_cancel": [{"tool": "flight_search", "args": {"destination": {"any_of": ["Mumbai", "BOM"]}},
                         "anchor_event": "e3", "within_ms": 5},
                        {"tool": "flight_search", "args": {"destination": {"any_of": ["Pune", "PNQ"]}},
                         "anchor_event": "e5", "within_ms": 5}],
        "forbidden_calls": [{"tool": "flight_search", "args": {"destination": {"any_of": ["Mumbai", "Pune"]}},
                             "after_event": "e5"}],
        "final_snapshot": {"slots": {"destination": {"any_of": ["Goa", "GOI"]}, "origin": "Delhi"}},
        "final_mentions": ["SG-401"], "final_must_not_mention": ["AI-101", "UK-301"]},
        {"flight_search": {"latency_ms": 1500, "fixtures": FIX}})
    assert not failed, failed
    cancels = [(r["t_ms"], r["data"]["call_id"]) for r in outs(rec, "cancel")]
    assert cancels == [(1000.0, "call-0001"), (1040.0, "call-0002")]
    assert len(finals(rec)) == 1


def test_interruption_during_a_retry():
    rec, failed = run("adv3_interrupt_during_retry", [
        text("e1", 0, "Find flights from Delhi to Mumbai on October 12th."),
        interrupt("e2", 800), text("e3", 800, "Actually, to Kolkata."),
    ], {"required_calls": [{"tool": "flight_search", "args": {"destination": {"any_of": ["Kolkata", "CCU"]}}}],
        "must_cancel": [{"tool": "flight_search", "args": {"destination": {"any_of": ["Mumbai", "BOM"]}},
                         "anchor_event": "e3", "within_ms": 5}],
        "forbidden_calls": [{"tool": "flight_search", "args": {"destination": {"any_of": ["Mumbai", "BOM"]}},
                             "after_event": "e3"}],
        "final_snapshot": {"slots": {"destination": {"any_of": ["Kolkata", "CCU"]}}},
        "final_mentions": ["AI-501"], "final_must_not_mention": ["AI-101"]},
        {"flight_search": {"latency_ms": 1500, "fixtures": FIX,
                           "faults": [{"kind": "error", "call_index": 1, "error": "503 upstream_unavailable",
                                       "latency_ms": 300}]}})
    assert not failed, failed
    mumbai = calls_to(rec, "Mumbai")
    assert len(mumbai) == 2  # original + one retry, both before the interruption
    assert [r["data"]["call_id"] for r in outs(rec, "cancel")] == [mumbai[1]["data"]["call_id"]]
    assert "try" not in final_text(rec).lower()  # the new search is a fresh call, not "retrying"


def test_cancel_request_after_call_already_failed():
    rec, failed = run("adv4_cancel_after_failure", [
        text("e1", 0, "Book flight AI-202 for Rahul Verma."),
        text("e2", 1000, "Cancel that."),
    ], {"required_calls": [{"tool": "book_flight", "status": "error"}], "max_writes": {"book_flight": 0}},
        {"book_flight": {"latency_ms": 300, "faults": [{"kind": "error", "call_index": 1, "error": "payment_declined",
                                                      "retryable": False}]}})
    assert not failed, failed
    assert not outs(rec, "cancel")  # never cancel a call that already finished
    assert not [r for r in rec if r["kind"] == "cancel_ignored"]
    assert len(outs(rec, "tool_call")) == 1  # non-retryable write failure: no retry
    texts = [f["data"]["text"] for f in finals(rec)]
    assert "nothing was changed" in texts[0] and "No changes were made" in texts[1]
    assert finals(rec)[-1]["data"]["state_snapshot"] == {"intent": None, "slots": {}}


def test_speculative_read_is_cancelled_and_reissued_when_final_words_change_it():
    rec, failed = run("adv5_speculation_revised", [
        text("e1", 0, "Find flights from Delhi to Mumbai on October 12th", eot=False),
        text("e2", 600, "no, to Pune.", eot=True),
    ], {"required_calls": [{"tool": "flight_search", "args": {"destination": {"any_of": ["Pune", "PNQ"]}}}],
        "must_cancel": [{"tool": "flight_search", "args": {"destination": {"any_of": ["Mumbai", "BOM"]}},
                         "anchor_event": "e2", "within_ms": 5}],
        "final_snapshot": {"slots": {"destination": {"any_of": ["Pune", "PNQ"]}}},
        "final_mentions": ["UK-301"]},
        {"flight_search": {"latency_ms": 1500, "fixtures": FIX}})
    assert not failed, failed
    first = outs(rec, "tool_call")[0]
    assert first["t_ms"] == 0.0 and first["data"]["arguments"]["destination"] == "Mumbai"  # speculative
    assert not [s for s in outs(rec, "speak") if s["t_ms"] < 600]  # silent while the user is talking


def test_speculation_confirmed_by_final_words_is_adopted_not_reissued():
    rec, failed = run("adv6_speculation_adopted", [
        text("e1", 0, "Find flights from Delhi to Goa on October 12th", eot=False),
        text("e2", 500, "please.", eot=True),
    ], {"required_calls": [{"tool": "flight_search", "args": {"destination": {"any_of": ["Goa", "GOI"]}},
                            "max_count": 1}],
        "final_snapshot": {"slots": {"destination": {"any_of": ["Goa", "GOI"]}}}, "final_mentions": ["SG-401"]},
        {"flight_search": {"latency_ms": 1500, "fixtures": FIX}})
    assert not failed, failed
    assert len(outs(rec, "tool_call")) == 1 and not outs(rec, "cancel")
    assert finals(rec)[0]["t_ms"] == 1500.0  # result time: speculation hid 500 ms of latency


@pytest.mark.parametrize("case,partial", [("write", "Book flight AI-202 for Rahul Verma"),
                                          ("mid_repair", "Find flights from Delhi to Mumbai, uh, no wait")])
def test_never_speculate_on_writes_or_mid_repair(case, partial):
    rec, _ = run(f"adv7_no_speculation_{case}", [text("e1", 0, partial, eot=False)], {},
                 {"flight_search": {"latency_ms": 500, "fixtures": FIX}}, duration=3000)
    assert not outs(rec, "tool_call")
