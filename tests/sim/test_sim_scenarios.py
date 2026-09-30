# tests/sim/test_sim_scenarios.py
import copy
import json
from collections import Counter

import pytest

from conftest import SCENARIO_DIR
from sim import scoring
from sim.harness import run_scenario
from sim.scenario import ScenarioError, from_dict, load_dir
from sim_agents import ORACLE, call, cancel, factory, final, say

SCENARIOS = {s.id: s for s in load_dir(SCENARIO_DIR)}


def test_nine_canonical_scenarios_with_guide_mix():
    assert len(SCENARIOS) == 9
    assert Counter(s.modality for s in SCENARIOS.values()) == {"text": 5, "audio": 3, "visual": 1}


def test_topics_covered():
    ids = " ".join(SCENARIOS)
    for topic in ("self_correction", "destination_change", "chained", "retry", "clarification",
                  "unseen_tool", "duplicate_booking", "cancel_everything", "ambiguous_frame"):
        assert topic in ids


@pytest.mark.parametrize("sid", sorted(SCENARIOS))
def test_expected_block_shape(sid):
    exp = SCENARIOS[sid].expected
    assert isinstance(exp["required_calls"], list) and exp["required_calls"]
    assert isinstance(exp["forbidden_calls"], list) and exp["forbidden_calls"]
    assert "intent" in exp["final_snapshot"]
    assert SCENARIOS[sid].duration_ms <= 120_000


@pytest.mark.parametrize("sid", sorted(SCENARIOS))
def test_media_assets_exist(sid):
    sc = SCENARIOS[sid]
    for e in sc.events:
        if e["type"] in ("audio_clip", "video_frame"):
            p = sc.base_dir / e["path"]
            assert p.exists() and p.stat().st_size > 100
            magic = p.read_bytes()[:4]
            assert magic == (b"RIFF" if e["type"] == "audio_clip" else b"\x89PNG")


@pytest.mark.parametrize("sid", sorted(SCENARIOS))
def test_oracle_passes_every_check(sid, tmp_path):
    sc = SCENARIOS[sid]
    res = run_scenario(sc, factory(ORACLE[sid]), trace_path=tmp_path / f"{sid}.jsonl")
    card = scoring.score(res.records, sc.expected, sid, sc.modality, sc.cancel_grace_ms)
    assert card.passed, [(c.name, c.detail) for c in card.checks if not c.passed]
    assert card.total == pytest.approx(100.0)
    assert res.end_reason == "quiescent"


def _run(sid, script):
    sc = SCENARIOS[sid]
    res = run_scenario(sc, factory(script))
    card = scoring.score(res.records, sc.expected, sid, sc.modality, sc.cancel_grace_ms)
    return {c.name for c in card.checks if not c.passed}, card, res


def test_trap_double_booking_is_caught():
    script = copy.deepcopy(ORACLE["s07_text_duplicate_booking_trap"])
    script["e2"] = [call("c2", "book_flight", flight_id="AI-202", passenger_name="Rahul Verma")]
    bad, card, res = _run("s07_text_duplicate_booking_trap", script)
    assert "no duplicate state-changing calls" in bad and "forbidden[0] book_flight" in bad
    assert len(res.state["bookings"]) == 2


def test_trap_cancel_and_rebook_is_caught():
    script = copy.deepcopy(ORACLE["s07_text_duplicate_booking_trap"])
    script["e2"] = [cancel("c1"), call("c2", "book_flight", flight_id="AI-202", passenger_name="Rahul Verma")]
    script["result:c2:ok"] = script.pop("result:c1:ok")
    bad, _, _ = _run("s07_text_duplicate_booking_trap", script)
    assert "forbidden[0] book_flight" in bad


def test_trap_stale_search_not_cancelled():
    script = copy.deepcopy(ORACLE["s02_text_destination_change"])
    script["e3"] = []
    bad, _, _ = _run("s02_text_destination_change", script)
    assert "must_cancel[0] flight_search" in bad


def test_trap_stale_search_rerun():
    script = copy.deepcopy(ORACLE["s01_audio_self_correction"])
    script["e2"].append(call("c3", "flight_search", origin="DEL", destination="BOM", date="2026-10-12"))
    bad, _, _ = _run("s01_audio_self_correction", script)
    assert "forbidden[0] flight_search" in bad


def test_trap_guessing_instead_of_clarifying():
    script = {"e1": [call("c1", "flight_search", origin="BOM", destination="DEL", date="2026-09-30")],
              "result:c1:ok": [final("UK-955", "search_flights", origin="BOM", destination="DEL", date="2026-09-30")]}
    bad, _, _ = _run("s05_audio_clarification", script)
    assert {"clarification asked", "forbidden[0] flight_search"} <= bad


def test_trap_guessing_the_dryer():
    script = {"e2": [call("c1", "manual_lookup", device_model="DV90T", query="error")],
              "result:c1:ok": [final("x", "troubleshoot", device_model="DV90T")]}
    bad, _, _ = _run("s09_visual_ambiguous_frame", script)
    assert {"forbidden[0] manual_lookup", "forbidden[1] manual_lookup", "clarification asked"} <= bad


def test_trap_retry_storm():
    script = copy.deepcopy(ORACLE["s04_text_retry_after_error"])
    args = dict(origin="CCU", destination="DEL", date="2026-11-02")
    script["result:c1:error"] = [call(f"r{i}", "flight_search", **args) for i in range(4)]
    script["result:r0:ok"] = [final("AI-763", "search_flights", **args)]
    bad, _, _ = _run("s04_text_retry_after_error", script)
    assert "required[1] flight_search" in bad or "required[2] flight_search" in bad


def test_trap_booking_after_cancel_everything():
    script = copy.deepcopy(ORACLE["s08_audio_cancel_everything"])
    script["e3"] = [say("ok", "ack"), call("b1", "book_flight", flight_id="X-1", passenger_name="Anil Kumar"),
                    final("ok", None)]
    bad, _, res = _run("s08_audio_cancel_everything", script)
    assert "forbidden[0] book_flight" in bad


def test_trap_unseen_tool_wrong_enum_is_protocol_error():
    script = {"e1": [call("c1", "upgrade_seat", booking_ref="QX7P2M", cabin="Business Class")],
              "result:c1:error": [final("failed", "upgrade_seat", booking_ref="QX7P2M")]}
    bad, _, _ = _run("s06_text_unseen_tool", script)
    assert {"required[0] upgrade_seat", "no protocol errors (ids, unknown tools, bad args)"} <= bad


def test_loader_rejects_bad_scenarios():
    good = json.loads((SCENARIO_DIR / "s02_text_destination_change.json").read_text())
    for mutate in (lambda d: d.pop("expected"),
                   lambda d: d["expected"].pop("forbidden_calls"),
                   lambda d: d["events"].reverse(),
                   lambda d: d.update(duration_ms=200_000),
                   lambda d: d["expected"]["must_cancel"][0].update(anchor_event="nope"),
                   lambda d: d["events"].append(dict(d["events"][0]))):
        d = copy.deepcopy(good)
        mutate(d)
        with pytest.raises(ScenarioError):
            from_dict(d)


def test_bench_scorer_reads_our_traces_if_present():
    bench = pytest.importorskip("bench.scorer")
    sid = "s03_text_chained_search_book"
    sc = SCENARIOS[sid]
    res = run_scenario(sc, factory(ORACLE[sid]))
    s = bench.score_trace(res.records, sc.raw)
    assert s.total >= 90
