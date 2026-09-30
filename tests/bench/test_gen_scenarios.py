# tests/bench/test_gen_scenarios.py
from __future__ import annotations

import json
from collections import Counter

import pytest

from bench import gen_scenarios as g
from bench import scorer
from bench.run_suite import shim_action


@pytest.fixture(scope="module")
def suite():
    return g.generate(60, seed=5)


def test_mix_and_ids(suite):
    assert len(suite) == 60
    assert Counter(s["modality"] for s in suite) == {"text": 30, "audio": 18, "visual": 12}
    assert len({s["id"] for s in suite}) == 60
    assert {s["family"] for s in suite} == set(g.FAMILIES)


def test_deterministic_per_seed(suite):
    assert json.dumps(g.generate(60, seed=5)) == json.dumps(suite)
    assert json.dumps(g.generate(60, seed=6)) != json.dumps(suite)


def test_other_sizes_keep_the_mix():
    small = g.generate(10, seed=1)
    assert Counter(s["modality"] for s in small) == {"text": 5, "audio": 3, "visual": 2}


def test_valid_for_harness_loader(suite):
    scenario = pytest.importorskip("sim.scenario")
    for s in suite:
        scenario.from_dict(s)  # raises ScenarioError on any schema problem


def test_events_sorted_and_modality_consistent(suite):
    for s in suite:
        ts = [e["t_ms"] for e in s["events"]]
        assert ts == sorted(ts) and ts[-1] < s["duration_ms"] <= 120000
        kinds = {e["type"] for e in s["events"]}
        if s["modality"] == "audio":
            assert "audio_clip" in kinds and "text_chunk" not in kinds
        if s["modality"] == "visual":
            assert "video_frame" in kinds
        if s["modality"] == "text":
            assert kinds <= {"text_chunk", "interrupt"}


def test_adversarial_features_present(suite):
    fam = {f: [s for s in suite if s["family"] == f] for f in g.FAMILIES}
    # near-simultaneous: correction within 60 ms of the interrupt
    for s in fam["near_simul"]:
        ti = next(e["t_ms"] for e in s["events"] if e["type"] == "interrupt")
        assert s["events"][-1]["t_ms"] - ti <= 60
    # hesitation: >= 2 s gaps before end of turn
    for s in fam["hesitation"]:
        ts = [e["t_ms"] for e in s["events"]]
        assert min(b - a for a, b in zip(ts, ts[1:])) >= 2000
    # faults injected
    assert all(any("faults" in c for c in s["tools"]["config"].values()) for s in fam["fault"])
    # unseen tools carry unusual parameter names
    names = {p for s in fam["unseen"] for t in s["tools"].get("extra", [])
             for p in (t["parameters"].get("properties") or {})}
    assert {"venueSlug", "party_sz", "awb_no"} & names
    # duplicate traps cap writes
    assert all(s["expected"].get("max_write_calls") for s in fam["dup_trap"])


def test_expectations_are_not_vacuous(suite):
    """An agent that does nothing must score badly on every scenario."""
    for s in suite:
        empty = [{"seq": 0, "t_ms": 0, "dir": "in", "kind": e["type"], "data": {**e, "event_id": e["id"]}}
                 for e in s["events"]]
        assert scorer.score_trace(empty, s).total < 65, s["id"]


def test_write_creates_files_and_assets(tmp_path):
    scs = g.generate(10, seed=2)
    paths = g.write(scs, tmp_path)
    assert len(paths) == 10 and all(p.exists() for p in paths)
    pytest.importorskip("sim.assets")
    for s in scs:
        for e in s["events"]:
            if "path" in e:
                assert (tmp_path / e["path"]).exists()


def test_shim_maps_protocol_names_to_wire():
    a = shim_action({"type": "tool_call", "call_id": "c1", "name": "t", "arguments": {"x": 1}, "state_snapshot": {"intent": None, "slots": {}}})
    assert a["tool"] == "t" and a["args"] == {"x": 1} and "snapshot" in a
    assert shim_action({"type": "final_response", "text": "x"})["type"] == "final"
    assert shim_action({"type": "speak", "text": "x", "kind": "info"})["kind"] == "answer"
