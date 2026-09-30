# /mnt/project-files/theme5/tests/test_harness_scorer.py
"""The old mini harness and proxy scorer, kept as unit-test fixtures."""
import asyncio
import sys
from pathlib import Path

import pytest

FIX = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(FIX))

from mini_harness import MockTool, load_scenario, run_scenario  # noqa: E402
from mini_scorer import score, transcript  # noqa: E402

SCN = sorted((FIX / "scenarios").glob("0*.json"))


@pytest.mark.parametrize("path", SCN, ids=[p.stem for p in SCN])
def test_sample_scenarios_score_full(path):
    res = asyncio.run(run_scenario(load_scenario(path)))
    s = score(res)
    assert not res.timed_out
    assert s.total == 100.0, s.notes
    assert transcript(res)


def test_mock_tool_faults_and_templates():
    m = MockTool("x", False, 100, {"to": "{destination}", "n": "{n}"}, {1: {"attempt": 1, "error": "timeout"}})
    assert m.invoke({"destination": "Goa"})[1]["ok"] is False
    lat, res = m.invoke({"destination": "Goa", "n": 2})
    assert lat == 100 and res["result"] == {"to": "Goa", "n": 2}


def test_scorer_penalises_duplicate_commit_and_missing_cancel():
    scn = load_scenario(SCN[0])
    res = asyncio.run(run_scenario(scn))
    res.commits.append(("search_flights", {"a": 1}))
    res.commits.append(("search_flights", {"a": 1}))
    assert score(res).safety == 0.0
    res2 = asyncio.run(run_scenario(scn))
    res2.actions = [a for a in res2.actions if a["type"] != "cancel"]
    assert score(res2).interruption < 35.0
