# tests/sim/test_sim_run.py
import json

import pytest

from conftest import SCENARIO_DIR
from sim import run


def test_cli_runs_all_and_writes_report(tmp_path, capsys):
    code = run.main([str(SCENARIO_DIR), "--agent", "sim_agents:FinalOnTurn", "--out", str(tmp_path), "--report"])
    assert code == 1  # a do-nothing agent must fail the expected checks
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["summary"]["scenarios"] == 9 and rep["summary"]["passed"] == 0
    assert (tmp_path / "report.md").exists()
    assert len(list((tmp_path / "traces").glob("*.jsonl"))) == 9
    out = capsys.readouterr().out
    assert "s07_text_duplicate_booking_trap" in out and "FAIL" in out


def test_cli_only_filter(tmp_path):
    code = run.main([str(SCENARIO_DIR), "--agent", "sim_agents:FinalOnTurn", "--out", str(tmp_path),
                     "--only", "s06"])
    rep = json.loads((tmp_path / "report.json").read_text())
    assert [r["scenario_id"] for r in rep["scenarios"]] == ["s06_text_unseen_tool"]
    assert code == 1


def test_real_agent_smoke(tmp_path):
    """The team's agent runs through every scenario without crashing the harness.
    Pass/fail of its behaviour is reported, not asserted."""
    pytest.importorskip("theme5.agent")
    rows = run.run_all(run.load_dir(SCENARIO_DIR), "theme5.agent:Agent", tmp_path)
    assert len(rows) == 9
    assert all(r["agent_error"] is None for r in rows)
    assert all(not any(c["name"] == "all actions schema-valid" and not c["passed"] for c in r["sim"]["checks"])
               for r in rows)
