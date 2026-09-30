# sim/run.py
"""Runner: replay scenarios against an agent, write traces, print a report.

    python -m sim.run scenarios/ --report
    python -m sim.run scenarios/s02_text_destination_change.json --agent theme5.agent:Agent -v

Outputs (under --out, default runs/latest):
    traces/<scenario_id>.jsonl   full event/action trace (sim/trace.py format)
    report.json                  per-scenario checks and scores
    report.md                    same, human-readable (with --report)

Scores: `sim.scoring` checks each scenario's `expected` block (pass/fail). The
rubric (TC/IR/LAT/SP/QM) comes from `bench.scorer` when importable, else from
the sim estimate. Both are proxies; the official scorer is unreleased.
Exit code 0 when every scenario passes its expected checks, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import scoring
from .adapter import DEFAULT_AGENT, load_agent_factory, load_codec
from .harness import HarnessConfig, run_scenario
from .scenario import Scenario, load_dir


def _bench():
    try:
        from bench import scorer as bench_scorer  # type: ignore
        return bench_scorer
    except Exception:  # noqa: BLE001 - optional
        return None


def run_all(scenarios: List[Scenario], agent_spec: str, out: Path, codec_spec: Optional[str] = None,
            config: Optional[HarnessConfig] = None, factory=None) -> List[Dict[str, Any]]:
    factory = factory or load_agent_factory(agent_spec)
    codec = load_codec(codec_spec, agent_spec)
    bench = _bench()
    rows = []
    for sc in scenarios:
        tpath = out / "traces" / f"{sc.id}.jsonl"
        res = run_scenario(sc, factory, codec, config, trace_path=tpath)
        card = scoring.score(res.records, sc.expected, sc.id, sc.modality, sc.cancel_grace_ms)
        row = {"scenario_id": sc.id, "title": sc.title, "modality": sc.modality,
               "end_reason": res.end_reason, "virtual_ms": round(res.virtual_ms, 1),
               "wall_s": round(res.wall_s, 3), "agent_error": res.agent_error,
               "trace": str(tpath), "sim": card.to_dict()}
        if bench is not None:
            try:
                row["bench"] = bench.score_trace(res.records, sc.raw).to_dict()
            except Exception as exc:  # noqa: BLE001 - never let the optional scorer kill a run
                row["bench_error"] = repr(exc)
        rows.append(row)
    return rows


def summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    w = [scoring.MULTIMODAL_WEIGHT if r["modality"] != "text" else 1.0 for r in rows]
    totals = [_rubric(r)["total"] for r in rows]
    return {"scenarios": n, "passed": sum(r["sim"]["passed"] for r in rows),
            "rubric": "bench" if rows and all("bench" in r for r in rows) else "sim estimate",
            "weighted_total": round(sum(t * x for t, x in zip(totals, w)) / sum(w), 2) if n else 0.0}


def _rubric(r: Dict[str, Any]) -> Dict[str, float]:
    """One rubric per row: bench/scorer.py when available, else the sim estimate."""
    b = r.get("bench")
    if b is not None:
        return {"TC": b["TC"], "IR": b["IR"], "LAT": b["LAT"], "SP": b["SP"], "QM": b["QM"], "total": b["total"]}
    sc = r["sim"]["scores"]
    return {"TC": sc["task"], "IR": sc["interruption"], "LAT": sc["latency"], "SP": sc["safety"], "QM": 1.0,
            "total": r["sim"]["total"]}


def format_table(rows: List[Dict[str, Any]]) -> str:
    src = "bench" if rows and all("bench" in r for r in rows) else "sim estimate"
    head = (f"{'scenario':<34} {'mod':<6} {'pass':<4} {'TC':>5} {'IR':>5} {'LAT':>5} {'SP':>5} {'QM':>5} "
            f"{'total':>6}  end")
    lines = [f"rubric: {src}", head, "-" * len(head)]
    for r in rows:
        x = _rubric(r)
        lines.append(f"{r['scenario_id'][:34]:<34} {r['modality']:<6} {'yes' if r['sim']['passed'] else 'NO':<4} "
                     f"{x['TC']:>5.1f} {x['IR']:>5.1f} {x['LAT']:>5.1f} {x['SP']:>5.1f} {x['QM']:>5.2f} "
                     f"{x['total']:>6.1f}  {r['end_reason']}")
    return "\n".join(lines)


def format_markdown(rows: List[Dict[str, Any]], agent_spec: str) -> str:
    s = summary(rows)
    md = ["# Scenario report", "", f"Agent: `{agent_spec}`  ",
          f"Passed {s['passed']}/{s['scenarios']} expected-block checks; rubric ({s['rubric']}, "
          f"multimodal x1.5): {s['weighted_total']}", "",
          "```", format_table(rows), "```", ""]
    for r in rows:
        fails = [c for c in r["sim"]["checks"] if not c["passed"]]
        md.append(f"## {r['scenario_id']} ({'pass' if r['sim']['passed'] else 'FAIL'})")
        md.append(f"{r['title']}. End: {r['end_reason']} at {r['virtual_ms']} ms virtual, "
                  f"{r['wall_s']} s wall. Latencies (ms): {r['sim']['latencies_ms']}. Trace: `{r['trace']}`")
        if r.get("agent_error"):
            md.append(f"- agent error: `{r['agent_error']}`")
        for c in fails:
            md.append(f"- FAIL {c['name']}: {c['detail']}")
        for note in (r.get("bench") or {}).get("notes", [])[:6]:
            md.append(f"- bench: {note}")
        md.append("")
    return "\n".join(md)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim.run", description="Replay Theme 05 scenarios against an agent.")
    ap.add_argument("path", type=Path, help="scenario file or directory")
    ap.add_argument("--agent", default=DEFAULT_AGENT, help=f"module:attr (default {DEFAULT_AGENT})")
    ap.add_argument("--codec", default=None, help="module with decode_event/encode_action (optional)")
    ap.add_argument("--out", type=Path, default=Path("runs/latest"))
    ap.add_argument("--report", action="store_true", help="write report.md and print per-scenario failures")
    ap.add_argument("--only", action="append", default=[], help="run only scenario ids containing this text")
    ap.add_argument("--strip-oracle", action="store_true", help="drop transcript/labels from media events")
    ap.add_argument("--charge-compute", action="store_true",
                    help="advance the virtual clock by real agent compute time (latency realism, less deterministic)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    skipped: List[Path] = []
    scenarios = load_dir(a.path, skipped)
    if a.only:
        scenarios = [s for s in scenarios if any(o in s.id for o in a.only)]
    for f in skipped:
        print(f"skipped (not a sim scenario): {f}", file=sys.stderr)
    if not scenarios:
        print("no scenarios to run", file=sys.stderr)
        return 2
    cfg = HarnessConfig(strip_oracle=a.strip_oracle, charge_compute=a.charge_compute)
    rows = run_all(scenarios, a.agent, a.out, a.codec, cfg)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps({"summary": summary(rows), "scenarios": rows}, indent=1, default=str))
    print(format_table(rows))
    print(json.dumps(summary(rows)))
    if a.report:
        md = format_markdown(rows, a.agent)
        (a.out / "report.md").write_text(md)
        for r in rows:
            fails = [c for c in r["sim"]["checks"] if not c["passed"]]
            if fails or a.verbose:
                print(f"\n{r['scenario_id']}:")
                for c in (r["sim"]["checks"] if a.verbose else fails):
                    print(f"  {'ok  ' if c['passed'] else 'FAIL'} {c['name']}: {c['detail']}")
        print(f"\nreport: {a.out / 'report.md'}")
    return 0 if all(r["sim"]["passed"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
