# bench/run_suite.py
"""Run a scenario directory through the harness (sim/) with an agent, score every
trace with bench.scorer, and print the score table plus the worst failures.

    python -m bench.run_suite bench/scenarios60 --agent theme5.agent:Agent \
        --out bench/runs/latest [--shim] [--worst 5]

--shim translates the agent's action field names to the harness wire format
(sim/wire.py) before the harness sees them: tool_call name/arguments -> tool/args,
final_response -> final, state_snapshot -> snapshot, speak kind info -> answer.
It exists only to measure agent behaviour while protocol.py and wire.py still
disagree; the unshimmed run is the honest integration number.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from bench import scorer

WIRE_RENAMES = {"final_response": "final"}


def shim_action(a: Any) -> Any:
    if not isinstance(a, dict):
        return a
    a = dict(a)
    a["type"] = WIRE_RENAMES.get(a.get("type"), a.get("type"))
    if a["type"] == "tool_call":
        if "tool" not in a and "name" in a:
            a["tool"] = a.pop("name")
        if "args" not in a and "arguments" in a:
            a["args"] = a.pop("arguments")
    if "snapshot" not in a and "state_snapshot" in a:
        a["snapshot"] = a.pop("state_snapshot")
    if a["type"] == "speak" and a.get("kind") not in ("filler", "ack", "progress", "answer"):
        a["kind"] = "answer"
    return a


class ShimAgent:
    def __init__(self, inner: Any):
        self.inner = inner

    async def run(self, inbox: asyncio.Queue, outbox: asyncio.Queue) -> None:
        mid: asyncio.Queue = asyncio.Queue()

        async def pump() -> None:
            while True:
                outbox.put_nowait(shim_action(await mid.get()))

        p = asyncio.get_running_loop().create_task(pump())
        try:
            await self.inner.run(inbox, mid)
        finally:
            while not mid.empty():
                outbox.put_nowait(shim_action(mid.get_nowait()))
            p.cancel()


def run(scen_dir: Path, agent_spec: str, out: Path, shim: bool = False,
        only: Optional[List[str]] = None, charge_compute: bool = False,
        strip_oracle: bool = False) -> List[Dict[str, Any]]:
    from sim import adapter, harness, scenario

    base = adapter.load_agent_factory(agent_spec)
    factory: Callable[[], Any] = (lambda: ShimAgent(base())) if shim else base
    codec = adapter.load_codec(None, agent_spec)
    cfg = harness.HarnessConfig(charge_compute=charge_compute, strip_oracle=strip_oracle)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sc in scenario.load_dir(scen_dir):
        if only and sc.id not in only:
            continue
        tp = out / f"{sc.id}.jsonl"
        t0 = time.perf_counter()
        err = None
        try:
            res = harness.run_scenario(sc, factory, codec, cfg, trace_path=tp)
            records = res.records
        except Exception as exc:  # a harness crash is reported, never fatal to the suite
            err = f"{type(exc).__name__}: {exc}"
            records = []
            traceback.print_exc(file=sys.stderr)
        wall = time.perf_counter() - t0
        s = scorer.score_trace(records, sc.raw)
        if err:
            s.notes.insert(0, f"HARNESS CRASH: {err}")
        rows.append({"score": s, "records": records, "wall_s": wall})
    return rows


def report(rows: List[Dict[str, Any]], worst: int = 5, label: str = "") -> str:
    scores = [r["score"] for r in rows]
    lines = [f"# Scored run {label}".rstrip(), "", "```", scorer.format_table(scores), "```", ""]
    agg = scorer.suite_score(scores)
    lines += [f"Suite (multimodal x1.5): **{agg['weighted_mean']}**, unweighted {agg['mean']}, "
              f"by modality {agg.get('by_modality')}, wall time max "
              f"{max((r['wall_s'] for r in rows), default=0):.2f}s", ""]
    fams: Dict[str, List[float]] = {}
    for s in scores:
        fams.setdefault(s.scenario_id.split("_", 2)[-1], []).append(s.total)
    lines += ["By family: " + ", ".join(f"{k} {sum(v) / len(v):.1f}" for k, v in sorted(fams.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))), ""]
    lines += [f"## {worst} worst", ""]
    for r in sorted(rows, key=lambda r: r["score"].total)[:worst]:
        s = r["score"]
        lines.append(f"### {s.scenario_id}: {s.total:.1f} (TC {s.TC} IR {s.IR} LAT {s.LAT} SP {s.SP} QM {s.QM})")
        lines += [f"- {n}" for n in s.notes[:8]]
        lines += ["```"] + scorer.excerpt(r["records"], s.evidence_seq) + ["```", ""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scenarios", type=Path)
    ap.add_argument("--agent", default="theme5.agent:Agent")
    ap.add_argument("--out", type=Path, default=Path("bench/runs/latest"))
    ap.add_argument("--shim", action="store_true")
    ap.add_argument("--worst", type=int, default=5)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--charge-compute", action="store_true",
                    help="add real agent compute time to the virtual clock (measures true latency)")
    ap.add_argument("--strip-oracle", action="store_true", help="drop oracle transcripts/labels from media events")
    a = ap.parse_args(argv)
    rows = run(a.scenarios, a.agent, a.out, a.shim, a.only, a.charge_compute, a.strip_oracle)
    flags = [f for f, on in (("shimmed", a.shim), ("compute charged", a.charge_compute), ("oracle stripped", a.strip_oracle)) if on]
    text = report(rows, a.worst, f"({', '.join(flags)})" if flags else "")
    (a.out / "REPORT.md").write_text(text + "\n", encoding="utf-8")
    (a.out / "scores.json").write_text(json.dumps([r["score"].to_dict() for r in rows], indent=1, default=str))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
