# /mnt/project-files/theme5/bench/profile_suite.py
"""Profile a scenario suite: time to first spoken action, blocking callbacks on
the event loop, memory growth across scenarios, and wall/virtual time per
scenario against the 120 s cap.

    python -m bench.profile_suite bench/scenarios60 --agent theme5.agent:Agent \
        --out bench/runs/profile [--repeat 3] [--block-ms 5] [--cprofile]

Runs every scenario on the sim harness with compute charged to the virtual
clock (so slow agent code shows up as latency), and:
- first spoken action: per user turn boundary (end-of-turn text/audio,
  interrupt, frame), ms until the agent's first speak/clarify/final_response;
  also until its first action of any kind.
- blocking calls: every event-loop callback that ran longer than --block-ms of
  real time, with the callback's repr (asyncio's Handle._run is wrapped).
- memory: tracemalloc current size after each scenario (after gc); growth from
  the first to the last scenario and the top growth sites.
- 120 s: wall seconds and virtual ms per scenario, and the harness end reason.
Writes PROFILE.md and profile.json under --out.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

SPOKEN = ("speak", "clarify", "final", "final_response")
BOUNDARY_KINDS = ("interrupt", "frame", "video_frame")


def _is_boundary(rec: Dict[str, Any]) -> bool:
    if rec.get("dir") != "in":
        return False
    k, d = rec.get("kind"), rec.get("data") or {}
    if k in BOUNDARY_KINDS:
        return True
    if k in ("text", "text_chunk", "audio", "audio_clip", "end_of_turn"):
        return k == "end_of_turn" or bool(d.get("end_of_turn", d.get("final", k.startswith("audio"))))
    return False


MERGE_MS = 1500.0  # same rule as bench.scorer A30: a silent boundary this close to the next one merges into it


def first_action_latencies(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per boundary: ms to the first spoken action and to the first action.
    A boundary with no spoken action before a next boundary within MERGE_MS
    is merged into it (a frame shown just before the question, a bare
    interrupt followed by words)."""
    out: List[Dict[str, Any]] = []
    recs = sorted(records, key=lambda r: (r["t_ms"], r["seq"]))
    for i, r in enumerate(recs):
        if not _is_boundary(r):
            continue
        spoken = anyact = None
        nxt = None
        for x in recs[i + 1:]:
            if x["dir"] == "in" and _is_boundary(x):
                nxt = x["t_ms"]
                break
            if x["dir"] != "out":
                continue
            if anyact is None:
                anyact = x["t_ms"] - r["t_ms"]
            if x["kind"] in SPOKEN:
                spoken = x["t_ms"] - r["t_ms"]
                break
        if spoken is None and nxt is not None and nxt - r["t_ms"] < MERGE_MS:
            continue
        out.append({"t_ms": r["t_ms"], "kind": r["kind"], "spoken_ms": spoken, "first_action_ms": anyact})
    return out


class BlockMonitor:
    """Wrap asyncio.Handle._run to record callbacks that hold the loop too long."""

    def __init__(self, threshold_ms: float) -> None:
        self.threshold_s = threshold_ms / 1000.0
        self.hits: List[Dict[str, Any]] = []
        self.total_s = 0.0
        self.count = 0
        self.scenario = ""
        self._orig: Optional[Callable[..., Any]] = None
        self.gc_pauses: List[Dict[str, Any]] = []
        self._gc_t0 = 0.0

    def _gc(self, phase: str, info: Dict[str, Any]) -> None:
        if phase == "start":
            self._gc_t0 = time.perf_counter()
        else:
            self.gc_pauses.append({"scenario": self.scenario, "gen": info.get("generation"),
                                   "ms": round((time.perf_counter() - self._gc_t0) * 1000, 3)})

    def __enter__(self) -> "BlockMonitor":
        from asyncio import events

        orig = events.Handle._run
        mon = self

        def _run(handle: Any) -> None:
            t0 = time.perf_counter()
            try:
                orig(handle)
            finally:
                dt = time.perf_counter() - t0
                mon.total_s += dt
                mon.count += 1
                if dt >= mon.threshold_s:
                    mon.hits.append({"scenario": mon.scenario, "ms": round(dt * 1000, 2),
                                     "callback": _describe(handle)})

        self._orig = orig
        events.Handle._run = _run  # type: ignore[method-assign]
        gc.callbacks.append(self._gc)
        return self

    def __exit__(self, *exc: Any) -> None:
        from asyncio import events

        events.Handle._run = self._orig  # type: ignore[method-assign]
        gc.callbacks.remove(self._gc)


def _describe(handle: Any) -> str:
    cb = getattr(handle, "_callback", None)
    task = getattr(cb, "__self__", None)
    if isinstance(task, asyncio.Task):
        coro = task.get_coro()
        code = getattr(coro, "cr_code", None)
        frame = getattr(coro, "cr_frame", None)
        where = f"{code.co_filename.rsplit('/', 1)[-1]}:{frame.f_lineno}" if code and frame else ""
        return f"task {getattr(coro, '__qualname__', coro)} {where}".strip()
    return repr(cb)[:160]


def _agent_only(snap: "tracemalloc.Snapshot") -> "tracemalloc.Snapshot":
    """Drop the profiler's own bookkeeping (rows, snapshots) from a snapshot."""
    return snap.filter_traces((tracemalloc.Filter(False, __file__, all_frames=True),
                               tracemalloc.Filter(False, tracemalloc.__file__)))


def _pct(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round(p / 100 * (len(xs) - 1)))))
    return round(xs[k], 2)


def profile(scen_dir: Path, agent_spec: str, repeat: int = 1, block_ms: float = 5.0,
            charge_compute: bool = True, only: Optional[List[str]] = None) -> Dict[str, Any]:
    from sim import adapter, harness, scenario

    factory = adapter.load_agent_factory(agent_spec)
    codec = adapter.load_codec(None, agent_spec)
    cfg = harness.HarnessConfig(charge_compute=charge_compute)
    scs = [s for s in scenario.load_dir(scen_dir) if not only or s.id in only]
    rows: List[Dict[str, Any]] = []
    mem: List[Dict[str, Any]] = []
    # Pass 1, timing: compute charged, loop callbacks timed. tracemalloc stays
    # off here because it slows every allocation and would inflate latency.
    t_all = time.perf_counter()
    with BlockMonitor(block_ms) as mon:
        for rep in range(repeat):
            for sc in scs:
                mon.scenario = sc.id
                res = harness.run_scenario(sc, factory, codec, cfg)
                lats = first_action_latencies(res.records)
                rows.append({"scenario": sc.id, "rep": rep, "modality": sc.modality, "wall_s": round(res.wall_s, 4),
                             "virtual_ms": round(res.virtual_ms, 1), "duration_ms": sc.duration_ms,
                             "end_reason": res.end_reason, "agent_error": res.agent_error, "latencies": lats,
                             "records": len(res.records)})
    suite_wall = time.perf_counter() - t_all
    # Pass 2, memory: same scenarios, deterministic clock, tracemalloc on.
    mcfg = harness.HarnessConfig(charge_compute=False)
    gc.collect()
    tracemalloc.start(10)
    first_snap = None
    for rep in range(repeat):
        for sc in scs:
            harness.run_scenario(sc, factory, codec, mcfg)
            gc.collect()
            cur, peak = tracemalloc.get_traced_memory()
            snap = _agent_only(tracemalloc.take_snapshot())
            own = sum(st.size for st in snap.statistics("filename"))
            mem.append({"scenario": sc.id, "rep": rep, "current_kb": round(own / 1024, 1),
                        "all_kb": round(cur / 1024, 1), "peak_kb": round(peak / 1024, 1)})
            if first_snap is None:
                first_snap = snap
            del snap
    last_snap = _agent_only(tracemalloc.take_snapshot())
    tracemalloc.stop()
    growth = [f"{s.traceback[-1].filename.rsplit('/', 2)[-1]}:{s.traceback[-1].lineno} {s.size_diff / 1024:+.1f} KB "
              f"({s.count_diff:+d} blocks)"
              for s in last_snap.compare_to(first_snap, "lineno")[:8]] if first_snap is not None else []

    spoken = [x["spoken_ms"] for r in rows for x in r["latencies"] if x["spoken_ms"] is not None]
    anyact = [x["first_action_ms"] for r in rows for x in r["latencies"] if x["first_action_ms"] is not None]
    silent = [(r["scenario"], x["t_ms"], x["kind"]) for r in rows for x in r["latencies"] if x["spoken_ms"] is None]
    slowest = sorted(((x["spoken_ms"], r["scenario"], x["t_ms"], x["kind"]) for r in rows for x in r["latencies"]
                      if x["spoken_ms"] is not None), reverse=True)[:8]
    hits = sorted(mon.hits, key=lambda h: -h["ms"])
    return {
        "scenarios": len(scs), "repeat": repeat, "charge_compute": charge_compute,
        "suite_wall_s": round(suite_wall, 2),
        "spoken_ms": {"n": len(spoken), "p50": _pct(spoken, 50), "p95": _pct(spoken, 95),
                      "max": max(spoken) if spoken else None,
                      "mean": round(statistics.mean(spoken), 2) if spoken else None},
        "first_action_ms": {"n": len(anyact), "p50": _pct(anyact, 50), "p95": _pct(anyact, 95),
                            "max": max(anyact) if anyact else None},
        "boundaries_without_spoken_action": silent,
        "slowest_spoken": [{"ms": round(a, 2), "scenario": b, "boundary_t_ms": c, "kind": d} for a, b, c, d in slowest],
        "loop": {"callbacks": mon.count, "callback_s": round(mon.total_s, 3), "block_threshold_ms": block_ms,
                 "blocking": len(hits), "worst": hits[:10],
                 "gc": {"collections": len(mon.gc_pauses), "total_ms": round(sum(g["ms"] for g in mon.gc_pauses), 2),
                        "max": max(mon.gc_pauses, key=lambda g: g["ms"], default=None)}},
        "memory": {"note": "current_kb excludes allocations made by this profiler", "first_kb": mem[0]["current_kb"] if mem else None, "last_kb": mem[-1]["current_kb"] if mem else None,
                   "peak_kb": max((m["peak_kb"] for m in mem), default=None), "top_growth": growth,
                   "series_kb": [m["current_kb"] for m in mem]},
        "cap": {"max_wall_s": max((r["wall_s"] for r in rows), default=0.0),
                "max_virtual_ms": max((r["virtual_ms"] for r in rows), default=0.0),
                "end_reasons": _count(r["end_reason"] for r in rows),
                "agent_errors": [(r["scenario"], r["agent_error"]) for r in rows if r["agent_error"]]},
        "rows": rows,
    }


def _count(xs: Any) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


def render(p: Dict[str, Any], label: str = "") -> str:
    s, a, lp, m, c = p["spoken_ms"], p["first_action_ms"], p["loop"], p["memory"], p["cap"]
    lines = [f"# Profile {label}".rstrip(), "",
             f"{p['scenarios']} scenarios x {p['repeat']}, compute charged: {p['charge_compute']}, "
             f"suite wall {p['suite_wall_s']} s", "",
             "| metric | p50 | p95 | max |", "|---|---|---|---|",
             f"| first spoken action (ms) | {s['p50']} | {s['p95']} | {s['max']} |",
             f"| first action of any kind (ms) | {a['p50']} | {a['p95']} | {a['max']} |", "",
             f"Boundaries with no spoken action before the next one: {len(p['boundaries_without_spoken_action'])}", "",
             "## Slowest paths to first spoken action", ""]
    lines += [f"- {x['ms']} ms: {x['scenario']} after {x['kind']} at {x['boundary_t_ms']} ms" for x in p["slowest_spoken"]]
    lines += ["", f"## Event loop: {lp['callbacks']} callbacks, {lp['callback_s']} s total, "
                  f"{lp['blocking']} over {lp['block_threshold_ms']} ms", ""]
    lines += [f"- {h['ms']} ms in {h['scenario']}: `{h['callback']}`" for h in lp["worst"]]
    g = lp.get("gc") or {}
    lines += ["", f"GC: {g.get('collections')} collections, {g.get('total_ms')} ms total, worst {g.get('max')}"]
    lines += ["", f"## Memory: {m['first_kb']} KB after first scenario, {m['last_kb']} KB after last, "
                  f"peak {m['peak_kb']} KB", ""]
    lines += [f"- {g}" for g in m["top_growth"]]
    lines += ["", f"## 120 s cap: max wall {c['max_wall_s']} s, max virtual {c['max_virtual_ms']} ms, "
                  f"end reasons {c['end_reasons']}, agent errors {len(c['agent_errors'])}"]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scenarios", type=Path)
    ap.add_argument("--agent", default="theme5.agent:Agent")
    ap.add_argument("--out", type=Path, default=Path("bench/runs/profile"))
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--block-ms", type=float, default=5.0)
    ap.add_argument("--no-charge", action="store_true", help="do not charge compute to the virtual clock")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--label", default="")
    ap.add_argument("--cprofile", action="store_true", help="also write cProfile stats (cumulative top 30)")
    a = ap.parse_args(argv)
    run = lambda: profile(a.scenarios, a.agent, a.repeat, a.block_ms, not a.no_charge, a.only)  # noqa: E731
    if a.cprofile:
        import cProfile
        import io
        import pstats

        pr = cProfile.Profile()
        p = pr.runcall(run)
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(30)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "cprofile.txt").write_text(buf.getvalue())
    else:
        p = run()
    a.out.mkdir(parents=True, exist_ok=True)
    text = render(p, a.label)
    (a.out / "PROFILE.md").write_text(text, encoding="utf-8")
    (a.out / "profile.json").write_text(json.dumps(p, indent=1, default=str))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
