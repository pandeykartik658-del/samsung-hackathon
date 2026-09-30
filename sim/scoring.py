# sim/scoring.py
"""Proxy scorer: checks a trace against a scenario's `expected` block and
estimates the guide's rubric (section 5). Works from trace records only.

expected block
  required_calls   [{tool, args?, status?: ok|error|timeout|cancelled|any (default ok),
                     min_count?: 1, max_count?}]
  ordered_calls    bool; first matches of required_calls must be issued in list order
  forbidden_calls  [{tool, args?, after_event?, before_event?}]  no matching call issued in window
  must_cancel      [{tool, args?, anchor_event, within_ms?}] matching calls still in flight at the
                   anchor must be cancelled within `within_ms` (default scenario cancel_grace_ms)
  clarify          {required: true, after_event?, before_event?}  a clarify action in the window
  max_writes       {tool: n}  non-failed calls of that tool (ok, or never finished)
  final_snapshot   {intent, slots: {...}}  subset match on the last final action
  final_mentions   [matcher]  each must appear in the last final's text
  final_must_not_mention [str]  stale values that must not appear in it
Key names match bench/scorer.py so both scorers read the same scenario files.

Rubric weights come from the guide; every threshold below is an ASSUMPTION:
  latency: <= 300 ms -> full, >= 2000 ms -> zero, linear between.
  quality multiplier (0.8-1.2) is not modelled (reported as 1.0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import wire
from .mock_tools import args_match, canonical_args, value_matches

WEIGHTS = {"task": 40.0, "interruption": 35.0, "latency": 15.0, "safety": 10.0}
LAT_FULL_MS = 300.0
LAT_ZERO_MS = 2000.0
MULTIMODAL_WEIGHT = 1.5  # guide 5: hidden scores weight multimodal scenarios 1.5x


@dataclass
class Check:
    name: str
    category: str
    passed: bool
    detail: str = ""
    hard: bool = True


@dataclass
class ScoreCard:
    scenario_id: str
    modality: str
    checks: List[Check] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    total: float = 0.0
    latencies_ms: List[Optional[float]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.hard)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id, "modality": self.modality, "passed": self.passed,
            "total": round(self.total, 2), "scores": {k: round(v, 2) for k, v in self.scores.items()},
            "latencies_ms": self.latencies_ms,
            "checks": [c.__dict__ for c in self.checks],
        }


# -- trace reconstruction ---------------------------------------------------------

def reconstruct(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    calls: Dict[str, Dict[str, Any]] = {}
    actions, events, sys = [], [], []
    for r in records:
        d = r["data"]
        if r["dir"] == "out":
            actions.append(r)
            if r["kind"] == "cancel" and d.get("call_id") in calls:
                c = calls[d["call_id"]]
                c.setdefault("cancel_ms", r["t_ms"])
        elif r["dir"] == "in":
            events.append(r)
        else:
            sys.append(r)
            if r["kind"] == "tool_started":
                calls[d["call_id"]] = {"call_id": d["call_id"], "tool": d["tool"], "args": d["args"],
                                       "issued_ms": r["t_ms"], "side_effect": d.get("side_effect", "read"),
                                       "status": "pending", "end_ms": None}
            elif r["kind"] == "tool_completed" and d["call_id"] in calls:
                calls[d["call_id"]].update(status=d["status"], end_ms=r["t_ms"])
            elif r["kind"] == "tool_cancelled" and d["call_id"] in calls:
                calls[d["call_id"]].update(status="cancelled", end_ms=r["t_ms"])
    return {"calls": sorted(calls.values(), key=lambda c: c["issued_ms"]),
            "actions": actions, "events": events, "sys": sys}


def text_mentions(m: Any, text: str) -> bool:
    import re
    if isinstance(m, dict):
        if "any_of" in m:
            return any(text_mentions(x, text) for x in m["any_of"])
        if "regex" in m:
            return re.search(m["regex"], text, re.I) is not None
        return False
    return str(m).lower() in text.lower()


def _event_times(events: List[Dict[str, Any]]) -> Dict[str, float]:
    return {r["data"].get("event_id"): r["t_ms"] for r in events if r["data"].get("event_id")}


def _in_window(t: float, lo: Optional[float], hi: Optional[float]) -> bool:
    return (lo is None or t >= lo) and (hi is None or t < hi)


def _lat_score(ms: Optional[float]) -> float:
    if ms is None:
        return 0.0
    if ms <= LAT_FULL_MS:
        return 1.0
    if ms >= LAT_ZERO_MS:
        return 0.0
    return 1.0 - (ms - LAT_FULL_MS) / (LAT_ZERO_MS - LAT_FULL_MS)


def _wavg(parts: List[tuple]) -> float:
    tot = sum(w for _, w in parts)
    return sum(v * w for v, w in parts) / tot if tot else 1.0


# -- scoring -------------------------------------------------------------------------

def score(records: List[Dict[str, Any]], expected: Dict[str, Any], scenario_id: str = "",
          modality: str = "text", cancel_grace_ms: float = 300.0) -> ScoreCard:
    rc = reconstruct(records)
    calls, actions, events = rc["calls"], rc["actions"], rc["events"]
    et = _event_times(events)
    card = ScoreCard(scenario_id, modality)
    add = card.checks.append

    # Task completion ----------------------------------------------------------------
    req_hits = []
    first_idx = []
    for i, spec in enumerate(expected.get("required_calls", [])):
        want = spec.get("status", "ok")
        m = [c for c in calls if c["tool"] == spec["tool"] and args_match(spec.get("args"), c["args"])
             and (want == "any" or c["status"] == want)]
        lo, hi = spec.get("min_count", 1), spec.get("max_count")
        ok = len(m) >= lo and (hi is None or len(m) <= hi)
        req_hits.append(ok)
        first_idx.append(m[0]["issued_ms"] if m else None)
        add(Check(f"required[{i}] {spec['tool']}", "task", ok,
                  f"{len(m)} matching call(s) with status {want}; need {lo}" + (f"..{hi}" if hi is not None else "+")))
    if expected.get("ordered_calls") and first_idx:
        seq = [t for t in first_idx if t is not None]
        ok = len(seq) == len(first_idx) and seq == sorted(seq)
        add(Check("required calls in order", "task", ok, f"first-match times {first_idx}"))
        req_hits.append(ok)

    finals = [a for a in actions if a["kind"] == "final"]
    last_final = finals[-1]["data"] if finals else None
    add(Check("final response emitted", "task", last_final is not None,
              f"{len(finals)} final action(s)"))
    snap_parts: List[bool] = []
    exp_snap = expected.get("final_snapshot")
    if exp_snap is not None:
        snap = (last_final or {}).get("snapshot") or {}
        if "intent" in exp_snap:
            snap_parts.append(value_matches(exp_snap["intent"], snap.get("intent")))
        slots = snap.get("slots") or {}
        for k, v in (exp_snap.get("slots") or {}).items():
            snap_parts.append(args_match({k: v}, slots))
        add(Check("final snapshot matches", "task", all(snap_parts),
                  f"{sum(snap_parts)}/{len(snap_parts)} fields match; got {snap}"))
    text_ok = True
    mentions = expected.get("final_mentions") or []
    banned = expected.get("final_must_not_mention") or []
    if mentions or banned:
        txt = (last_final or {}).get("text") or ""
        missing = [m for m in mentions if not text_mentions(m, txt)]
        stale = [b for b in banned if text_mentions(b, txt)]
        text_ok = last_final is not None and not missing and not stale
        add(Check("final text grounded", "task", text_ok,
                  f"missing {missing}, stale {stale}; final text: {txt[:120]!r}"))

    clar_ok = None
    cl = expected.get("clarify")
    if cl and cl.get("required", True):
        lo = et.get(cl.get("after_event")) if cl.get("after_event") else None
        hi = et.get(cl.get("before_event")) if cl.get("before_event") else None
        clar = [a for a in actions if a["kind"] == "clarify" and _in_window(a["t_ms"], lo, hi)]
        clar_ok = bool(clar)
        add(Check("clarification asked", "task", clar_ok, f"{len(clar)} clarify action(s) in window"))

    task_parts = [
        (sum(req_hits) / len(req_hits) if req_hits else 1.0, 0.45),
        (sum(snap_parts) / len(snap_parts) if snap_parts else 1.0, 0.35),
        (1.0 if (last_final is not None and text_ok) else 0.0, 0.10),
    ]
    if clar_ok is not None:
        task_parts.append((1.0 if clar_ok else 0.0, 0.10))
    card.scores["task"] = WEIGHTS["task"] * _wavg(task_parts)

    # Interruption recovery --------------------------------------------------------------
    intr: List[bool] = []
    for i, spec in enumerate(expected.get("forbidden_calls", [])):
        lo = et.get(spec["after_event"]) if spec.get("after_event") else None
        hi = et.get(spec["before_event"]) if spec.get("before_event") else None
        bad = [c for c in calls if c["tool"] == spec["tool"] and args_match(spec.get("args"), c["args"])
               and _in_window(c["issued_ms"], lo, hi)]
        ok = not bad
        intr.append(ok)
        add(Check(f"forbidden[{i}] {spec['tool']}", "interruption", ok,
                  "none issued" if ok else f"issued at {[c['issued_ms'] for c in bad]} ms"))
    for i, spec in enumerate(expected.get("must_cancel", [])):
        anchor = et.get(spec["anchor_event"])
        within = spec.get("within_ms", cancel_grace_ms)
        live = [c for c in calls if c["tool"] == spec["tool"] and args_match(spec.get("args"), c["args"])
                and anchor is not None and c["issued_ms"] <= anchor + within
                and (c["end_ms"] is None or c["end_ms"] > anchor)]
        late = [c for c in live if c.get("cancel_ms") is None or c["cancel_ms"] > anchor + within]
        ok = not late
        intr.append(ok)
        detail = ("no matching call in flight at anchor" if not live else
                  "cancelled in time" if ok else
                  f"not cancelled within {within} ms: " +
                  ", ".join(f"{c['call_id']} cancel_ms={c.get('cancel_ms')}" for c in late))
        add(Check(f"must_cancel[{i}] {spec['tool']}", "interruption", ok, detail))
    card.scores["interruption"] = WEIGHTS["interruption"] * (sum(intr) / len(intr) if intr else 1.0)

    # Latency -----------------------------------------------------------------------------
    subst = [a for a in actions if wire.is_substantive(a["data"])]
    lat_scores = []
    for ev in events:
        if not wire.is_user_turn_boundary(ev["data"]):
            continue
        nxt = next((a for a in subst if a["t_ms"] >= ev["t_ms"]), None)
        ms = None if nxt is None else round(nxt["t_ms"] - ev["t_ms"], 3)
        card.latencies_ms.append(ms)
        lat_scores.append(_lat_score(ms))
    card.scores["latency"] = WEIGHTS["latency"] * (sum(lat_scores) / len(lat_scores) if lat_scores else 1.0)
    add(Check("latency", "latency", True, f"first substantive response per user turn (ms): {card.latencies_ms}",
              hard=False))

    # Safety & protocol -----------------------------------------------------------------------
    live_writes = [c for c in calls if c["side_effect"] == "write" and c["status"] in ("ok", "pending")]
    groups: Dict[str, int] = {}
    for c in live_writes:
        k = c["tool"] + "|" + canonical_args(c["args"])
        groups[k] = groups.get(k, 0) + 1
    dups = {k: n for k, n in groups.items() if n > 1}
    add(Check("no duplicate state-changing calls", "safety", not dups, f"duplicates: {dups}" if dups else "none"))
    for tool, n in (expected.get("max_writes") or expected.get("max_write_calls") or {}).items():
        cnt = sum(1 for c in live_writes if c["tool"] == tool)
        add(Check(f"max_writes {tool}<={n}", "safety", cnt <= n, f"{cnt} non-failed call(s)"))
        if cnt > n:
            dups[f"max:{tool}"] = cnt
    invalid = [a for a in actions if a["kind"] == "invalid"]
    add(Check("all actions schema-valid", "safety", not invalid,
              f"{len(invalid)} invalid: {[a['data'].get('errors') for a in invalid][:3]}"))
    proto_errs = [s for s in rc["sys"] if s["kind"] in ("duplicate_call_id",)
                  or (s["kind"] == "cancel_ignored" and s["data"].get("reason") == "unknown")
                  or (s["kind"] == "tool_started" and s["data"].get("rejected"))]
    add(Check("no protocol errors (ids, unknown tools, bad args)", "safety", not proto_errs,
              f"{len(proto_errs)} error(s): {[p['kind'] for p in proto_errs][:5]}"))
    card.scores["safety"] = WEIGHTS["safety"] * (0.5 * (not dups) + 0.25 * (not invalid) + 0.25 * (not proto_errs))

    card.total = sum(card.scores.values())
    return card


def aggregate(cards: List[ScoreCard]) -> Dict[str, Any]:
    if not cards:
        return {"count": 0}
    w = [MULTIMODAL_WEIGHT if c.modality in ("audio", "visual") else 1.0 for c in cards]
    return {
        "count": len(cards),
        "passed": sum(c.passed for c in cards),
        "mean_total": round(sum(c.total for c in cards) / len(cards), 2),
        "weighted_total": round(sum(c.total * x for c, x in zip(cards, w)) / sum(w), 2),
        "by_category": {k: round(sum(c.scores.get(k, 0) for c in cards) / len(cards), 2) for k in WEIGHTS},
        "by_modality": {m: round(sum(c.total for c in cards if c.modality == m) /
                                 max(1, sum(1 for c in cards if c.modality == m)), 2)
                        for m in sorted({c.modality for c in cards})},
    }
