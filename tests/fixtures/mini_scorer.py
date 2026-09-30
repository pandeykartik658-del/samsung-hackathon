# /mnt/project-files/theme5/tests/fixtures/mini_scorer.py
"""Unit-test fixture only (moved from theme5/). End-to-end runs use sim/ and
bench/scorer.py, the one rubric.

Proxy scorer mirroring the guide's rubric (Task 40, Interruption 35,
Latency 15, Safety 10). The official scorer is unreleased: weights are from
the guide, every threshold and sub-weight here is an ASSUMPTION."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mini_harness import RunResult
from theme5.protocol import (
    ACT_CANCEL, ACT_CLARIFY, ACT_FINAL, ACT_SPEAK, ACT_TOOL_CALL, EV_INTERRUPT, EV_TEXT, EV_AUDIO,
    validate_action,
)

CANCEL_GRACE_MS = 50.0  # ASSUMPTION: "within few ms grace period"
LATENCY_FULL_MS = 300.0  # ASSUMPTION: full marks up to this
LATENCY_ZERO_MS = 2000.0  # ASSUMPTION: zero marks from this


@dataclass
class Score:
    task: float
    interruption: float
    latency: float
    safety: float
    notes: list[str]

    @property
    def total(self) -> float:
        return self.task + self.interruption + self.latency + self.safety


def _norm(v: Any) -> Any:
    return v.strip().lower() if isinstance(v, str) else v


def _match(args: dict[str, Any], want: dict[str, Any]) -> bool:
    return all(_norm(args.get(k)) == _norm(v) for k, v in want.items())


def score(run: RunResult) -> Score:
    exp = run.scenario.get("expect", {})
    acts = run.actions
    notes: list[str] = []
    cancelled = {a["call_id"]: a for a in acts if a["type"] == ACT_CANCEL}
    calls = [a for a in acts if a["type"] == ACT_TOOL_CALL]
    finals = [a for a in acts if a["type"] == ACT_FINAL]

    # ---- task completion (40)
    want_calls = exp.get("calls", [])
    hit = 0
    for w in want_calls:
        if any(c["tool"] == w["tool"] and _match(c["args"], w.get("args", {})) and c["call_id"] not in cancelled for c in calls):
            hit += 1
        else:
            notes.append(f"missing call {w}")
    call_frac = hit / len(want_calls) if want_calls else 1.0
    want_final = exp.get("final")
    if want_final is None:
        final_frac = 1.0
    elif not finals:
        final_frac = 0.0
        notes.append("no final_response")
    else:
        snap = finals[-1]["state_snapshot"]
        checks = [snap.get("intent") == want_final["intent"]] if "intent" in want_final else []
        checks += [_match(snap.get("slots", {}), {k: v}) for k, v in want_final.get("slots", {}).items()]
        final_frac = sum(checks) / len(checks) if checks else 1.0
        if final_frac < 1:
            notes.append(f"final snapshot {snap} != {want_final}")
    task = 40.0 * (0.5 * call_frac + 0.5 * final_frac)

    # ---- interruption recovery (35)
    stale = exp.get("stale", [])
    if not stale:
        interruption = 35.0
    else:
        prompt, no_rerun = 0, 0
        for s in stale:
            after = float(s["after_t"])
            matching = [c for c in calls if c["tool"] == s["tool"] and _match(c["args"], s.get("args", {}))]
            before = [c for c in matching if c["t"] <= after]
            ok = all(c["call_id"] in cancelled and cancelled[c["call_id"]]["t"] - after <= CANCEL_GRACE_MS for c in before
                     if not any(e.get("call_id") == c["call_id"] and e["t"] <= after for e in run.events if e["type"] == "tool_result"))
            prompt += ok
            rerun = [c for c in matching if c["t"] > after]
            no_rerun += not rerun
            if not ok:
                notes.append(f"stale call not cancelled promptly: {s}")
            if rerun:
                notes.append(f"stale rerun: {s}")
        updated = 1.0 if (not want_final or final_frac == 1.0) else final_frac
        interruption = 35.0 * (0.5 * prompt / len(stale) + 0.3 * no_rerun / len(stale) + 0.2 * updated)

    # ---- latency (15): user-facing events only
    lats = [ms for typ, _, ms in run.latency if typ in (EV_TEXT, EV_AUDIO, EV_INTERRUPT)]
    user_turns = [e for e in run.events if (e["type"] == EV_TEXT and e.get("end_of_turn")) or e["type"] in (EV_AUDIO,)]
    responded = [ms for ms in lats if ms is not None]
    if not user_turns:
        latency = 15.0
    else:
        per = [max(0.0, min(1.0, (LATENCY_ZERO_MS - ms) / (LATENCY_ZERO_MS - LATENCY_FULL_MS))) for ms in responded]
        per += [0.0] * max(0, len(user_turns) - len(responded))
        latency = 15.0 * sum(per) / len(per)

    # ---- safety & protocol (10)
    bad = [(a, e) for a in acts for e in [validate_action(a)] if e]
    for a, e in bad[:3]:
        notes.append(f"invalid action {a.get('type')}: {e}")
    commit_keys = [(t, tuple(sorted((k, str(_norm(v))) for k, v in args.items()))) for t, args in run.commits]
    dups = len(commit_keys) - len(set(commit_keys))
    for tool, cap in exp.get("max_commits", {}).items():
        n = sum(1 for t, _ in run.commits if t == tool)
        if n > cap:
            dups += n - cap
            notes.append(f"{tool} committed {n}x (cap {cap})")
    safety = 0.0 if dups else 10.0 * (1 - min(1.0, len(bad) / max(1, len(acts))))
    if run.timed_out:
        notes.append("scenario hit the wall-clock cap")
    return Score(round(task, 2), round(interruption, 2), round(latency, 2), round(safety, 2), notes)


def transcript(run: RunResult) -> list[str]:
    lines = []
    for a in run.actions:
        if a["type"] in (ACT_SPEAK, ACT_CLARIFY, ACT_FINAL):
            lines.append(f"[{a['t']:>7.0f}] {a['type']}: {a['text']}")
        elif a["type"] == ACT_TOOL_CALL:
            lines.append(f"[{a['t']:>7.0f}] call {a['call_id']} {a['tool']}({a['args']})")
        elif a["type"] == ACT_CANCEL:
            lines.append(f"[{a['t']:>7.0f}] cancel {a['call_id']} ({a.get('reason')})")
    return lines
