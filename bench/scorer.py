# bench/scorer.py
"""Offline replica of the Theme 05 rubric (guide section 5), scored strictly
from a trace.

    TC  Task Completion        40
    IR  Interruption Recovery  35
    LAT Response Latency       15
    SP  Safety & Protocol      10
    QM  quality multiplier     0.80 - 1.20
    scenario score = min(100, (TC + IR + LAT + SP) * QM)
    suite score    = weighted mean, audio/visual scenarios weight 1.5

The guide names the criteria but not the formulas. Every formula, threshold and
weight below that is not one of the five numbers above is an ASSUMPTION and is
listed in ASSUMPTIONS (printed by `python -m bench.scorer --assumptions`).

Input 1: trace records (sim/trace.py), one dict per JSONL line:
    {"seq", "t_ms", "dir": "in"|"out"|"sys", "kind", "data"}
Input 2: the scenario dict (sim/scenario.py) - only `id`, `modality`,
    `cancel_grace_ms` and `expected` are read. Expected keys used here:
    required_calls   [{tool, args?: matcher-dict, status?: ok|error|timeout|any,
                       min_count?, max_count?}]
    ordered_calls    bool
    forbidden_calls  [{tool, args?, after_event?, before_event?}]
    final_snapshot   {intent, slots: {name: matcher}} | null
    must_cancel      [{anchor_event, tool?, args?, within_ms?}]
    clarify          {after_event?, before_event?, required?: bool}
    max_write_calls  {tool: n}   (alias max_writes)
    final_text       {includes_any: [...], excludes: [...]}
    final_mentions   [matcher]   all must appear (extension)
    must_report_failure bool     honest failure report (extension)
These are the keys of sim/scoring.py (the harness's pass/fail checker); this
module turns the same expectations into rubric points.
Matchers: scalar (case/space-insensitive), {"any_of": [...]}, {"regex": ...},
{"present": bool}.

Field-name tolerance: actions may be `final` or `final_response`, snapshots
`snapshot` or `state_snapshot`, calls `tool`/`name` + `args`/`arguments`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

WEIGHTS = {"TC": 40.0, "IR": 35.0, "LAT": 15.0, "SP": 10.0}
QM_MIN, QM_MAX = 0.80, 1.20
MULTIMODAL_WEIGHT = 1.5

ASSUMPTIONS: Dict[str, str] = {
    "A01_total": "Scenario score = min(100, (TC+IR+LAT+SP) * QM). The guide gives 0-100 per scenario and a 0.8-1.2 multiplier but not how they combine or whether 100 is a cap.",
    "A02_suite": "Suite score = weighted mean of scenario scores, weight 1.5 for audio and visual, 1.0 for text (guide says multimodal hidden scenarios carry 1.5x).",
    "A03_invalid": "Actions the harness logged as kind='invalid' or that fail our schema check are ignored for TC/IR/LAT (the real kit most likely drops them) and cost SP.",
    "A10_tc_split": "TC 40 = tool execution 40% + argument extraction 20% + snapshot accuracy 25% + grounding 15% (guide lists these four criteria without weights).",
    "A11_tc_exec": "Tool execution = fraction of expected.required_calls that have a matching tool_call whose result has the expected status (default ok and not cancelled; 'any' = issued) at least min_count times and no more than max_count issues; x0.75 if ordered_calls and the first matches are out of order; each forbidden call fired subtracts 0.25 (floor 0). Clarification-only scenarios (no required calls, clarify.required) score execution by whether a clarify action was emitted.",
    "A12_tc_args": "Argument extraction = mean over required calls of the best fraction of expected args matched by any call to that tool (even failed or cancelled ones).",
    "A13_tc_snapshot": "Snapshot accuracy compares the LAST final action's snapshot: intent counts as one item, each expected slot as one item. Extra slots are not penalised. No final action -> 0.",
    "A14_tc_grounding": "Grounding = 1.0, minus 0.5 per false completion claim, minus 0.5 if the final mentions a value that appears only in stale (cancelled) results, x0 if final_text.includes_any is given and none appears, times the fraction of expected.final_mentions present; final_text.excludes counts as a stale value. If there are fresh ok goal results but the final mentions none of their id-like values or expected mentions, x0.7. No final -> 0.",
    "A20_ir_points": "Interruption points = explicit interrupt events, must_cancel anchors, and every user end-of-turn that arrives after the agent's first tool_call (corrections after results count). No points -> IR full marks (N/A is not penalised).",
    "A21_ir_split": "IR 35 = cancel promptness 35% + no stale re-runs 25% + snapshot updated 25% + re-plan 15%.",
    "A22_ir_cancel": "Superseded calls = calls issued strictly before and still in flight at an interruption point that match expected.must_cancel, or (when must_cancel is absent) whose args contradict the expected final snapshot. Each is scored 1.0 if cancelled within cancel_grace_ms (scenario, default 300 ms), linear to 0 at 2000 ms, 0 if never cancelled. Cancelling a call that was still valid costs nothing here (it may cost TC).",
    "A23_ir_stale": "Stale re-run = a tool_call after the LAST interruption point whose args contradict the expected final snapshot (calls answering an intermediate correction were valid when issued), or one after the first point that re-issues a cancelled call's exact args. Each costs 0.5 of the component. Using a stale result in the final also costs 0.5.",
    "A24_ir_snapshot": "Snapshot updated = fraction of expected final slots correct in the first snapshot the agent emits after the last interruption point (any action carrying a snapshot), averaged with the final snapshot score.",
    "A25_ir_replan": "Re-plan = fraction of interruption points followed (before the next user event more than 1500 ms later, or scenario end) by a substantive speech, clarify, final, or new tool_call.",
    "A30_lat_points": "Latency points = user turn boundaries: end-of-turn text/audio, interrupt, and video_frame. A boundary followed by another boundary within 1500 ms with no substantive action between is merged into the later one.",
    "A31_lat_substantive": "Substantive = speak with kind != 'filler', clarify, or final. Tool calls and cancels are not spoken actions and do not stop the clock.",
    "A32_lat_curve": "Per-boundary score: <=300 ms -> 1.0; 300-1000 ms linear to 0.5; 1000-3000 ms linear to 0; none before the next boundary/end -> 0. LAT = 15 * mean.",
    "A40_sp_split": "SP 10 = schema validity 40% + call-id hygiene 20% + no duplicate writes 40%.",
    "A41_sp_valid": "Validity = fraction of out records that are valid actions; x0.5 if there is no final action; x0.5 per agent_error record (floor 0); 0 if wall_cap_exceeded.",
    "A42_sp_ids": "Call-id hygiene = 1 minus 0.25 per reused call_id, cancel of an unknown id, or cancel of an already-completed call.",
    "A43_sp_dupes": "Duplicate write = a state-modifying tool_call with the same tool + canonical args as an earlier one that is still pending or succeeded, or that is retried after a timeout (outcome unknown) unless the manifest marks the tool idempotent. Also a tool exceeding expected.max_write_calls, counted as max(committed writes, non-failed calls). Any duplicate -> component 0.",
    "A44_sp_class": "A tool is state-modifying unless the manifest explicitly marks it read-only (side_effect 'read', read_only true, readOnlyHint true, kind 'read'). Same rule as SPEC U11.",
    "A50_qm": "QM starts at 1.00. -0.10 per false completion claim (cap -0.20); -0.05 per filler beyond max(2, user turns) (cap -0.10); -0.03 per exact repeated utterance (cap -0.09); +0.05 grounded final; +0.05 final under 60 words; +0.05 clarified when clarification was expected, -0.05 when it was expected and missing; +0.05 honest failure report when must_report_failure. Clamped to 0.80-1.20. The real judge is probably an LLM or a human; this is a proxy.",
}

SUCCESS_RE = re.compile(r"\b(booked|reserved|confirmed|created|raised|filed|all set)\b", re.I)
NEGATION_RE = re.compile(r"\b(not|n't|couldn't|could not|unable|failed|fail|unsuccessful|no longer|haven't|hasn't|wasn't|won't|before i|once|will|shall|going to|about to|let me|if you|should i|do you want|want me to|would you like)\b", re.I)
FAILURE_RE = re.compile(r"\b(fail|failed|couldn't|could not|unable|error|problem|didn't go through|not (?:been )?(?:booked|created|completed)|unavailable|timed out|try again)\b", re.I)
ID_KEY_RE = re.compile(r"(_id|_ref|ref|id|code|number)$", re.I)

EOT_KINDS = ("text_chunk", "audio_clip")
USER_KINDS = ("text_chunk", "audio_clip", "video_frame", "interrupt")
ACTION_ALIASES = {"final_response": "final", "say": "speak", "clarification": "clarify", "cancellation": "cancel"}
READ_MARKERS = ("read", "read_only", "readonly", "query", "none")


# --------------------------------------------------------------------------
# matching helpers
# --------------------------------------------------------------------------
def _norm(v: Any) -> Any:
    if isinstance(v, str):
        return " ".join(v.strip().lower().split())
    if isinstance(v, bool):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def value_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        if "any_of" in expected:
            return any(value_matches(e, actual) for e in expected["any_of"])
        if "regex" in expected:
            return actual is not None and re.search(expected["regex"], str(actual), re.I) is not None
        if "present" in expected:
            return (actual is not None) == bool(expected["present"])
        return isinstance(actual, dict) and all(value_matches(v, actual.get(k)) for k, v in expected.items())
    if isinstance(expected, str) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        return _norm(expected) == _norm(str(actual))
    if isinstance(actual, str) and isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return _norm(actual) == _norm(str(expected))
    return _norm(expected) == _norm(actual)


def args_fraction(expected: Optional[Dict[str, Any]], actual: Dict[str, Any]) -> float:
    exp = expected or {}
    if not exp:
        return 1.0
    hit = sum(1 for k, v in exp.items() if value_matches(v, actual.get(k)))
    return hit / len(exp)


def args_contradict(expected_slots: Dict[str, Any], args: Dict[str, Any]) -> bool:
    """True when a call argument names an expected slot but has another value."""
    for k, v in args.items():
        if k in expected_slots and not value_matches(expected_slots[k], v):
            return True
    return False


def canonical(tool: str, args: Dict[str, Any]) -> str:
    def norm(v):
        if isinstance(v, dict):
            return {k: norm(x) for k, x in sorted(v.items())}
        if isinstance(v, list):
            return [norm(x) for x in v]
        return _norm(v)
    return tool + "|" + json.dumps(norm(args or {}), sort_keys=True, default=str)


def _id_values(obj: Any, out: Optional[set] = None) -> set:
    """Id-like string values in a tool result (flight_id, booking_ref, ticket_id...)."""
    out = set() if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (str, int)) and not isinstance(v, bool) and ID_KEY_RE.search(str(k)):
                s = str(v).strip()
                if len(s) >= 3:
                    out.add(s)
            else:
                _id_values(v, out)
    elif isinstance(obj, list):
        for x in obj:
            _id_values(x, out)
    return out


def _mentions(text: str, value: str) -> bool:
    return _norm(value) in _norm(text)


def _count_words(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


# --------------------------------------------------------------------------
# trace model
# --------------------------------------------------------------------------
@dataclass
class Call:
    call_id: str
    tool: str
    args: Dict[str, Any]
    t_ms: float
    seq: int
    write: bool
    status: Optional[str] = None  # ok | error | timeout | cancelled (from result)
    result: Any = None
    t_result: Optional[float] = None
    cancelled_at: Optional[float] = None

    @property
    def done_ok(self) -> bool:
        return self.status == "ok" and self.cancelled_at is None

    def in_flight_at(self, t: float) -> bool:
        # a call issued on the interruption's own tick is a response to it, not stale work
        if self.t_ms >= t:
            return False
        if self.t_result is not None and self.t_result <= t:
            return False
        if self.cancelled_at is not None and self.cancelled_at <= t:
            return False
        return True


@dataclass
class TraceView:
    records: List[Dict[str, Any]]
    user: List[Dict[str, Any]] = field(default_factory=list)       # in, USER_KINDS
    actions: List[Dict[str, Any]] = field(default_factory=list)    # valid out records, normalised
    invalid: List[Dict[str, Any]] = field(default_factory=list)
    calls: Dict[str, Call] = field(default_factory=dict)
    call_order: List[Call] = field(default_factory=list)
    reused_ids: int = 0
    bad_cancels: int = 0
    manifest: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    commits: List[Dict[str, Any]] = field(default_factory=list)
    agent_errors: int = 0
    wall_cap: bool = False
    end_ms: float = 0.0
    event_times: Dict[str, float] = field(default_factory=dict)

    def finals(self) -> List[Dict[str, Any]]:
        return [a for a in self.actions if a["type"] == "final"]


def _get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def is_write_tool(tool_def: Optional[Dict[str, Any]]) -> bool:
    """A44: state-modifying unless explicitly read-only."""
    if not tool_def:
        return True
    for k in ("read_only", "readonly", "readOnly", "is_read_only"):
        if k in tool_def:
            return not bool(tool_def[k])
    ann = tool_def.get("annotations")
    if isinstance(ann, dict) and "readOnlyHint" in ann:
        return not bool(ann["readOnlyHint"])
    for k in ("side_effect", "side_effects", "effect", "kind", "category"):
        v = tool_def.get(k)
        if isinstance(v, str):
            return v.strip().lower() not in READ_MARKERS
        if isinstance(v, bool) and k.startswith("side_effect"):
            return v
    for k in ("state_modifying", "mutating"):
        if k in tool_def:
            return bool(tool_def[k])
    return True


def is_idempotent_tool(tool_def: Optional[Dict[str, Any]]) -> bool:
    if not tool_def:
        return False
    if tool_def.get("idempotent") is True:
        return True
    ann = tool_def.get("annotations")
    if isinstance(ann, dict) and ann.get("idempotentHint") is True:
        return True
    props = ((tool_def.get("parameters") or {}).get("properties") or {})
    return any("idempotency" in k.lower() for k in props)


def normalise_action(data: Any) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Return (normalised action, schema errors)."""
    if not isinstance(data, dict):
        return None, ["action is not an object"]
    t = str(data.get("type", "")).lower()
    t = ACTION_ALIASES.get(t, t)
    a: Dict[str, Any] = {"type": t, "raw": data}
    errs: List[str] = []
    if t == "speak":
        a["text"] = _get(data, "text", default="")
        a["kind"] = str(_get(data, "kind", default="answer")).lower()
        if not isinstance(a["text"], str) or not a["text"].strip():
            errs.append("speak.text empty")
    elif t == "tool_call":
        a["call_id"] = _get(data, "call_id", "id")
        a["tool"] = _get(data, "tool", "name")
        args = _get(data, "args", "arguments", default={})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                errs.append("args not JSON")
                args = {}
        a["args"] = args
        if not isinstance(a["call_id"], str) or not a["call_id"]:
            errs.append("call_id missing")
        if not isinstance(a["tool"], str) or not a["tool"]:
            errs.append("tool missing")
        if not isinstance(args, dict):
            errs.append("args not an object")
    elif t == "cancel":
        a["call_id"] = _get(data, "call_id", "id")
        if not isinstance(a["call_id"], str) or not a["call_id"]:
            errs.append("call_id missing")
    elif t == "clarify":
        a["text"] = _get(data, "text", "question", default="")
        if not isinstance(a["text"], str) or not a["text"].strip():
            errs.append("clarify.text empty")
    elif t == "final":
        a["text"] = _get(data, "text", default="")
        snap = _get(data, "snapshot", "state_snapshot", "state")
        a["snapshot"] = snap
        if not isinstance(a["text"], str):
            errs.append("final.text not a string")
        if not isinstance(snap, dict) or "intent" not in snap or not isinstance(snap.get("slots"), dict):
            errs.append("final snapshot malformed")
    else:
        errs.append(f"unknown action type {t!r}")
    if t != "final":
        snap = _get(data, "snapshot", "state_snapshot")
        if isinstance(snap, dict) and isinstance(snap.get("slots"), dict):
            a["snapshot"] = snap
    return (a if not errs else None), errs


def build_view(records: Iterable[Dict[str, Any]]) -> TraceView:
    recs = sorted((r for r in records if isinstance(r, dict)),
                  key=lambda r: (float(r.get("t_ms", 0.0)), int(r.get("seq", 0))))
    v = TraceView(records=recs)
    for r in recs:
        d = r.get("data") or {}
        t = float(r.get("t_ms", 0.0))
        v.end_ms = max(v.end_ms, t)
        kind = r.get("kind")
        direction = r.get("dir")
        if direction == "in":
            eid = _get(d, "event_id", "id") if isinstance(d, dict) else None
            if eid is not None and kind != "tool_result":
                v.event_times.setdefault(str(eid), t)
            if kind == "tool_manifest":
                for td in _get(d, "tools", "manifest", default=[]) or []:
                    if isinstance(td, dict) and td.get("name"):
                        v.manifest[td["name"]] = td
            elif kind in USER_KINDS:
                v.user.append({**r, "t_ms": t})
            elif kind == "tool_result":
                c = v.calls.get(str(d.get("call_id")))
                if c is not None and c.t_result is None:
                    c.status = str(d.get("status", "ok")).lower()
                    c.result = _get(d, "result", "output", default=_get(d, "error"))
                    c.t_result = t
        elif direction == "out":
            a, errs = normalise_action(d) if kind != "invalid" else (None, ["harness marked invalid"])
            if a is None:
                v.invalid.append({**r, "errors": errs})
                continue
            a.update({"t_ms": t, "seq": r.get("seq", 0)})
            v.actions.append(a)
            if a["type"] == "tool_call":
                cid = a["call_id"]
                if cid in v.calls:
                    v.reused_ids += 1
                    continue
                c = Call(cid, a["tool"], a["args"], t, a["seq"], is_write_tool(v.manifest.get(a["tool"])))
                v.calls[cid] = c
                v.call_order.append(c)
            elif a["type"] == "cancel":
                c = v.calls.get(a["call_id"])
                if c is None or (c.t_result is not None and c.t_result <= t):
                    v.bad_cancels += 1
                elif c.cancelled_at is None:
                    c.cancelled_at = t
        elif direction == "sys":
            if kind == "write_committed":
                v.commits.append({**d, "t_ms": t})
            elif kind == "duplicate_call_id":
                v.reused_ids += 1
            elif kind == "agent_error":
                v.agent_errors += 1
            elif kind == "wall_cap_exceeded":
                v.wall_cap = True
            elif kind == "tool_cancelled":
                c = v.calls.get(str(d.get("call_id")))
                if c is not None and c.cancelled_at is None:
                    c.cancelled_at = t
    return v


def _event_time(v: TraceView, event_id: Optional[str], scenario: Dict[str, Any]) -> Optional[float]:
    if not event_id:
        return None
    if event_id in v.event_times:
        return v.event_times[event_id]
    for e in scenario.get("events", []):
        if e.get("id") == event_id:
            return float(e.get("t_ms", 0.0))
    return None


def _is_eot(r: Dict[str, Any]) -> bool:
    d = r.get("data") or {}
    if r["kind"] in EOT_KINDS:
        return bool(_get(d, "end_of_turn", "eot", "is_final", default=False))
    return r["kind"] in ("interrupt", "video_frame")


def _substantive(a: Dict[str, Any]) -> bool:
    return a["type"] in ("clarify", "final") or (a["type"] == "speak" and a.get("kind") != "filler")


# --------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------
@dataclass
class ScenarioScore:
    scenario_id: str
    modality: str
    TC: float
    IR: float
    LAT: float
    SP: float
    QM: float
    total: float
    weight: float
    components: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    evidence_seq: List[int] = field(default_factory=list)  # trace seqs worth showing
    latencies_ms: List[Optional[float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# --------------------------------------------------------------------------
# sub-scores
# --------------------------------------------------------------------------
def _false_claims(v: TraceView) -> List[Dict[str, Any]]:
    """Speech that claims a write happened before any write succeeded."""
    ok_write_times = sorted(c.t_result for c in v.call_order if c.write and c.done_ok and c.t_result is not None)
    ok_write_times += [c["t_ms"] for c in v.commits]
    first_ok = min(ok_write_times) if ok_write_times else None
    out = []
    for a in v.actions:
        if a["type"] not in ("speak", "final"):
            continue
        text = a.get("text") or ""
        if not SUCCESS_RE.search(text) or NEGATION_RE.search(text):
            continue
        if first_ok is None or a["t_ms"] < first_ok:
            out.append(a)
    return out


def score_tc(v: TraceView, exp: Dict[str, Any], sc: Dict[str, Any], s: ScenarioScore) -> Tuple[float, Dict[str, Any]]:
    req = exp.get("required_calls") or []
    clar = exp.get("clarify")
    clar_required = isinstance(clar, dict) and bool(clar.get("required", True))
    c_after = _event_time(v, clar.get("after_event"), sc) if isinstance(clar, dict) else None
    c_before = _event_time(v, clar.get("before_event"), sc) if isinstance(clar, dict) else None
    clarified = any(a["type"] == "clarify" and (c_after is None or a["t_ms"] >= c_after)
                    and (c_before is None or a["t_ms"] < c_before) for a in v.actions)

    # execution
    if req:
        hits = 0
        firsts: List[int] = []
        for rc in req:
            want = rc.get("status", "ok")
            match = [c for c in v.call_order if c.tool == rc.get("tool") and args_fraction(rc.get("args"), c.args) == 1.0]
            good = [c for c in match if want == "any" or (c.done_ok if want == "ok" else c.status == want)]
            ok = len(good) >= int(rc.get("min_count", 1))
            if ok and rc.get("max_count") is not None and len(match) > int(rc["max_count"]):
                ok = False
                s.notes.append(f"TC: {rc.get('tool')} issued {len(match)}x (max {rc['max_count']})")
            hits += ok
            if match:
                firsts.append(match[0].seq)
            if not ok and not match:
                s.notes.append(f"TC: required call {rc.get('tool')} {json.dumps(rc.get('args') or {})} never issued")
            elif not ok:
                s.notes.append(f"TC: required call {rc.get('tool')} issued but did not complete with status {want}")
        execution = hits / len(req)
        if exp.get("ordered_calls") and firsts != sorted(firsts):
            execution *= 0.75
            s.notes.append("TC: required calls issued out of order")
    else:
        execution = 1.0
    if clar_required:
        if not clarified:
            s.notes.append("TC: expected a clarification in its window, none emitted")
        execution = (execution + (1.0 if clarified else 0.0)) / (2 if req else 1)
    forb = _forbidden_hits(v, exp, sc)
    if forb:
        s.notes.append(f"TC: {len(forb)} forbidden call(s): " + ", ".join(f"{c.tool}{json.dumps(c.args)}" for c in forb))
        s.evidence_seq += [c.seq for c in forb]
    execution = max(0.0, execution - 0.25 * len(forb))

    # arguments
    if req:
        fr = []
        for rc in req:
            cand = [args_fraction(rc.get("args"), c.args) for c in v.call_order if c.tool == rc.get("tool")]
            fr.append(max(cand) if cand else 0.0)
        arg_score = sum(fr) / len(fr)
    else:
        arg_score = 1.0

    # snapshot
    finals = v.finals()
    exp_snap = exp.get("final_snapshot")
    snap_score, snap_detail = 0.0, {}
    if finals:
        snap_score, snap_detail = snapshot_accuracy(exp_snap, finals[-1].get("snapshot") or {})
        for k, okk in snap_detail.items():
            if not okk:
                s.notes.append(f"TC: final snapshot {k} wrong (got {json.dumps((finals[-1].get('snapshot') or {}).get('slots', {}).get(k) if k != 'intent' else (finals[-1].get('snapshot') or {}).get('intent'))})")
        s.evidence_seq.append(finals[-1]["seq"])
    else:
        s.notes.append("TC: no final action")

    # grounding
    claims = _false_claims(v)
    stale_used = _stale_grounding(v, exp)
    if finals:
        text = finals[-1].get("text") or ""
        g = 1.0 - 0.5 * len(claims) - (0.5 if stale_used else 0.0)
        mentions = exp.get("final_mentions") or []
        ft = exp.get("final_text") or {}
        if ft.get("includes_any"):
            if not any(_text_matches(m, text) for m in ft["includes_any"]):
                g *= 0.0
                s.notes.append(f"TC: final mentions none of {ft['includes_any']}")
        elif mentions:
            present = sum(1 for m in mentions if _text_matches(m, text))
            g *= present / len(mentions)
            if present < len(mentions):
                s.notes.append(f"TC: final mentions {present}/{len(mentions)} expected facts")
        if mentions and ft.get("includes_any"):
            g *= sum(1 for m in mentions if _text_matches(m, text)) / len(mentions)
        if not mentions and not ft.get("includes_any"):
            fresh = set()
            for c in v.call_order:
                if c.done_ok:
                    fresh |= _id_values(c.result)
            if fresh and not any(_mentions(text, x) for x in fresh):
                g *= 0.7
        grounding = max(0.0, g)
    else:
        grounding = 0.0
    for a in claims:
        s.notes.append(f"TC: false completion claim at {a['t_ms']:.0f} ms: {a.get('text')!r}")
        s.evidence_seq.append(a["seq"])
    if stale_used:
        s.notes.append(f"TC: final uses stale value(s) {sorted(stale_used)}")

    tc = WEIGHTS["TC"] * (0.40 * execution + 0.20 * arg_score + 0.25 * snap_score + 0.15 * grounding)
    s.components.update({"tc_exec": execution, "tc_args": arg_score, "tc_snapshot": snap_score, "tc_grounding": grounding})
    return tc, {"claims": claims, "stale_used": stale_used, "clarified": clarified, "clar_required": clar_required}


def _text_matches(m: Any, text: str) -> bool:
    if isinstance(m, dict):
        if "any_of" in m:
            return any(_text_matches(x, text) for x in m["any_of"])
        if "regex" in m:
            return re.search(m["regex"], text, re.I) is not None
        return False
    return _mentions(text, str(m))


def _forbidden_hits(v: TraceView, exp: Dict[str, Any], sc: Dict[str, Any]) -> List[Call]:
    hits = []
    for fc in exp.get("forbidden_calls") or []:
        t_after = _event_time(v, fc.get("after_event"), sc)
        t_before = _event_time(v, fc.get("before_event"), sc)
        for c in v.call_order:
            if c.tool != fc.get("tool"):
                continue
            if fc.get("args") and args_fraction(fc["args"], c.args) < 1.0:
                continue
            if t_after is not None and c.t_ms < t_after:
                continue
            if t_before is not None and c.t_ms >= t_before:
                continue
            if c not in hits:
                hits.append(c)
    return hits


def _stale_grounding(v: TraceView, exp: Dict[str, Any]) -> set:
    finals = v.finals()
    if not finals:
        return set()
    text = finals[-1].get("text") or ""
    stale, fresh = set(), set()
    for c in v.call_order:
        if c.result is None:
            continue
        ids = _id_values(c.result)
        (stale if c.cancelled_at is not None else fresh if c.status == "ok" else set()).update(ids)
    used = {x for x in stale - fresh if _mentions(text, x)}
    for m in list(exp.get("final_must_not_mention") or []) + list((exp.get("final_text") or {}).get("excludes") or []):
        if _mentions(text, str(m)):
            used.add(str(m))
    return used


def snapshot_accuracy(expected: Optional[Dict[str, Any]], got: Dict[str, Any]) -> Tuple[float, Dict[str, bool]]:
    if expected is None:
        # Expected "no task" (user cancelled everything): intent null and no task slots.
        ok = (got or {}).get("intent") in (None, "", "none")
        return (1.0 if ok else 0.0), {"intent": ok}
    got_slots = (got or {}).get("slots") or {}
    detail: Dict[str, bool] = {}
    if "intent" in expected:
        detail["intent"] = value_matches(expected["intent"], (got or {}).get("intent"))
    for k, m in (expected.get("slots") or {}).items():
        detail[k] = value_matches(m, got_slots.get(k))
    if not detail:
        return 1.0, {}
    return sum(detail.values()) / len(detail), detail


def interruption_points(v: TraceView, exp: Dict[str, Any], scenario: Dict[str, Any]) -> List[float]:
    pts = set()
    first_call = v.call_order[0].t_ms if v.call_order else None
    for r in v.user:
        if r["kind"] == "interrupt":
            pts.add(r["t_ms"])
        elif r["kind"] in EOT_KINDS and _is_eot(r) and first_call is not None and r["t_ms"] > first_call:
            pts.add(r["t_ms"])
    for mc in exp.get("must_cancel") or []:
        t = _event_time(v, mc.get("anchor_event"), scenario)
        if t is not None:
            pts.add(t)
    return sorted(pts)


def score_ir(v: TraceView, exp: Dict[str, Any], scenario: Dict[str, Any], s: ScenarioScore) -> float:
    pts = interruption_points(v, exp, scenario)
    s.components["ir_points"] = float(len(pts))
    if not pts:
        s.components.update({"ir_cancel": 1.0, "ir_stale": 1.0, "ir_snapshot": 1.0, "ir_replan": 1.0})
        return WEIGHTS["IR"]
    grace = float(scenario.get("cancel_grace_ms", 300))
    exp_slots = (exp.get("final_snapshot") or {}).get("slots") or {}

    # cancel promptness
    superseded: List[Tuple[Call, float, float]] = []
    must = exp.get("must_cancel")
    if must:
        for mc in must:
            t = _event_time(v, mc.get("anchor_event"), scenario)
            if t is None:
                continue
            for c in v.call_order:
                if c.in_flight_at(t) and (not mc.get("tool") or c.tool == mc["tool"]) \
                        and args_fraction(mc.get("args"), c.args) == 1.0:
                    superseded.append((c, t, float(mc.get("within_ms", grace))))
    else:
        for t in pts:
            for c in v.call_order:
                if c.in_flight_at(t) and args_contradict(exp_slots, c.args):
                    superseded.append((c, t, grace))
    seen, per = set(), []
    for c, t, g in superseded:
        if c.call_id in seen:
            continue
        seen.add(c.call_id)
        if c.cancelled_at is None:
            per.append(0.0)
            s.notes.append(f"IR: superseded {c.tool} {c.call_id} never cancelled")
            s.evidence_seq.append(c.seq)
        else:
            dt = c.cancelled_at - t
            per.append(1.0 if dt <= g else max(0.0, 1.0 - (dt - g) / max(1.0, 2000.0 - g)))
            if dt > g:
                s.notes.append(f"IR: cancel of {c.call_id} took {dt:.0f} ms (grace {g:.0f})")
    cancel = sum(per) / len(per) if per else 1.0

    # stale re-runs: contradicting the final state after the LAST point (calls that
    # followed an intermediate correction were valid when issued), or re-issuing a
    # cancelled call after the first point without the user asking again
    t0, t_last = pts[0], pts[-1]
    cancelled = [(canonical(c.tool, c.args), c.cancelled_at) for c in v.call_order if c.cancelled_at is not None]
    stale_runs = 0
    for c in v.call_order:
        if c.t_ms < t0:
            continue
        key = canonical(c.tool, c.args)
        reissue = any(k == key and tc <= c.t_ms for k, tc in cancelled) and not _reasked(v, c)
        if (c.t_ms >= t_last and args_contradict(exp_slots, c.args)) or reissue:
            stale_runs += 1
            s.notes.append(f"IR: stale re-run {c.tool} {json.dumps(c.args)} at {c.t_ms:.0f} ms")
            s.evidence_seq.append(c.seq)
    stale = max(0.0, 1.0 - 0.5 * stale_runs - (0.5 if _stale_grounding(v, exp) else 0.0))

    # snapshot updated
    t_last = pts[-1]
    after = [a for a in v.actions if a["t_ms"] >= t_last and isinstance(a.get("snapshot"), dict)]
    first_snap = snapshot_accuracy(exp.get("final_snapshot"), after[0]["snapshot"])[0] if after else 0.0
    finals = v.finals()
    final_snap = snapshot_accuracy(exp.get("final_snapshot"), finals[-1].get("snapshot") or {})[0] if finals else 0.0
    snap = (first_snap + final_snap) / 2

    # re-plan (A25): the window runs to the next user event; user events that
    # follow within 1500 ms chain into one window (barge-in + correction bursts)
    user_times = sorted(r["t_ms"] for r in v.user)
    replans = 0
    for t in pts:
        nxt, cur = v.end_ms + 1, t
        for u in user_times:
            if u <= cur:
                continue
            if u - cur < 1500:
                cur = u
                continue
            nxt = u
            break
        if any(t <= a["t_ms"] < nxt and (_substantive(a) or a["type"] == "tool_call") for a in v.actions):
            replans += 1
    replan = replans / len(pts)
    if replan < 1.0:
        s.notes.append(f"IR: no re-plan after {len(pts) - replans}/{len(pts)} interruption point(s)")

    s.components.update({"ir_cancel": cancel, "ir_stale": stale, "ir_snapshot": snap, "ir_replan": replan})
    return WEIGHTS["IR"] * (0.35 * cancel + 0.25 * stale + 0.25 * snap + 0.15 * replan)


def _reasked(v: TraceView, c: Call) -> bool:
    """A re-issue of a cancelled call is fine if the user asked again after the cancel."""
    cancels = [x.cancelled_at for x in v.call_order
               if x.cancelled_at is not None and canonical(x.tool, x.args) == canonical(c.tool, c.args)]
    last_cancel = max(cancels) if cancels else None
    return last_cancel is not None and any(last_cancel <= r["t_ms"] <= c.t_ms and _is_eot(r) for r in v.user)


def lat_curve(ms: Optional[float]) -> float:
    if ms is None:
        return 0.0
    if ms <= 300:
        return 1.0
    if ms <= 1000:
        return 1.0 - 0.5 * (ms - 300) / 700
    if ms <= 3000:
        return 0.5 - 0.5 * (ms - 1000) / 2000
    return 0.0


def score_lat(v: TraceView, s: ScenarioScore) -> float:
    bounds = [r["t_ms"] for r in v.user if _is_eot(r)]
    subst = [a["t_ms"] for a in v.actions if _substantive(a)]
    points: List[float] = []
    for i, b in enumerate(bounds):
        nxt = bounds[i + 1] if i + 1 < len(bounds) else None
        resp = next((t for t in subst if t >= b and (nxt is None or t < nxt)), None)
        if resp is None and nxt is not None and nxt - b < 1500:
            continue  # A30 merge into the next boundary
        points.append(b)
    lats: List[Optional[float]] = []
    for i, b in enumerate(points):
        nxt = points[i + 1] if i + 1 < len(points) else None
        resp = next((t for t in subst if t >= b and (nxt is None or t < nxt)), None)
        lats.append(None if resp is None else resp - b)
    s.latencies_ms = lats
    if not lats:
        s.components["lat_mean"] = 1.0
        return WEIGHTS["LAT"]
    per = [lat_curve(x) for x in lats]
    for b, x in zip(points, lats):
        if x is None:
            s.notes.append(f"LAT: no substantive response to user input at {b:.0f} ms")
        elif x > 1000:
            s.notes.append(f"LAT: {x:.0f} ms to first substantive action after {b:.0f} ms")
    s.components["lat_mean"] = sum(per) / len(per)
    return WEIGHTS["LAT"] * s.components["lat_mean"]


def score_sp(v: TraceView, exp: Dict[str, Any], s: ScenarioScore) -> float:
    total_out = len(v.actions) + len(v.invalid)
    validity = (len(v.actions) / total_out) if total_out else 0.0
    if v.invalid:
        s.notes.append(f"SP: {len(v.invalid)} invalid action(s): {v.invalid[0].get('errors')}")
        s.evidence_seq.append(v.invalid[0].get("seq", 0))
    if not v.finals():
        validity *= 0.5
    validity = max(0.0, validity - 0.5 * v.agent_errors)
    if v.agent_errors:
        s.notes.append(f"SP: {v.agent_errors} agent error(s)")
    if v.wall_cap:
        validity = 0.0
        s.notes.append("SP: wall-clock cap exceeded")

    ids = max(0.0, 1.0 - 0.25 * (v.reused_ids + v.bad_cancels))
    if v.reused_ids or v.bad_cancels:
        s.notes.append(f"SP: {v.reused_ids} reused call_id(s), {v.bad_cancels} bad cancel(s)")

    dupes = duplicate_writes(v, exp)
    for d in dupes:
        s.notes.append(f"SP: duplicate write {d}")
    for c in v.call_order:
        if c.write and any(d.endswith(c.call_id) for d in dupes):
            s.evidence_seq.append(c.seq)
    nodup = 0.0 if dupes else 1.0
    s.components.update({"sp_valid": validity, "sp_ids": ids, "sp_nodup": nodup})
    return WEIGHTS["SP"] * (0.4 * validity + 0.2 * ids + 0.4 * nodup)


def duplicate_writes(v: TraceView, exp: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    prior: Dict[str, List[Call]] = {}
    for c in v.call_order:
        if not c.write:
            continue
        key = canonical(c.tool, c.args)
        for p in prior.get(key, []):
            pending = p.t_result is None or p.t_result > c.t_ms
            if pending and p.cancelled_at is not None and p.cancelled_at <= c.t_ms:
                pending = False  # cancelled before completion: no side effect (mock semantics)
            succeeded = p.done_ok and p.t_result is not None and p.t_result <= c.t_ms
            unknown = p.status == "timeout" and not is_idempotent_tool(v.manifest.get(c.tool))
            if pending or succeeded or unknown:
                why = "pending" if pending else "succeeded" if succeeded else "timed out (outcome unknown)"
                out.append(f"{c.tool} {json.dumps(c.args)} while earlier call {why}: {c.call_id}")
                break
        prior.setdefault(key, []).append(c)
    counts: Dict[str, int] = {}
    for cm in v.commits:
        counts[cm.get("tool", "?")] = counts.get(cm.get("tool", "?"), 0) + 1
    limits = dict(exp.get("max_writes") or {})
    limits.update(exp.get("max_write_calls") or {})
    for tool, n in limits.items():
        # non-failed calls: ok, or never finished (outcome unknown), cancelled-before-result excluded
        live = sum(1 for c in v.call_order if c.tool == tool
                   and (c.status == "ok" or (c.status is None and c.cancelled_at is None)))
        k = max(counts.get(tool, 0), live)
        if k > n:
            out.append(f"{tool} committed/issued {k} times (max {n})")
    return out


def score_qm(v: TraceView, exp: Dict[str, Any], tc_info: Dict[str, Any], s: ScenarioScore) -> float:
    qm = 1.0
    claims = len(tc_info["claims"])
    qm -= min(0.20, 0.10 * claims)
    fillers = sum(1 for a in v.actions if a["type"] == "speak" and a.get("kind") == "filler")
    turns = sum(1 for r in v.user if _is_eot(r))
    extra = max(0, fillers - max(2, turns))
    if extra:
        s.notes.append(f"QM: {fillers} fillers for {turns} user turns")
    qm -= min(0.10, 0.05 * extra)
    texts = [_norm(a.get("text") or "") for a in v.actions if a["type"] in ("speak", "clarify", "final")]
    repeats = len(texts) - len(set(texts))
    qm -= min(0.09, 0.03 * repeats)
    finals = v.finals()
    if finals:
        text = finals[-1].get("text") or ""
        if s.components.get("tc_grounding", 0) >= 0.99 and not tc_info["stale_used"]:
            qm += 0.05
        if 0 < _count_words(text) < 60:
            qm += 0.05
        if exp.get("must_report_failure"):
            honest = bool(FAILURE_RE.search(text)) and not (SUCCESS_RE.search(text) and not NEGATION_RE.search(text))
            qm += 0.05 if honest else -0.05
            if not honest:
                s.notes.append("QM: failure not reported honestly in final")
    if tc_info["clar_required"]:
        qm += 0.05 if tc_info["clarified"] else -0.05
    return max(QM_MIN, min(QM_MAX, qm))


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def score_trace(records: Iterable[Dict[str, Any]], scenario: Dict[str, Any]) -> ScenarioScore:
    sc = scenario.raw if hasattr(scenario, "raw") else scenario
    v = build_view(records)
    exp = sc.get("expected") or {}
    modality = sc.get("modality", "text")
    s = ScenarioScore(sc.get("id", "?"), modality, 0, 0, 0, 0, 1.0, 0, MULTIMODAL_WEIGHT if modality in ("audio", "visual") else 1.0)
    tc, info = score_tc(v, exp, sc, s)
    ir = score_ir(v, exp, sc, s)
    lat = score_lat(v, s)
    sp = score_sp(v, exp, s)
    qm = score_qm(v, exp, info, s)
    s.TC, s.IR, s.LAT, s.SP, s.QM = round(tc, 2), round(ir, 2), round(lat, 2), round(sp, 2), round(qm, 3)
    s.total = round(min(100.0, (tc + ir + lat + sp) * qm), 2)
    s.evidence_seq = sorted(set(s.evidence_seq))
    return s


def suite_score(scores: List[ScenarioScore]) -> Dict[str, Any]:
    if not scores:
        return {"weighted_mean": 0.0, "mean": 0.0, "n": 0}
    wsum = sum(x.weight for x in scores)
    by_mod: Dict[str, List[float]] = {}
    for x in scores:
        by_mod.setdefault(x.modality, []).append(x.total)
    return {
        "weighted_mean": round(sum(x.total * x.weight for x in scores) / wsum, 2),
        "mean": round(sum(x.total for x in scores) / len(scores), 2),
        "n": len(scores),
        "by_modality": {k: round(sum(v) / len(v), 2) for k, v in sorted(by_mod.items())},
        "by_bucket": {k: round(sum(getattr(x, k) for x in scores) / len(scores), 2) for k in ("TC", "IR", "LAT", "SP", "QM")},
    }


def excerpt(records: List[Dict[str, Any]], seqs: List[int], context: int = 2, limit: int = 14) -> List[str]:
    """Short, human-readable trace lines around the evidence records."""
    by_seq = {int(r.get("seq", i)): r for i, r in enumerate(records)}
    order = sorted(by_seq)
    if not any(sq in by_seq for sq in seqs):
        # nothing specific to point at (e.g. the agent never acted): show the conversation
        seqs = [sq for sq in order if by_seq[sq].get("dir") in ("in", "out")
                and by_seq[sq].get("kind") not in ("tool_manifest", "session_end")][:limit]
        context = 0
    keep: List[int] = []
    for sq in seqs:
        if sq not in by_seq:
            continue
        i = order.index(sq)
        keep += order[max(0, i - context): i + context + 1]
    lines = []
    for sq in sorted(set(keep))[:limit]:
        r = by_seq[sq]
        d = dict(r.get("data") or {})
        for k in ("tools", "session"):
            if k in d:
                d[k] = "..."
        body = json.dumps(d, sort_keys=True, default=str)
        mark = "*" if sq in seqs and context else " "
        lines.append(f"{mark}{r.get('t_ms', 0):>9.0f}ms {r.get('dir', '?'):>3} {r.get('kind', '?'):<16} {body[:150]}")
    return lines


def format_table(scores: List[ScenarioScore]) -> str:
    head = f"{'scenario':<34} {'mod':<6} {'TC':>5} {'IR':>5} {'LAT':>5} {'SP':>5} {'QM':>5} {'total':>6}"
    rows = [head, "-" * len(head)]
    for x in scores:
        rows.append(f"{x.scenario_id[:34]:<34} {x.modality:<6} {x.TC:>5.1f} {x.IR:>5.1f} {x.LAT:>5.1f} {x.SP:>5.1f} {x.QM:>5.2f} {x.total:>6.1f}")
    agg = suite_score(scores)
    b = agg.get("by_bucket", {})
    rows.append("-" * len(head))
    rows.append(f"{'MEAN (unweighted)':<34} {'':<6} {b.get('TC', 0):>5.1f} {b.get('IR', 0):>5.1f} {b.get('LAT', 0):>5.1f} {b.get('SP', 0):>5.1f} {b.get('QM', 0):>5.2f} {agg['mean']:>6.1f}")
    rows.append(f"{'SUITE (multimodal x1.5)':<34} {'':<6} {'':>5} {'':>5} {'':>5} {'':>5} {'':>5} {agg['weighted_mean']:>6.1f}")
    return "\n".join(rows)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                out.append({"seq": 10**9 + i, "t_ms": 0, "dir": "out", "kind": "invalid", "data": {"raw": line[:200]}})
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Score Theme 05 traces with our replica of the guide's rubric.")
    ap.add_argument("--assumptions", action="store_true", help="print every scoring assumption and exit")
    ap.add_argument("--trace", type=Path, help="one trace JSONL")
    ap.add_argument("--scenario", type=Path, help="scenario JSON for --trace")
    ap.add_argument("--traces", type=Path, help="dir of <scenario_id>.jsonl traces")
    ap.add_argument("--scenarios", type=Path, help="dir of scenario JSON files for --traces")
    ap.add_argument("--worst", type=int, default=5)
    ap.add_argument("--json", type=Path, help="write per-scenario scores here")
    a = ap.parse_args(argv)
    if a.assumptions:
        for k, txt in ASSUMPTIONS.items():
            print(f"{k}: {txt}")
        return 0
    pairs: List[Tuple[Path, Dict[str, Any]]] = []
    if a.trace and a.scenario:
        pairs.append((a.trace, json.loads(a.scenario.read_text())))
    elif a.traces and a.scenarios:
        for sp in sorted(a.scenarios.glob("*.json")):
            sc = json.loads(sp.read_text())
            tp = a.traces / f"{sc['id']}.jsonl"
            pairs.append((tp, sc))
    else:
        ap.error("give --trace/--scenario or --traces/--scenarios")
    scores, recs = [], {}
    for tp, sc in pairs:
        r = _load_jsonl(tp) if tp.exists() else []
        recs[sc["id"]] = r
        x = score_trace(r, sc)
        if not tp.exists():
            x.notes.insert(0, f"no trace file {tp}")
        scores.append(x)
    print(format_table(scores))
    print(json.dumps(suite_score(scores), indent=1))
    for x in sorted(scores, key=lambda z: z.total)[: a.worst]:
        print(f"\n## {x.scenario_id} ({x.total:.1f})")
        for n in x.notes[:8]:
            print("  - " + n)
        for line in excerpt(recs[x.scenario_id], x.evidence_seq):
            print("    " + line)
    if a.json:
        a.json.write_text(json.dumps([x.to_dict() for x in scores], indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
