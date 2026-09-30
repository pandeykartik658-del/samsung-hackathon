# sim/scenario.py
"""Scenario file loading and validation.

Scenario JSON (ASSUMPTION: our own format; the official kit's is unreleased):
{
  "id": "s01_...", "title": str, "modality": "text"|"audio"|"visual",
  "description": str,
  "duration_ms": int (virtual deadline, <= 120000),
  "session": {...}                       sent inside the tool_manifest event,
  "tools": {"enabled": [...], "extra": [tool_def + "mock"], "config": {tool: cfg}},
  "cancel_grace_ms": int (default 300),
  "events": [{"id": "e1", "t_ms": int, "type": <event type>, ...fields}],
  "expected": {...}                      see sim/scoring.py
}
A tool_manifest event at t=0 is injected automatically unless the scenario has one.
Asset paths in events are relative to the scenario file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from . import wire

MAX_DURATION_MS = 120_000
MODALITIES = ("text", "audio", "visual")


class ScenarioError(ValueError):
    pass


@dataclass
class Scenario:
    id: str
    title: str
    modality: str
    events: List[Dict[str, Any]]
    expected: Dict[str, Any]
    duration_ms: int = 30_000
    description: str = ""
    session: Dict[str, Any] = field(default_factory=dict)
    tools: Dict[str, Any] = field(default_factory=dict)
    cancel_grace_ms: int = 300
    base_dir: Path = field(default_factory=Path.cwd)
    raw: Dict[str, Any] = field(default_factory=dict)

    def event_time(self, event_id: str) -> float:
        for e in self.events:
            if e["id"] == event_id:
                return float(e["t_ms"])
        raise KeyError(event_id)


def from_dict(d: Dict[str, Any], base_dir: Union[str, Path, None] = None) -> Scenario:
    errs = []
    for k in ("id", "modality", "events", "expected"):
        if k not in d:
            errs.append(f"missing '{k}'")
    if errs:
        raise ScenarioError(f"{d.get('id', '?')}: " + "; ".join(errs))
    if d["modality"] not in MODALITIES:
        errs.append(f"modality must be one of {MODALITIES}")
    dur = int(d.get("duration_ms", 30_000))
    if not 0 < dur <= MAX_DURATION_MS:
        errs.append(f"duration_ms must be in (0, {MAX_DURATION_MS}]")
    seen = set()
    last_t = -1.0
    for e in d["events"]:
        eid = e.get("id")
        if not eid or eid in seen:
            errs.append(f"event id missing or duplicate: {eid!r}")
        seen.add(eid)
        if e.get("type") not in wire.EVENT_TYPES or e.get("type") in ("tool_result", "session_end"):
            errs.append(f"{eid}: scripted event type {e.get('type')!r} not allowed")
        t = e.get("t_ms")
        if not isinstance(t, (int, float)) or t < 0 or t >= dur:
            errs.append(f"{eid}: t_ms must be within [0, duration_ms)")
        elif t < last_t:
            errs.append(f"{eid}: events must be sorted by t_ms")
        else:
            last_t = t
    exp = d["expected"]
    for key in ("required_calls", "forbidden_calls", "final_snapshot"):
        if key not in exp:
            errs.append(f"expected.{key} missing")
    for ref in _event_refs(exp):
        if ref not in seen:
            errs.append(f"expected references unknown event '{ref}'")
    if errs:
        raise ScenarioError(f"{d['id']}: " + "; ".join(errs))
    return Scenario(
        id=d["id"], title=d.get("title", d["id"]), modality=d["modality"],
        events=d["events"], expected=exp, duration_ms=dur,
        description=d.get("description", ""), session=d.get("session", {}),
        tools=d.get("tools", {}), cancel_grace_ms=int(d.get("cancel_grace_ms", 300)),
        base_dir=Path(base_dir) if base_dir else Path.cwd(), raw=d,
    )


def _event_refs(exp: Dict[str, Any]) -> List[str]:
    refs = []
    for key in ("forbidden_calls", "must_cancel", "clarify"):
        items = exp.get(key) or []
        if isinstance(items, dict):
            items = [items]
        for it in items:
            for k in ("after_event", "before_event", "anchor_event"):
                if k in it:
                    refs.append(it[k])
    return refs


def load(path: Union[str, Path]) -> Scenario:
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        d = json.load(f)
    sc = from_dict(d, base_dir=p.parent)
    for e in sc.events:
        if "path" in e and not (p.parent / e["path"]).exists():
            raise ScenarioError(f"{sc.id}: asset not found: {e['path']} (run python -m sim.assets)")
    return sc


def is_harness_scenario(d: Any) -> bool:
    """Our format: has `expected` and events with `t_ms`. Other JSON files are skipped."""
    return isinstance(d, dict) and "expected" in d and all("t_ms" in e for e in d.get("events", []))


def load_dir(path: Union[str, Path], skipped: Optional[List[Path]] = None) -> List[Scenario]:
    """Load one file or every *.json in a directory. Files in another format are
    skipped (and listed in `skipped` if given); malformed ones in ours raise."""
    p = Path(path)
    if p.is_file():
        return [load(p)]
    out = []
    for f in sorted(p.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ScenarioError(f"{f}: invalid JSON: {exc}") from exc
        if not is_harness_scenario(d):
            if skipped is not None:
                skipped.append(f)
            continue
        out.append(load(f))
    return out
