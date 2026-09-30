# sim/trace.py
"""Event/action trace, written as JSONL (one record per line).

Record: {"seq": int, "t_ms": float, "dir": "in"|"out"|"sys", "kind": str, "data": {...}}
  in   event delivered to the agent (kind = event type)
  out  action received from the agent (kind = action type, or "invalid")
  sys  harness bookkeeping: scenario_start, tool_started, tool_completed,
       tool_cancelled, cancel_ignored, write_committed, agent_error,
       wall_cap_exceeded, scenario_end
The scorer (sim/scoring.py) works from these records only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


class Trace:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def log(self, t_ms: float, direction: str, kind: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        rec = {"seq": len(self.records), "t_ms": round(t_ms, 3), "dir": direction, "kind": kind, "data": data or {}}
        self.records.append(rec)
        return rec

    def write_jsonl(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r, sort_keys=True, default=_default) + "\n")
        return p

    def of(self, direction: str, kind: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        return (r for r in self.records if r["dir"] == direction and (kind is None or r["kind"] == kind))


def _default(o: Any) -> Any:
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return repr(o)


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
