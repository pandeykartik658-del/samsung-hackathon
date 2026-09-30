# sim/mock_tools.py
"""Deterministic mock tool environment (guide section 4: "Mock Environment").

Tools: flight_search (read), book_flight (write), create_ticket (write),
manual_lookup (read), plus any scenario-defined unseen tools.

Per-tool config (all optional):
  latency_ms   base latency (default per tool below)
  timeout_ms   how long a `timeout` fault hangs before reporting status=timeout
  fixtures     [{match: {arg: value}, response: {...}}] first match wins
  faults       [{kind: error|timeout|slow, call_index: n | match: {...},
                 error: "msg", latency_ms: n, factor: x}]
               call_index is 1-based per tool, counted over calls issued.

`plan()` is pure: same (tool, args, call_index) -> same (delay, status, payload).
Writes commit to `state` only when the call completes (a call cancelled before
completion leaves no side effect). ASSUMPTION: the real kit's commit semantics
are unknown; this is the strictest reasonable model for the duplicate-booking checks.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_MANIFEST: List[Dict[str, Any]] = [
    {
        "name": "flight_search",
        "description": "Search available flights between two airports on a date.",
        "side_effect": "read",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "IATA code of departure airport"},
                "destination": {"type": "string", "description": "IATA code of arrival airport"},
                "date": {"type": "string", "description": "Departure date, YYYY-MM-DD"},
                "passengers": {"type": "integer", "description": "Number of passengers"},
            },
            "required": ["origin", "destination", "date"],
        },
    },
    {
        "name": "book_flight",
        "description": "Book a seat on a specific flight for a passenger. State-modifying.",
        "side_effect": "write",
        "parameters": {
            "type": "object",
            "properties": {
                "flight_id": {"type": "string"},
                "passenger_name": {"type": "string"},
            },
            "required": ["flight_id", "passenger_name"],
        },
    },
    {
        "name": "create_ticket",
        "description": "Open a support ticket for a device issue. State-modifying.",
        "side_effect": "write",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "device_model": {"type": "string"},
                "severity": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "required": ["summary"],
        },
    },
    {
        "name": "manual_lookup",
        "description": "Look up troubleshooting steps in a device manual.",
        "side_effect": "read",
        "parameters": {
            "type": "object",
            "properties": {
                "device_model": {"type": "string"},
                "query": {"type": "string"},
                "frame_id": {"type": "string"},
            },
            "required": ["device_model", "query"],
        },
    },
]

DEFAULT_LATENCY_MS = {
    "flight_search": 1200,
    "book_flight": 900,
    "create_ticket": 600,
    "manual_lookup": 700,
}
DEFAULT_TIMEOUT_MS = 5000
DEFAULT_UNSEEN_LATENCY_MS = 500

_TYPE_CHECK = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
}


def _h(*parts: Any) -> int:
    s = json.dumps(parts, sort_keys=True, default=str)
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


def canonical_args(args: Dict[str, Any]) -> str:
    """Canonical form used for duplicate detection (case/space-insensitive strings)."""
    def norm(v):
        if isinstance(v, str):
            return " ".join(v.strip().lower().split())
        if isinstance(v, dict):
            return {k: norm(x) for k, x in sorted(v.items())}
        if isinstance(v, list):
            return [norm(x) for x in v]
        return v
    return json.dumps(norm(args), sort_keys=True)


def value_matches(expected: Any, actual: Any) -> bool:
    """Matcher used by fixtures, faults and expected blocks.

    scalar             equal (strings case/space-insensitive)
    {"any_of": [...]}  any listed value matches
    {"regex": "..."}   re.search, case-insensitive, on str(actual)
    {"present": bool}  key present / absent (handled by args_match)
    dict               recursive subset match
    """
    import re

    if isinstance(expected, dict):
        if "any_of" in expected:
            return any(value_matches(e, actual) for e in expected["any_of"])
        if "regex" in expected:
            return actual is not None and re.search(expected["regex"], str(actual), re.I) is not None
        if "present" in expected:
            return (actual is not None) == bool(expected["present"])
        return isinstance(actual, dict) and args_match(expected, actual)
    if isinstance(expected, str) and isinstance(actual, str):
        return " ".join(expected.lower().split()) == " ".join(actual.lower().split())
    return expected == actual


def args_match(expected: Optional[Dict[str, Any]], actual: Dict[str, Any]) -> bool:
    for k, exp in (expected or {}).items():
        if isinstance(exp, dict) and set(exp) == {"present"}:
            if (k in actual) != bool(exp["present"]):
                return False
            continue
        if not value_matches(exp, actual.get(k)):
            return False
    return True


def normalize_params(params: Any) -> Dict[str, Any]:
    """Accept the three manifest parameter styles protocol.parse_tool accepts:
    JSON Schema {properties, required}, a flat {name: spec} map, or [{name, ...}]."""
    if isinstance(params, list):
        props = {str(p["name"]): {k: v for k, v in p.items() if k != "name"} for p in params if isinstance(p, dict)}
        return {"properties": props, "required": [n for n, sp in props.items() if sp.get("required")]}
    if isinstance(params, dict) and "properties" not in params:
        props = {k: (v if isinstance(v, dict) else {"type": str(v)}) for k, v in params.items()
                 if k not in ("type", "required", "additionalProperties")}
        return {"properties": props, "required": [n for n, sp in props.items() if sp.get("required") is True]}
    return params if isinstance(params, dict) else {}


def validate_args(tool_def: Dict[str, Any], args: Dict[str, Any]) -> List[str]:
    params = normalize_params(tool_def.get("parameters") or {})
    props = params.get("properties") or {}
    errs = []
    for r in params.get("required", []):
        if r not in args or args[r] in (None, ""):
            errs.append(f"missing required argument '{r}'")
    for k, v in args.items():
        spec = props.get(k)
        if spec is None:
            if params.get("additionalProperties") is False:
                errs.append(f"unexpected argument '{k}'")
            continue
        t = spec.get("type")
        if t in _TYPE_CHECK and not _TYPE_CHECK[t](v):
            errs.append(f"argument '{k}' must be {t}")
        if "enum" in spec and v not in spec["enum"]:
            errs.append(f"argument '{k}' must be one of {spec['enum']}")
        if "pattern" in spec and isinstance(v, str):
            import re
            if not re.fullmatch(spec["pattern"], v):
                errs.append(f"argument '{k}' does not match {spec['pattern']}")
    return errs


@dataclass
class Plan:
    delay_ms: float
    status: str  # ok | error | timeout
    payload: Dict[str, Any]  # result (ok) or {"error": msg}
    commit: Optional[Dict[str, Any]] = None  # write side effect applied on completion
    rejected: Optional[str] = None  # "unknown_tool" | "invalid_arguments" (protocol error)
    retryable: bool = False  # hint sent with error/timeout results


@dataclass
class MockState:
    bookings: List[Dict[str, Any]] = field(default_factory=list)
    tickets: List[Dict[str, Any]] = field(default_factory=list)
    writes: List[Dict[str, Any]] = field(default_factory=list)  # every committed write

    def to_dict(self) -> Dict[str, Any]:
        return {"bookings": self.bookings, "tickets": self.tickets, "writes": self.writes}


class MockToolServer:
    def __init__(self, tools_cfg: Optional[Dict[str, Any]] = None):
        cfg = tools_cfg or {}
        enabled = cfg.get("enabled")
        base = [t for t in DEFAULT_MANIFEST if enabled is None or t["name"] in enabled]
        extra = [dict(t) for t in cfg.get("extra", [])]
        self._unseen_mocks: Dict[str, Dict[str, Any]] = {t["name"]: t.pop("mock", {}) for t in extra}
        self.manifest: List[Dict[str, Any]] = copy.deepcopy(base) + extra
        for t in self.manifest:
            # Explicit flag too: protocol.py and the trace viewer check `read_only` first.
            t.setdefault("side_effect", "read" if t.get("read_only") else "write")
            t.setdefault("read_only", t["side_effect"] != "write")
        self.by_name = {t["name"]: t for t in self.manifest}
        self.config: Dict[str, Dict[str, Any]] = cfg.get("config", {})
        self.call_counts: Dict[str, int] = {}
        self.state = MockState()

    def is_write(self, tool: str) -> bool:
        t = self.by_name.get(tool)
        return bool(t) and t.get("side_effect") == "write"

    def next_call_index(self, tool: str) -> int:
        self.call_counts[tool] = self.call_counts.get(tool, 0) + 1
        return self.call_counts[tool]

    # -- planning ------------------------------------------------------------
    def plan(self, tool: str, args: Dict[str, Any], call_index: int) -> Plan:
        tdef = self.by_name.get(tool)
        if tdef is None:
            return Plan(50, "error", {"error": f"unknown_tool: {tool}"}, rejected="unknown_tool")
        errs = validate_args(tdef, args)
        if errs:
            return Plan(50, "error", {"error": "invalid_arguments: " + "; ".join(errs)},
                        rejected="invalid_arguments")
        cfg = self.config.get(tool, {})
        default_lat = DEFAULT_LATENCY_MS.get(tool, self._unseen_mocks.get(tool, {}).get("latency_ms", DEFAULT_UNSEEN_LATENCY_MS))
        latency = float(cfg.get("latency_ms", default_lat))
        for f in cfg.get("faults", []):
            if not self._fault_applies(f, args, call_index):
                continue
            kind = f["kind"]
            if kind == "error":
                return Plan(f.get("latency_ms", latency), "error", {"error": f.get("error", "internal_error")},
                            retryable=bool(f.get("retryable", True)))
            if kind == "timeout":
                return Plan(f.get("timeout_ms", cfg.get("timeout_ms", DEFAULT_TIMEOUT_MS)), "timeout",
                            {"error": "timeout"}, retryable=True)
            if kind == "slow":
                latency = float(f["latency_ms"]) if "latency_ms" in f else latency * float(f.get("factor", 4.0))
        result, commit = self._respond(tool, args, cfg)
        return Plan(latency, "ok", result, commit)

    @staticmethod
    def _fault_applies(f: Dict[str, Any], args: Dict[str, Any], call_index: int) -> bool:
        if "call_index" in f and f["call_index"] != call_index:
            return False
        if "match" in f and not args_match(f["match"], args):
            return False
        return True

    def _respond(self, tool: str, args: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        for fx in cfg.get("fixtures", []):
            if args_match(fx.get("match"), args):
                resp = copy.deepcopy(fx["response"])
                return resp, ({"tool": tool, "args": args, "result": resp} if self.is_write(tool) else None)
        handler = getattr(self, f"_gen_{tool}", None)
        if handler is not None:
            resp = handler(args)
        else:
            mock = self._unseen_mocks.get(tool, {})
            resp = copy.deepcopy(mock.get("response", {"ok": True}))
        return resp, ({"tool": tool, "args": args, "result": resp} if self.is_write(tool) else None)

    # -- generated responses (used when no fixture matches) -------------------
    def _gen_flight_search(self, a: Dict[str, Any]) -> Dict[str, Any]:
        seed = _h("fs", a.get("origin", "").upper(), a.get("destination", "").upper(), a.get("date"))
        airlines = ["6E", "AI", "UK", "SG", "QP"]
        flights = []
        for i in range(3):
            s = _h(seed, i)
            flights.append({
                "flight_id": f"{airlines[s % 5]}-{100 + s % 900}",
                "origin": a.get("origin", "").upper(),
                "destination": a.get("destination", "").upper(),
                "date": a.get("date"),
                "depart": f"{6 + (s >> 8) % 14:02d}:{(s >> 16) % 4 * 15:02d}",
                "price_inr": 3000 + (s >> 24) % 7000,
            })
        flights.sort(key=lambda f: f["depart"])
        return {"flights": flights}

    def _gen_book_flight(self, a: Dict[str, Any]) -> Dict[str, Any]:
        n = len(self.state.bookings)  # a duplicate booking gets a different ref, as a real system would
        ref = "".join("ABCDEFGHJKLMNPQRSTUVWXYZ23456789"[(_h("bk", a, n) >> (5 * i)) % 32] for i in range(6))
        return {"booking_ref": ref, "flight_id": a.get("flight_id"), "status": "confirmed"}

    def _gen_create_ticket(self, a: Dict[str, Any]) -> Dict[str, Any]:
        n = len(self.state.tickets)
        return {"ticket_id": f"TCK-{_h('tk', a, n) % 100000:05d}", "status": "open"}

    def _gen_manual_lookup(self, a: Dict[str, Any]) -> Dict[str, Any]:
        return {"device_model": a.get("device_model"), "section": "General troubleshooting",
                "steps": ["Power-cycle the device.", "Check the manual index for the reported code."]}

    # -- side effects ----------------------------------------------------------
    def commit(self, commit: Dict[str, Any], t_ms: float, call_id: str) -> None:
        rec = {"t_ms": t_ms, "call_id": call_id, **commit}
        self.state.writes.append(rec)
        if commit["tool"] == "book_flight":
            self.state.bookings.append(rec)
        elif commit["tool"] == "create_ticket":
            self.state.tickets.append(rec)
