# /mnt/project-files/theme5/theme5/tools.py
"""Manifest-driven tools: registry, slot->parameter binding, argument
validation, retry policy, non-blocking call runner and chained calls.

Nothing here knows a tool by name. Everything is derived from the manifest
(parsed in protocol_manifest.py), so unseen tools work without code changes.

Wire-format facts are ASSUMPTIONs isolated in protocol.py / protocol_manifest.py.
Policy choices made here are marked ASSUMPTION inline.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Awaitable, Callable, Iterable, Mapping

from .protocol import DEFAULT_CHOICE, MAX_RETRIES, IdGen, ToolResult, ToolSpec, new_id
from .protocol_manifest import ParamDef, ToolDef, from_toolspec, parse_manifest_defs, parse_tool_def


def canonical_args(args: dict[str, Any]) -> str:
    def norm(v: Any) -> Any:
        if isinstance(v, str):
            return v.strip().lower()
        if isinstance(v, dict):
            return {k: norm(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        return v
    return json.dumps(norm(args), sort_keys=True, separators=(",", ":"), default=str)


# =============================================================================
# Registry
# =============================================================================

class ToolRegistry:
    """Current scenario's tools. A new manifest replaces the set (SPEC U11)."""

    def __init__(self, tools: Iterable[ToolDef | ToolSpec | Mapping[str, Any]] = ()) -> None:
        self._tools: dict[str, ToolDef] = {}
        self.errors: list[str] = []
        self.load(tools)

    def load(self, tools: Iterable[ToolDef | ToolSpec | Mapping[str, Any]]) -> None:
        for t in tools:
            try:
                d = t if isinstance(t, ToolDef) else from_toolspec(t) if isinstance(t, ToolSpec) else parse_tool_def(t)
            except Exception as exc:
                self.errors.append(str(exc))
                continue
            self._tools[d.name] = d

    def load_manifest(self, payload: Any, replace: bool = True) -> list[str]:
        """Parse a raw manifest payload. Never raises; returns skipped-entry errors."""
        tools, errors = parse_manifest_defs(payload)
        if replace and tools:
            self._tools.clear()
        self.load(tools)
        self.errors.extend(errors)
        return errors

    def get(self, name: str) -> ToolDef | None:
        t = self._tools.get(name)
        if t is None and isinstance(name, str):  # tolerate case/separator drift
            key = _squash(name)
            t = next((d for n, d in self._tools.items() if _squash(n) == key), None)
        return t

    def all(self) -> list[ToolDef]:
        return list(self._tools.values())

    def read_only(self) -> list[ToolDef]:
        return [t for t in self._tools.values() if t.read_only]

    def state_modifying(self) -> list[ToolDef]:
        return [t for t in self._tools.values() if t.state_modifying]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.get(name) is not None

    def __len__(self) -> int:
        return len(self._tools)


# =============================================================================
# Name matching: slots -> parameters
# =============================================================================

_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

# Abbreviations seen in real APIs, expanded before concept lookup.
_ABBREV = {
    "src": "source", "dst": "destination", "dest": "destination", "orig": "origin", "frm": "from",
    "dep": "departure", "arr": "arrival", "num": "number", "no": "number", "nbr": "number", "qty": "quantity",
    "cnt": "count", "pax": "passengers", "dt": "date", "tm": "time", "flt": "flight", "ref": "id",
    "reference": "id", "identifier": "id", "desc": "description", "addr": "address", "tel": "phone",
    "mobile": "phone", "prio": "priority", "sev": "severity", "img": "image", "pic": "image",
    "err": "error", "msg": "message", "dev": "device", "loc": "location", "cty": "city", "req": "requested",
    "iata": "airport", "info": "information", "mail": "email", "ppl": "people",
    "persons": "people", "person": "people", "travelers": "travellers", "traveler": "travellers",
    "traveller": "travellers", "passenger": "passengers", "seat": "seats", "adult": "adults",
}

# Concept -> {token: weight}. Keys are the slot names nlu/slots produce.
# ASSUMPTION: this vocabulary; unknown slot names fall back to their own tokens.
_CONCEPTS: dict[str, dict[str, float]] = {
    "origin": {"origin": 2, "source": 2, "from": 2, "departure": 1, "depart": 1, "departing": 1,
               "leaving": 1, "start": 1, "pickup": 1},
    "destination": {"destination": 2, "to": 2, "arrival": 1, "arrive": 1, "arriving": 1, "target": 1,
                    "going": 1, "dropoff": 1},
    "date": {"date": 2, "day": 2, "when": 2},
    "time": {"time": 2, "hour": 1, "clock": 1},
    "passengers": {"passengers": 2, "seats": 2, "travellers": 2, "people": 2, "adults": 2, "guests": 2,
                   "party": 1, "quantity": 1, "count": 0.5, "number": 0.5},
    "name": {"name": 2, "fullname": 2, "full": 1, "holder": 1, "customer": 1, "people": 0.5},
    "email": {"email": 2, "address": 0.5},
    "phone": {"phone": 2, "contact": 1, "number": 0.5},
    "cabin": {"cabin": 2, "class": 2, "fare": 1, "tier": 1},
    "priority": {"priority": 2, "severity": 2, "urgency": 2, "level": 0.5},
    "device": {"device": 2, "product": 2, "appliance": 2, "model": 1, "unit": 1, "equipment": 1, "item": 0.5},
    "error_code": {"error": 2, "code": 1, "fault": 1},
    "part": {"part": 2, "component": 2, "button": 1, "label": 1},
    "choice": {"choice": 2, "option": 2, "selection": 2, "index": 1},
    "frame": {"frame": 2, "image": 2, "photo": 2, "picture": 2, "snapshot": 1},
    "issue": {"issue": 2, "problem": 2, "summary": 2, "description": 1, "subject": 1, "details": 1},
}

_DESC_STOP = {"the", "a", "an", "of", "to", "from", "on", "in", "for", "and", "or", "is", "be", "at", "by",
              "with", "as", "this", "that", "not", "when", "e", "g", "eg", "yyyy", "mm", "dd", "format", "iso"}
_LABEL_NOISE = {"requested", "iso", "utc", "val", "value", "str", "param", "arg"}
_EXACT = 100.0
_MIN_SCORE = 0.8  # ASSUMPTION: below this a name match is not trusted
_CLEAR_SCORE = 2.0  # at or above this the value was clearly meant for the param
_DESC_WEIGHT = 0.4


def _squash(s: Any) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def tokens(name: str) -> list[str]:
    """'numSeatsReq' -> ['number', 'seats', 'requested']; 'qx_src_iata' -> ['qx', 'source', 'airport']."""
    out: list[str] = []
    for part in re.split(r"[\s_\-./:,;()]+", str(name)):
        for w in _CAMEL.findall(part):
            w = w.lower()
            out.append(_ABBREV.get(w, w))
    return out


def _concept_weights(slot: str) -> dict[str, float]:
    c = _CONCEPTS.get(slot)
    own = {t: 2.0 for t in tokens(slot)}
    return {**own, **c} if c else own


def match_score(slot: str, p: ParamDef) -> float:
    """How strongly a slot name (or output key) matches a parameter by name."""
    if _squash(slot) == _squash(p.name) or any(_squash(slot) == _squash(a) for a in p.aliases):
        return _EXACT
    w = _concept_weights(slot)
    name_toks = set(tokens(p.name))
    score = sum(w.get(t, 0.0) for t in name_toks)
    desc_toks = {t for t in tokens(p.description) if t not in _DESC_STOP} - name_toks
    score += _DESC_WEIGHT * sum(w.get(t, 0.0) for t in desc_toks)
    return score


def humanize(p: ParamDef) -> str:
    """Short spoken label for a parameter, preferring a short description."""
    d = p.description.strip().rstrip(".")
    if d and len(d.split()) <= 6:
        return d[0].lower() + d[1:]
    words = [t for t in tokens(p.name) if len(t) > 1 and t not in _LABEL_NOISE]
    if p.format in ("date", "date-time") and "date" not in words:
        words = [w for w in words if w not in ("when", "day")] + ["date"]
    return " ".join(words) or p.name


# =============================================================================
# Coercion and validation
# =============================================================================

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_EMAIL = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.]+$")
_TRUE = {"true", "yes", "y", "1", "on"}
_FALSE = {"false", "no", "n", "0", "off"}


class CoercionError(ValueError):
    pass


def _enum_match(v: Any, enum: tuple[Any, ...]) -> Any:
    if v in enum:
        return v
    key = _squash(v) if isinstance(v, str) else None
    for e in enum:
        if key is not None and isinstance(e, str) and _squash(e) == key:
            return e
        if not isinstance(e, str) and isinstance(v, str):
            try:
                if type(e)(v) == e:
                    return e
            except (TypeError, ValueError):
                pass
    raise CoercionError(f"must be one of {list(enum)}")


def coerce(v: Any, p: ParamDef) -> Any:
    """Convert a slot value to the parameter's declared type. Raises CoercionError."""
    t = p.type
    if v is None:
        raise CoercionError("no value")
    if t == "array":
        items = list(v) if isinstance(v, (list, tuple, set)) else [v]
        if p.items_type:
            sub = ParamDef(p.name, p.items_type)
            items = [coerce(x, sub) for x in items]
        out: Any = items
    elif t == "integer":
        if isinstance(v, bool):
            raise CoercionError("expected an integer")
        if isinstance(v, int):
            out = v
        elif isinstance(v, float) and v.is_integer():
            out = int(v)
        elif isinstance(v, str) and re.fullmatch(r"\s*[+-]?\d+\s*", v):
            out = int(v)
        else:
            raise CoercionError("expected an integer")
    elif t == "number":
        if isinstance(v, bool):
            raise CoercionError("expected a number")
        try:
            out = v if isinstance(v, (int, float)) else float(str(v).strip())
        except ValueError:
            raise CoercionError("expected a number") from None
    elif t == "boolean":
        if isinstance(v, bool):
            out = v
        elif str(v).strip().lower() in _TRUE:
            out = True
        elif str(v).strip().lower() in _FALSE:
            out = False
        else:
            raise CoercionError("expected yes or no")
    elif t == "string":
        if isinstance(v, (dict, list, tuple)):
            raise CoercionError("expected text")
        out = v.isoformat() if isinstance(v, (date, datetime)) else str(v)
    else:  # object / any
        out = v
    if p.enum:
        out = _enum_match(out, p.enum)
    return out


def _format_ok(v: Any, fmt: str | None) -> bool:
    if not fmt or not isinstance(v, str):
        return True
    try:
        if fmt == "date":
            return bool(_ISO_DATE.match(v)) and bool(date.fromisoformat(v))
        if fmt == "date-time":
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return True
        if fmt == "email":
            return bool(_EMAIL.match(v))
    except ValueError:
        return False
    return True


def validate_args(tool: ToolDef, args: Mapping[str, Any]) -> list[str]:
    """Schema check before any call leaves the agent. Empty list == valid."""
    if not isinstance(args, Mapping):
        return ["args must be an object"]
    errs: list[str] = []
    known = {p.name: p for p in tool.params}
    for k in args:
        if k not in known:
            errs.append(f"{k}: unknown parameter")
    for p in tool.params:
        if p.name not in args or args[p.name] is None:
            if p.required:
                errs.append(f"{p.name}: required")
            continue
        v = args[p.name]
        try:
            cv = coerce(v, p)
        except CoercionError as e:
            errs.append(f"{p.name}: {e}")
            continue
        if cv != v or type(cv) is not type(v):
            errs.append(f"{p.name}: wrong type (expected {p.type})")
            continue
        if p.type in ("integer", "number"):
            if p.minimum is not None and v < p.minimum:
                errs.append(f"{p.name}: below minimum {p.minimum:g}")
            if p.maximum is not None and v > p.maximum:
                errs.append(f"{p.name}: above maximum {p.maximum:g}")
        if isinstance(v, str):
            if p.min_length is not None and len(v) < p.min_length:
                errs.append(f"{p.name}: shorter than {p.min_length}")
            if p.max_length is not None and len(v) > p.max_length:
                errs.append(f"{p.name}: longer than {p.max_length}")
            if p.pattern and not re.search(p.pattern, v):
                errs.append(f"{p.name}: does not match pattern")
            if not _format_ok(v, p.format):
                errs.append(f"{p.name}: not a valid {p.format}")
    try:
        json.dumps(dict(args))
    except (TypeError, ValueError):
        errs.append("args not JSON-serialisable")
    return errs


def _param_error(p: ParamDef, v: Any) -> str | None:
    """Single-parameter check used during binding (range/format/pattern)."""
    errs = validate_args(ToolDef("_", "", (p,), True), {p.name: v})
    return errs[0].split(": ", 1)[1] if errs else None


# =============================================================================
# Binding: slots (+ previous tool outputs) -> arguments
# =============================================================================

@dataclass
class Binding:
    args: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)  # required params with no value
    invalid: dict[str, str] = field(default_factory=dict)  # param -> reason (value given but unusable)
    sources: dict[str, str] = field(default_factory=dict)  # param -> slot/output key used

    @property
    def complete(self) -> bool:
        return not self.missing and not self.invalid


def _value_evidence(v: Any, p: ParamDef) -> bool:
    """Value shape alone identifies the parameter (used only for leftovers)."""
    if p.enum:
        try:
            _enum_match(v, p.enum)
            return True
        except CoercionError:
            return False
    if isinstance(v, str):
        if p.format == "date":
            return bool(_ISO_DATE.match(v))
        if p.format == "email":
            return bool(_EMAIL.match(v))
    return False


def _assign(tool: ToolDef, values: Mapping[str, Any], b: Binding, taken: set[str]) -> None:
    """Greedy one-to-one assignment by name score, type-checked."""
    order = {k: i for i, k in enumerate(values)}
    cands: list[tuple[float, int, str, str]] = []
    for p in tool.params:
        if p.name in taken:
            continue
        for k, v in values.items():
            if v is None:
                continue
            s = match_score(k, p)
            if p.type == "string" and not p.enum and isinstance(v, (int, float, bool)) and s < _EXACT:
                s *= 0.5  # a number is weak evidence for a text param (e.g. passengers -> name)
            if s >= _MIN_SCORE:
                cands.append((s, -order[k], p.name, k))
    cands.sort(reverse=True)
    used: set[str] = set()
    for s, _, pname, k in cands:
        if pname in taken or k in used:
            continue
        p = tool.param(pname)
        assert p is not None
        v = values[k]
        try:
            cv = coerce(v, p)
            err = _param_error(p, cv)
        except CoercionError as e:
            cv, err = None, str(e)
        if err:
            if s >= _CLEAR_SCORE:  # clearly meant for this param: report, don't silently skip
                b.invalid.setdefault(pname, f"{v!r}: {err}")
            continue
        b.args[pname] = cv
        b.sources[pname] = k
        b.invalid.pop(pname, None)
        taken.add(pname)
        used.add(k)
    # Leftover values whose shape uniquely fits one unfilled parameter.
    for p in tool.params:
        if p.name in taken or p.name in b.invalid:
            continue
        fits = [k for k, v in values.items() if k not in used and v is not None and _value_evidence(v, p)]
        if len(fits) == 1:
            k = fits[0]
            b.args[p.name] = coerce(values[k], p)
            b.sources[p.name] = k
            taken.add(p.name)
            used.add(k)


_NOT_ARGS = ("choice", "frame")  # selection/perception slots, not tool inputs by themselves


def bind_args(tool: ToolDef, slots: Mapping[str, Any] | None = None,
              context: Mapping[str, Any] | None = None,
              explicit: Mapping[str, Any] | None = None) -> Binding:
    """Map values to a tool's parameters by name and type.

    Priority: explicit args > user slots > context (outputs of earlier tools).
    Required params without a value -> `missing`; values that clearly target a
    param but fail its type/enum/range -> `invalid`. Required params with a
    manifest default take the default; the idempotency param is filled later.
    """
    b = Binding()
    taken: set[str] = set()
    for k, v in (explicit or {}).items():
        p = tool.param(k)
        if p is None:
            b.invalid[k] = "unknown parameter"
            continue
        try:
            b.args[k] = coerce(v, p)
            taken.add(k)
            b.sources[k] = "explicit"
        except CoercionError as e:
            b.invalid[k] = f"{v!r}: {e}"
    if slots:
        _assign(tool, {k: v for k, v in slots.items() if k not in _NOT_ARGS}, b, taken)
    if context:
        _assign(tool, context, b, taken)
    for p in tool.params:
        if p.name in taken or p.name in b.invalid:
            continue
        if p.required and p.has_default:
            b.args[p.name] = p.default
            b.sources[p.name] = "default"
        elif p.required and p.name != tool.idempotency_param:
            b.missing.append(p.name)
    return b


def _join(labels: list[str]) -> str:
    return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]


def clarification(tool: ToolDef, missing: Iterable[str], invalid: Mapping[str, str] | None = None) -> str:
    """One targeted question covering the missing / unusable parameters."""
    invalid = dict(invalid or {})
    names = list(invalid) + [m for m in missing if m not in invalid]
    params = [p for p in (tool.param(n) for n in names) if p is not None]
    if not params:
        return "Could you give me a bit more detail?"
    first = params[0]
    if len(params) == 1 and first.enum and 1 < len(first.enum) <= 5:
        opts = [str(e).replace("_", " ").lower() for e in first.enum]
        lead = "That option isn't available. " if first.name in invalid else ""
        return f"{lead}Which {humanize(first)} would you like: {', '.join(opts[:-1])} or {opts[-1]}?"
    labels = _join([humanize(p) for p in params])
    if all(p.name in invalid for p in params):
        return f"I couldn't use the {labels} you gave. Could you say it again?"
    return f"Could you tell me the {labels}?"


# =============================================================================
# Chaining: output of one tool -> inputs of the next
# =============================================================================

_PRICE_KEYS = ("price", "fare", "cost", "amount", "total")
_TIME_KEYS = ("depart", "departure", "time", "start", "date")
_DURATION_KEYS = ("duration", "minutes", "travel_time", "eta")


def _pick_key(item: Mapping[str, Any], hints: tuple[str, ...]) -> str | None:
    for h in hints:
        for k, v in item.items():
            if h in k.lower() and isinstance(v, (int, float, str)) and not isinstance(v, bool):
                return k
    return None


def pick_item(items: list[Any], choice: Any = None) -> Any:
    """choice: 1-based int, -1 for last, or 'cheapest'/'earliest'/'fastest'/'latest'.
    ASSUMPTION (protocol.DEFAULT_CHOICE): no usable choice -> first item."""
    if not items:
        return None
    if isinstance(choice, str) and all(isinstance(i, Mapping) for i in items):
        c = choice.lower()
        hints = {"cheapest": _PRICE_KEYS, "earliest": _TIME_KEYS, "latest": _TIME_KEYS,
                 "fastest": _DURATION_KEYS}.get(c)
        key = _pick_key(items[0], hints) if hints else None
        ranked = [i for i in items if key in i] if key else []
        try:
            if ranked:
                return (max if c == "latest" else min)(ranked, key=lambda i: i[key])
        except TypeError:  # mixed value types: fall through to default
            pass
        choice = None
    n = choice if isinstance(choice, int) and not isinstance(choice, bool) else DEFAULT_CHOICE
    if n < 0:
        return items[n] if -n <= len(items) else items[0]
    return items[n - 1] if 1 <= n <= len(items) else items[0]


def _singular(s: str) -> str:
    return s[:-1] if s.endswith("s") and not s.endswith("ss") else s


def extract_outputs(result: Any, choice: Any = None, prefix: str = "") -> dict[str, Any]:
    """Flatten a tool result into name->value pairs the next tool can bind.
    Lists of objects reduce to the chosen item; an item's 'id' is also exposed
    as '<singular list name>_id' ({"trains": [{"id": ..}]} -> train_id)."""
    if isinstance(result, list):
        item = pick_item(result, choice)
        return extract_outputs(item, None, prefix) if item is not None else {}
    out: dict[str, Any] = {}
    if not isinstance(result, Mapping):
        if result is not None and prefix:
            out[prefix] = result
        return out
    nested: list[tuple[str, Any]] = []
    for k, v in result.items():
        if isinstance(v, (Mapping, list)):
            nested.append((str(k), v))
        else:
            out[str(k)] = v
            if str(k).lower() == "id" and prefix:
                out.setdefault(f"{_singular(prefix)}_id", v)
    for k, v in nested:
        if isinstance(v, list) and v and not isinstance(v[0], (Mapping, list)):
            out.setdefault(k, v)  # list of scalars is a value, not a choice
            continue
        for sk, sv in extract_outputs(v, choice, k).items():
            out.setdefault(sk, sv)
    return out


_REF = re.compile(r"^\$(prev|slots|steps\.([\w-]+))(?:\.(.+))?$")


def _dig(obj: Any, path: str | None) -> Any:
    if not path:
        return obj
    for part in path.split("."):
        if isinstance(obj, Mapping):
            obj = obj.get(part)
        elif isinstance(obj, list) and re.fullmatch(r"-?\d+", part):
            i = int(part)
            obj = obj[i] if -len(obj) <= i < len(obj) else None
        else:
            return None
    return obj


def resolve_refs(args: Mapping[str, Any], prev: Any, steps: Mapping[str, Any],
                 slots: Mapping[str, Any]) -> dict[str, Any]:
    """'$prev.trains.0.id', '$steps.search.total', '$slots.name' -> values.
    Unresolvable refs are dropped so binding can report them as missing."""
    out: dict[str, Any] = {}
    for k, v in args.items():
        m = _REF.match(v) if isinstance(v, str) else None
        if not m:
            out[k] = v
            continue
        root = prev if m.group(1) == "prev" else slots if m.group(1) == "slots" else steps.get(m.group(2))
        val = _dig(root, m.group(3))
        if val is not None:
            out[k] = val
    return out


@dataclass
class Step:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)  # literals or $refs; the rest is bound automatically
    name: str | None = None
    choice: Any = None  # item of the previous result list to use; default slots['choice']


# =============================================================================
# Retry policy
# =============================================================================

@dataclass(frozen=True)
class RetryPolicy:
    """ASSUMPTION: values tuned for the 120 s cap, not given by the guide.
    Worst case per call: 3 attempts x 8 s + 0.25 + 0.5 s backoff ~ 25 s."""
    max_retries: int = MAX_RETRIES
    base_delay_s: float = 0.25
    factor: float = 2.0
    max_delay_s: float = 2.0
    call_timeout_s: float = 8.0  # per attempt, on the running loop's (virtual) clock
    retry_on_not_executed: bool = True  # ASSUMPTION: explicit "not executed" error => no side effect happened

    def delay(self, attempt: int) -> float:
        """Delay after failed attempt `attempt`. Deterministic (no jitter) for replay."""
        return min(self.max_delay_s, self.base_delay_s * self.factor ** max(0, attempt - 1))


_PERMANENT = re.compile(r"\b(invalid|not[ _-]?found|unknown|missing|required|forbidden|unauthori[sz]ed|"
                        r"denied|bad[ _-]?request|validation|sold[ _-]?out|no[ _-]?(?:seats|availability)|"
                        r"400|401|403|404|409|422)\b", re.I)
_NOT_EXECUTED = re.compile(r"not[ _-]?(?:executed|applied|processed|committed)|no changes? (?:were|was) made|"
                           r"rolled[ _-]?back|before execution", re.I)
_TRANSIENT = re.compile(r"time[ _-]?d?[ _-]?out|timeout|unavailable|temporar|try again|retry|rate[ _-]?limit|"
                        r"overload|busy|connection|reset|network|throttl|\b(?:429|500|502|503|504)\b", re.I)


def classify_error(res: ToolResult | None, timed_out: bool = False) -> str:
    """'permanent' | 'not_executed' | 'transient' | 'unknown'. ASSUMPTION: keyword rules."""
    if timed_out:
        return "transient"
    err = (res.error if res else "") or ""
    if _PERMANENT.search(err):
        return "permanent"
    if _NOT_EXECUTED.search(err):
        return "not_executed"
    if (res is not None and res.retryable) or _TRANSIENT.search(err):
        return "transient"
    return "unknown"


@dataclass(frozen=True)
class RetryDecision:
    retry: bool
    delay_s: float
    reason: str
    side_effect_unknown: bool = False  # a write may or may not have happened


def idempotency_guaranteed(tool: ToolDef, args: Mapping[str, Any]) -> bool:
    return tool.read_only or tool.idempotent or bool(tool.idempotency_param and args.get(tool.idempotency_param))


def decide_retry(tool: ToolDef, args: Mapping[str, Any], err_class: str, attempt: int,
                 policy: RetryPolicy) -> RetryDecision:
    """Read-only: retry anything not clearly permanent. State-modifying: retry
    only when a repeat cannot double the effect (idempotent tool, dedup key, or
    an explicit not-executed error); otherwise stop and report honestly."""
    maybe_happened = tool.state_modifying and err_class in ("transient", "unknown")
    if err_class == "permanent":
        return RetryDecision(False, 0.0, "permanent error")
    if attempt > policy.max_retries:
        return RetryDecision(False, 0.0, "retries exhausted", side_effect_unknown=maybe_happened)
    if tool.read_only:
        return RetryDecision(True, policy.delay(attempt), f"read-only, {err_class}")
    if idempotency_guaranteed(tool, args):
        return RetryDecision(True, policy.delay(attempt), f"idempotent write, {err_class}")
    if err_class == "not_executed" and policy.retry_on_not_executed:
        return RetryDecision(True, policy.delay(attempt), "write reported not executed")
    return RetryDecision(False, 0.0, "write without idempotency guarantee", side_effect_unknown=maybe_happened)


def _what(tool: ToolDef) -> str:
    w = tool.description.strip().rstrip(".") or tool.name.replace("_", " ")
    return w[0].lower() + w[1:]


def failure_note(tool: ToolDef, err: str | None, uncertain: bool, attempts: int) -> str:
    """Honest user-facing sentence. Never claims success, never hides doubt."""
    reason = f" ({err})" if err else ""
    tries = f" after {attempts} tries" if attempts > 1 else ""
    if tool.read_only:
        return f"I couldn't {_what(tool)}{tries}{reason}. Nothing was changed."
    if uncertain:
        return (f"The request to {_what(tool)} didn't return a clear answer{reason}. I can't confirm whether it "
                f"went through, so I haven't repeated it to avoid doing it twice.")
    return f"The request to {_what(tool)} failed{tries}{reason}, so nothing was changed."


# =============================================================================
# Call tracking and idempotency (interfaces shared with coordinator.py)
# =============================================================================

@dataclass
class Call:
    call_id: str
    tool: str
    args: dict[str, Any]
    state_modifying: bool
    purpose: str  # "goal" or "prereq"
    intent: str | None
    attempt: int = 1
    key: str = ""

    def __post_init__(self) -> None:
        self.key = f"{self.tool}|{canonical_args(self.args)}"


@dataclass
class CallTracker:
    """In-flight calls plus the ids we have cancelled (their results are stale)."""
    inflight: dict[str, Call] = field(default_factory=dict)
    cancelled: set[str] = field(default_factory=set)
    finished: dict[str, Call] = field(default_factory=dict)
    ids: IdGen = field(default_factory=IdGen)

    def start(self, tool: ToolSpec | ToolDef, args: dict[str, Any], purpose: str, intent: str | None,
              attempt: int = 1) -> Call:
        c = Call(self.ids("call"), tool.name, dict(args), tool.state_modifying, purpose, intent, attempt)
        self.inflight[c.call_id] = c
        return c

    def cancel(self, call_id: str) -> Call | None:
        c = self.inflight.pop(call_id, None)
        if c is not None:
            self.cancelled.add(call_id)
        return c

    def finish(self, call_id: str) -> Call | None:
        c = self.inflight.pop(call_id, None)
        if c is not None:
            self.finished[call_id] = c
        return c

    def is_stale(self, call_id: str) -> bool:
        return call_id in self.cancelled or call_id not in self.inflight

    def find(self, key: str) -> Call | None:
        return next((c for c in self.inflight.values() if c.key == key), None)


@dataclass
class IdempotencyLedger:
    """Zero duplicate state-changing calls: a write key is issued at most once
    unless its previous attempt definitively failed or was cancelled."""
    committed: dict[str, Any] = field(default_factory=dict)  # key -> result
    pending: set[str] = field(default_factory=set)

    def may_issue(self, key: str) -> bool:
        return key not in self.committed and key not in self.pending

    def issued(self, key: str) -> None:
        self.pending.add(key)

    def commit(self, key: str, result: Any) -> None:
        self.pending.discard(key)
        self.committed[key] = result

    def release(self, key: str) -> None:
        """Attempt failed or cancelled before commit: key may be issued again."""
        self.pending.discard(key)


def idempotency_token(tool: ToolDef, args: Mapping[str, Any]) -> str:
    """Deterministic dedup key: same logical request -> same token, across retries."""
    base = {k: v for k, v in args.items() if k != tool.idempotency_param}
    return "idem-" + hashlib.sha1(f"{tool.name}|{canonical_args(base)}".encode()).hexdigest()[:16]


# =============================================================================
# Runner: bind -> validate -> dedupe -> emit -> await result -> retry -> outcome
# =============================================================================

OK = "ok"
NEEDS_INPUT = "needs_input"  # missing/unusable args: ask `question`, nothing was called
INVALID = "invalid"  # bound args failed schema validation, nothing was called
FAILED = "failed"  # definitive failure, nothing changed
UNCERTAIN = "uncertain"  # a write may or may not have happened; not retried
IN_FLIGHT = "in_flight"  # identical write already pending: not issued again
UNKNOWN_TOOL = "unknown_tool"


@dataclass
class Outcome:
    status: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    attempts: int = 0
    call_ids: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    invalid: dict[str, str] = field(default_factory=dict)
    question: str | None = None  # clarification to ask the user
    note: str | None = None  # honest sentence for the user when not OK
    deduped: bool = False  # result reused from an earlier identical write

    @property
    def ok(self) -> bool:
        return self.status == OK


@dataclass
class ChainOutcome:
    outcomes: list[Outcome]
    results: dict[str, Any]  # step name -> result

    @property
    def ok(self) -> bool:
        return bool(self.outcomes) and all(o.ok for o in self.outcomes)

    @property
    def last(self) -> Outcome | None:
        return self.outcomes[-1] if self.outcomes else None


class ToolRunner:
    """Non-blocking tool execution on top of the action/event queues.

    `emit_call(call)` / `emit_cancel(call)` turn calls into protocol actions
    (the coordinator adds timestamps and snapshots). Results come back through
    `deliver()` when a tool_result event arrives. All waits use the running
    loop's clock (virtual under sim.vloop) and the injectable `sleep`.
    Run `call()`/`chain()` as tasks: cancelling the task emits a cancel for
    the in-flight call before the CancelledError propagates.
    """

    def __init__(self, registry: ToolRegistry, emit_call: Callable[[Call], Any],
                 emit_cancel: Callable[[Call], Any] | None = None, *, policy: RetryPolicy | None = None,
                 tracker: CallTracker | None = None, ledger: IdempotencyLedger | None = None,
                 sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
                 on_retry: Callable[[Call, RetryDecision], Any] | None = None) -> None:
        self.registry = registry
        self.emit_call = emit_call
        self.emit_cancel = emit_cancel or (lambda c: None)
        self.policy = policy or RetryPolicy()
        self.tracker = tracker or CallTracker()
        self.ledger = ledger or IdempotencyLedger()
        self.sleep = sleep
        self.on_retry = on_retry
        self._futures: dict[str, asyncio.Future[ToolResult]] = {}
        self.stale_results: list[ToolResult] = []

    # -- results ---------------------------------------------------------------
    def deliver(self, res: ToolResult) -> bool:
        """Route a tool result. False if stale (cancelled, unknown or duplicate id)."""
        fut = self._futures.get(res.call_id)
        if fut is None or fut.done() or self.tracker.is_stale(res.call_id):
            self.stale_results.append(res)
            return False
        fut.set_result(res)
        return True

    def cancel(self, call_id: str) -> bool:
        c = self.tracker.cancel(call_id)
        if c is None:
            return False
        self.emit_cancel(c)
        fut = self._futures.get(call_id)
        if fut is not None and not fut.done():
            fut.cancel()
        return True

    def cancel_all(self) -> list[str]:
        ids = list(self.tracker.inflight)
        for cid in ids:
            self.cancel(cid)
        return ids

    # -- one call --------------------------------------------------------------
    def prepare(self, tool_name: str, args: Mapping[str, Any] | None = None, *,
                slots: Mapping[str, Any] | None = None,
                context: Mapping[str, Any] | None = None) -> tuple[ToolDef | None, Outcome | None, dict[str, Any]]:
        """Bind and validate without calling: (tool, early outcome or None, call args).
        Synchronous, so the fast path can use it to decide whether to clarify."""
        tool = self.registry.get(tool_name)
        if tool is None:
            return None, Outcome(UNKNOWN_TOOL, str(tool_name), note=f"No tool named {tool_name} is available."), {}
        b = bind_args(tool, slots, context, args)
        if not b.complete:
            return tool, Outcome(NEEDS_INPUT, tool.name, dict(b.args), missing=b.missing, invalid=b.invalid,
                                 question=clarification(tool, b.missing, b.invalid)), b.args
        call_args = dict(b.args)
        if tool.idempotency_param and not call_args.get(tool.idempotency_param):
            call_args[tool.idempotency_param] = idempotency_token(tool, call_args)
        errs = validate_args(tool, call_args)
        if errs:
            return tool, Outcome(INVALID, tool.name, call_args, error="; ".join(errs),
                                 note="Some details don't fit what this service accepts: " + "; ".join(errs)), call_args
        return tool, None, call_args

    async def call(self, tool_name: str, args: Mapping[str, Any] | None = None, *,
                   slots: Mapping[str, Any] | None = None, context: Mapping[str, Any] | None = None,
                   purpose: str = "goal", intent: str | None = None) -> Outcome:
        tool, early, call_args = self.prepare(tool_name, args, slots=slots, context=context)
        if early is not None or tool is None:
            return early  # type: ignore[return-value]
        key = f"{tool.name}|{canonical_args(call_args)}"
        if tool.state_modifying:
            if key in self.ledger.committed:  # duplicate-booking trap: reuse, never re-issue
                return Outcome(OK, tool.name, call_args, result=self.ledger.committed[key], deduped=True)
            if key in self.ledger.pending:
                return Outcome(IN_FLIGHT, tool.name, call_args,
                               note="That request is already in progress, so I haven't sent it again.")
            self.ledger.issued(key)
        out = Outcome(FAILED, tool.name, call_args)
        loop = asyncio.get_running_loop()
        attempt = 1
        while True:
            c = self.tracker.start(tool, call_args, purpose, intent, attempt)
            out.call_ids.append(c.call_id)
            out.attempts = attempt
            fut: asyncio.Future[ToolResult] = loop.create_future()
            self._futures[c.call_id] = fut
            self.emit_call(c)
            res: ToolResult | None = None
            timed_out = False
            try:
                res = await asyncio.wait_for(asyncio.shield(fut), self.policy.call_timeout_s)
            except asyncio.TimeoutError:
                timed_out = True
                self.cancel(c.call_id)  # stop the orphan; its late result is stale
            except asyncio.CancelledError:
                if fut.cancelled() and c.call_id in self.tracker.cancelled and not self._task_cancelling():
                    # cancelled via runner.cancel(): report as a failed attempt, no retry
                    self._futures.pop(c.call_id, None)
                    if tool.state_modifying:
                        self.ledger.release(key)
                    out.status, out.error = FAILED, "cancelled"
                    return out
                # Task superseded (interruption): cancel first, then free the key (SPEC U10/U15).
                if c.call_id in self.tracker.inflight:
                    self.cancel(c.call_id)
                if tool.state_modifying:
                    self.ledger.release(key)
                raise
            finally:
                self._futures.pop(c.call_id, None)
            if res is not None:
                self.tracker.finish(c.call_id)
                if res.ok:
                    if tool.state_modifying:
                        self.ledger.commit(key, res.result)
                    out.status, out.result, out.error = OK, res.result, None
                    return out
            out.error = "timed out" if timed_out else (res.error if res else None) or "error"
            d = decide_retry(tool, call_args, classify_error(res, timed_out), attempt, self.policy)
            if d.retry:
                if self.on_retry is not None:
                    self.on_retry(c, d)
                await self.sleep(d.delay_s)
                attempt += 1
                continue
            if tool.state_modifying and d.side_effect_unknown and not idempotency_guaranteed(tool, call_args):
                out.status = UNCERTAIN  # key stays pending: never auto-reissued
            else:
                if tool.state_modifying:
                    self.ledger.release(key)
                out.status = FAILED
            out.note = failure_note(tool, out.error, out.status == UNCERTAIN, attempt)
            return out

    @staticmethod
    def _task_cancelling() -> bool:
        t = asyncio.current_task()
        return bool(t is not None and getattr(t, "cancelling", lambda: 0)())

    # -- chained calls ---------------------------------------------------------
    async def chain(self, steps: Iterable[Step], slots: Mapping[str, Any] | None = None,
                    intent: str | None = None) -> ChainOutcome:
        """Run steps in order. Each step binds from explicit args/$refs, then
        user slots, then the previous step's outputs. Stops at the first step
        that is not OK, so a clarification or failure is surfaced, never skipped."""
        slots = dict(slots or {})
        results: dict[str, Any] = {}
        outcomes: list[Outcome] = []
        prev: Any = None
        steps = list(steps)
        for i, st in enumerate(steps):
            explicit = resolve_refs(st.args, prev, results, slots)
            choice = st.choice if st.choice is not None else slots.get("choice")
            ctx = extract_outputs(prev, choice) if prev is not None else None
            o = await self.call(st.tool, explicit or None, slots=slots, context=ctx,
                                purpose="goal" if i == len(steps) - 1 else "prereq", intent=intent)
            outcomes.append(o)
            if not o.ok:
                break
            results[st.name or st.tool] = o.result
            prev = o.result
        return ChainOutcome(outcomes, results)
