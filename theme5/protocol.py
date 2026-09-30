# /mnt/project-files/theme5/theme5/protocol.py
"""Wire protocol for the Theme 05 agent.

The guide (v1.0.0) fixes only the *shape* of the contract:
  - two asynchronous queues: timestamped events in, actions out
  - inputs: text chunks with end-of-turn markers, WAV clips, PNG frames,
    interruption signals, async tool results, scenario tool manifests
  - outputs: spoken fillers, non-blocking tool calls with explicit call_id,
    cancellations, clarification requests, final responses carrying a
    State Snapshot (intent + slot values)
  - tools are read-only or state-modifying, declared in manifests
  - well-formed JSON payloads with valid snapshots and identifiers

Every concrete key name, type string, unit and default below is an ASSUMPTION
until the evaluation kit ships (see docs/SPEC.md, U01-U26). When it does,
only this file (and its companion protocol_media.py)
should change. Core modules use the internal names defined here and never
touch raw kit field names.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Mapping

# --------------------------------------------------------------------------
# Execution constraints (from the guide, G§6)
# --------------------------------------------------------------------------
SCENARIO_WALL_CAP_S = 120.0  # 120 s wall-clock cap per scenario
WARMUP_CAP_S = 300.0  # 300 s setup/warm-up hook

# --------------------------------------------------------------------------
# Tunables. ASSUMPTION: values chosen for the rubric, not given by the guide.
# --------------------------------------------------------------------------
MAX_RETRIES = 2  # U14: retries per failed call (writes only when retryable)
ASR_MIN_CONFIDENCE = 0.6  # U19: below this we confirm instead of act
LLM_TIMEOUT_S = 2.0  # slow-path LLM budget before rule fallback
ACK_ON_INTERRUPT = False  # U08: a bare interrupt holds state and waits for words
DEFAULT_CHOICE = 1  # with several options and no stated preference, take the first
CANCEL_PENDING_ON_END = False  # U23: whether to cancel in-flight calls at session end
SNAPSHOT_INTENT = "tool"  # U17: "tool" = goal tool name, "internal" = our intent label
SNAPSHOT_SLOTS = "tool_params"  # U16: "tool_params" = goal tool param names, "internal"
TS_UNIT = "ms"  # U05: unit of numeric timestamps when not ISO/epoch ("ms" or "s")
EMIT_CALL_META = False  # U12: put generation/idempotency_key on the wire (off: schema-only fields)
SPECULATE_READ_ONLY = True  # start read-only calls on partial turns when slots suffice (SPEC §5)
TOOL_TIMEOUT_MS: float | None = None  # U14: agent-side call timeout on the event clock (None: trust the kit)
# U06 ASSUMPTION: the 120 s cap counts from session start (our run() entry). At this
# point the watchdog closes any hanging turn with a safe final_response (watchdog.py).
WATCHDOG_AT_S = 110.0

# --------------------------------------------------------------------------
# Event types (internal). ASSUMPTION: inbound spellings in _EVENT_ALIASES (U03).
# --------------------------------------------------------------------------
EV_MANIFEST = "tool_manifest"
EV_TEXT = "text"
EV_EOT = "end_of_turn"
EV_AUDIO = "audio"
EV_FRAME = "frame"
EV_INTERRUPT = "interrupt"
EV_TOOL_RESULT = "tool_result"
EV_END = "end_session"
EV_UNKNOWN = "unknown"

_EVENT_ALIASES: dict[str, str] = {
    "toolmanifest": EV_MANIFEST, "manifest": EV_MANIFEST, "tools": EV_MANIFEST, "scenariotools": EV_MANIFEST,
    "text": EV_TEXT, "transcript": EV_TEXT, "textchunk": EV_TEXT, "usertext": EV_TEXT, "asrpartial": EV_TEXT,
    "asrfinal": EV_TEXT, "partialtranscript": EV_TEXT, "userspeech": EV_TEXT, "utterance": EV_TEXT,
    "endofturn": EV_EOT, "eot": EV_EOT, "turnend": EV_EOT, "userturnend": EV_EOT,
    "audio": EV_AUDIO, "audioclip": EV_AUDIO, "wav": EV_AUDIO, "audiochunk": EV_AUDIO,
    "frame": EV_FRAME, "videoframe": EV_FRAME, "image": EV_FRAME, "png": EV_FRAME,
    "interrupt": EV_INTERRUPT, "interruption": EV_INTERRUPT, "bargein": EV_INTERRUPT, "userinterrupt": EV_INTERRUPT,
    "toolresult": EV_TOOL_RESULT, "toolresponse": EV_TOOL_RESULT, "result": EV_TOOL_RESULT,
    "toolerror": EV_TOOL_RESULT, "toolfault": EV_TOOL_RESULT,
    "endsession": EV_END, "end": EV_END, "sessionend": EV_END, "eos": EV_END, "scenarioend": EV_END,
}
EOT_TOKENS = ("<EOT>", "<eot>", "[EOT]", "<end_of_turn>")

# --------------------------------------------------------------------------
# Action types (internal). ASSUMPTION: outbound spellings (U18).
# --------------------------------------------------------------------------
ACT_SPEAK = "speak"
ACT_TOOL_CALL = "tool_call"
ACT_CANCEL = "cancel"
ACT_CLARIFY = "clarify"
ACT_FINAL = "final_response"
ACTION_TYPES = (ACT_SPEAK, ACT_TOOL_CALL, ACT_CANCEL, ACT_CLARIFY, ACT_FINAL)

SPEAK_FILLER = "filler"  # "one moment": not substantive (U21)
SPEAK_ACK = "ack"  # acknowledges what was understood
SPEAK_PROGRESS = "progress"  # narrates a launched tool call
SPEAK_INFO = "info"  # substantive statement that is not a final response
SPEAK_KINDS = (SPEAK_FILLER, SPEAK_ACK, SPEAK_PROGRESS, SPEAK_INFO)

# Internal field -> wire field (U12). Only renamed fields are listed.
WIRE_FIELDS: dict[str, str] = {"tool": "name", "args": "arguments"}
WIRE_TYPES: dict[str, str] = {}  # internal action type -> wire type, e.g. {"speak": "say"}


@dataclass(frozen=True)
class Event:
    type: str
    t: float  # internal: milliseconds on the harness clock
    payload: Mapping[str, Any] = field(default_factory=dict)
    id: str | None = None
    raw_type: str = ""


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str = "string"
    required: bool = False
    enum: tuple[Any, ...] | None = None
    description: str = ""
    pattern: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: tuple[ParamSpec, ...]
    state_modifying: bool

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params if p.required)

    def param(self, name: str) -> ParamSpec | None:
        return next((p for p in self.params if p.name == name), None)


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    ok: bool
    result: Any = None
    error: str | None = None
    retryable: bool = False
    outcome_unknown: bool = False  # timed out: the tool may still have executed (U14)


def _first(d: Mapping[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _squash(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


# ---- timestamps (U05) -------------------------------------------------------

_TS_KEYS = ("t", "ts", "timestamp", "time", "time_ms", "t_ms")


def to_ms(v: Any, unit: str = TS_UNIT) -> float:
    """Normalise a timestamp to internal milliseconds. ISO-8601 -> epoch ms;
    numbers >= 1e11 are taken as epoch ms; otherwise `unit` applies.
    Missing, unparseable or non-finite values give 0.0 (never raises): the
    event clock only moves forward, so such an event keeps the current time."""
    try:
        if v is None or isinstance(v, bool):
            return 0.0
        if isinstance(v, str):
            try:
                f = float(v) * (1000.0 if unit == "s" else 1.0)
            except ValueError:
                f = datetime.fromisoformat(v.strip().replace("Z", "+00:00")).timestamp() * 1000.0
        else:
            f = float(v)
            if f < 1e11:
                f = f * 1000.0 if unit == "s" else f
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return f if math.isfinite(f) else 0.0


@dataclass
class WireFormat:
    """Per-session memory of how the harness spells timestamps, so outbound
    actions echo the same key and unit (U05). One instance per Agent (U26)."""
    ts_key: str = "t"
    unit: str = TS_UNIT

    def observe(self, raw: Mapping[str, Any]) -> None:
        k = next((k for k in _TS_KEYS if k in raw), None)
        if k is not None:
            self.ts_key = k
            if k in ("time_ms", "t_ms"):
                self.unit = "ms"

    def out_time(self, ms: float) -> float:
        return round(ms / 1000.0, 6) if self.unit == "s" else round(ms, 3)


# ---- events -----------------------------------------------------------------

def parse_event(raw: Mapping[str, Any] | str, unit: str = TS_UNIT) -> Event:
    """Tolerant event parser (U03, U04). Unknown types map to EV_UNKNOWN and
    are ignored by the agent, never raised. So does anything that is not a
    JSON object (bad JSON text, lists, numbers, bytes)."""
    try:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "replace")
        if isinstance(raw, str):
            raw = json.loads(raw)
    except ValueError:
        return Event(type=EV_UNKNOWN, t=0.0, payload={"malformed": str(raw)[:200]}, raw_type="malformed")
    if not isinstance(raw, Mapping):
        return Event(type=EV_UNKNOWN, t=0.0, payload={"malformed": repr(raw)[:200]},
                     raw_type=f"malformed:{type(raw).__name__}")
    raw_type = str(_first(raw, ("type", "event", "kind", "event_type"), EV_UNKNOWN))
    typ = _EVENT_ALIASES.get(_squash(raw_type), EV_UNKNOWN)
    t = to_ms(_first(raw, _TS_KEYS), unit)
    meta = ("type", "event", "kind", "event_type", "id", "event_id", "payload", "data") + _TS_KEYS
    payload: dict[str, Any] = {k: v for k, v in raw.items() if k not in meta}
    for nest in ("payload", "data"):
        inner = raw.get(nest)
        if isinstance(inner, Mapping):
            payload.update(inner)
        elif inner is not None:
            payload.setdefault(nest, inner)
    eid = _first(raw, ("id", "event_id"))
    return Event(type=typ, t=t, payload=payload, id=None if eid is None else str(eid), raw_type=raw_type)


# ---- event payload accessors (ASSUMPTION: key names) ------------------------

_TEXT_KEYS = ("text", "chunk", "transcript", "delta", "content")


def text_of(ev: Event) -> str:
    txt = str(_first(ev.payload, _TEXT_KEYS, ""))
    for tok in EOT_TOKENS:
        txt = txt.replace(tok, "")
    return txt.strip()


def is_end_of_turn(ev: Event) -> bool:
    if ev.type == EV_EOT:
        return True
    if bool(_first(ev.payload, ("end_of_turn", "eot", "final", "is_final", "endOfTurn", "turn_complete"), False)):
        return True
    raw = str(_first(ev.payload, _TEXT_KEYS, ""))
    return any(tok in raw for tok in EOT_TOKENS)


def is_cumulative(ev: Event) -> bool | None:
    """U07: explicit hint if the harness gives one, else None (infer by prefix)."""
    v = _first(ev.payload, ("cumulative", "is_cumulative"))
    if v is not None:
        return bool(v)
    if "delta" in ev.payload:
        return False
    return None


def merge_chunk(buffer: str, chunk: str, cumulative: bool | None = None) -> str:
    """Incremental chunks are appended; cumulative ones replace the buffer."""
    if not chunk:
        return buffer
    if cumulative is True or (cumulative is None and buffer and chunk.startswith(buffer)):
        return chunk
    return f"{buffer} {chunk}".strip() if buffer else chunk


def media_ref(ev: Event) -> str | None:
    ref = _first(ev.payload, ("frame_id", "audio_id", "clip_id", "image_id", "path", "uri", "url", "file"))
    if ref is None:
        ref = ev.id
    return None if ref is None else str(ref)


def audio_transcript(ev: Event) -> tuple[str | None, float]:
    """U19: if the harness ships a transcript with the clip, prefer it."""
    txt = _first(ev.payload, ("transcript", "text", "asr", "reference_text"))
    conf = float(_first(ev.payload, ("confidence", "asr_confidence"), 1.0))
    return (None if txt is None else str(txt)), conf


def frame_labels(ev: Event) -> list[dict[str, Any]]:
    """U20: normalise label lists to [{"label", "model", "text", "confidence"}].
    Accepts ["tv", ...] or [{"label"|"name", "model", "text", "confidence"|"score"}]."""
    raw = _first(ev.payload, ("labels", "objects", "detections"))
    out: list[dict[str, Any]] = []
    for item in raw if isinstance(raw, (list, tuple)) else []:
        if isinstance(item, Mapping):
            out.append({
                "label": str(_first(item, ("label", "name", "class"), "")).replace("_", " "),
                "model": _first(item, ("model", "model_number")),
                "text": _first(item, ("text", "ocr")),
                "confidence": None if _first(item, ("confidence", "score", "p")) is None
                else float(_first(item, ("confidence", "score", "p"))),
            })
        else:
            out.append({"label": str(item).replace("_", " "), "model": None, "text": None, "confidence": None})
    return out


def frame_caption(ev: Event) -> str | None:
    """U20: if the harness ships a caption/label list with the frame, use it."""
    labels = frame_labels(ev)
    if labels:
        return ", ".join(" ".join(str(x) for x in (d["label"], d["model"], d["text"]) if x) for d in labels)
    cap = _first(ev.payload, ("caption", "description", "ocr"))
    if isinstance(cap, (list, tuple)):
        return ", ".join(map(str, cap))
    return None if cap is None else str(cap)


_RETRYABLE_CODES = ("timeout", "unavailable", "503", "429", "not_executed", "transient")


def parse_tool_result(ev: Event) -> ToolResult:
    """U14: result carries call_id plus either a result or an error."""
    p = ev.payload
    err = _first(p, ("error", "err", "fault", "exception"))
    status = str(_first(p, ("status",), "")).lower()
    ok_raw = _first(p, ("ok", "success"))
    ok = bool(ok_raw) if ok_raw is not None else (
        err is None and status not in ("error", "failed", "failure", "timeout", "fault"))
    retryable = bool(_first(p, ("retryable", "transient", "not_executed"), False)) or status == "timeout"
    if isinstance(err, Mapping):
        retryable = retryable or bool(_first(err, ("retryable", "transient", "not_executed"), False))
        retryable = retryable or str(_first(err, ("code", "type"), "")).lower() in _RETRYABLE_CODES
        err = str(_first(err, ("message", "code", "type"), err))
    elif isinstance(err, str) and err.lower() in _RETRYABLE_CODES:
        retryable = True
    if _squash(ev.raw_type) in ("toolerror", "toolfault"):
        ok = False
    # ASSUMPTION (U14): a timeout says nothing about whether the tool ran, so a
    # state-modifying call that timed out is never retried with the same args.
    unknown = not ok and (status == "timeout" or bool(re.search(r"time[d ]?\s*out", str(err or ""), re.I)))
    return ToolResult(
        call_id=str(_first(p, ("call_id", "callId", "tool_call_id"), ev.id or "")),
        ok=ok,
        result=_first(p, ("result", "output", "value", "response")),
        error=None if err is None else str(err),
        retryable=retryable,
        outcome_unknown=unknown,
    )


# ---- manifest (U11) -----------------------------------------------------------
# Merged from protocol_manifest.py (tools thread). Every key name and marker
# spelling below is an ASSUMPTION: guide v1.0.0 only says manifests declare
# read-only vs state-modifying tools.
# Accepted shapes: flat {"name", "description", "parameters", <flags>}; OpenAI
# {"type": "function", "function": {...}}; MCP {"name", "inputSchema",
# "annotations": {"readOnlyHint", ...}}; Anthropic {"name", "input_schema"}.
# Schema = JSON Schema object | {param: {type, required, ...}} | [{name, type, ...}].
# Payload = list | {"tools"|"manifest"|"functions": [...]} | {tool_name: def}.
# Safety rule (U11): read-only only if some marker says so and none says
# otherwise; no marker or conflicting markers -> state-modifying.

_NO_DEFAULT = object()

# ASSUMPTION: marker spellings.
_READ_BOOL_KEYS = ("read_only", "readonly", "readOnly", "is_read_only", "readOnlyHint", "safe")
_WRITE_BOOL_KEYS = ("state_modifying", "stateModifying", "mutating", "side_effects", "sideEffects",
                    "side_effect", "writes", "is_write", "destructive", "destructiveHint", "modifies_state")
_KIND_KEYS = ("kind", "category", "effect", "access", "mode", "tool_type", "operation")
_READ_WORDS = {"read", "read_only", "readonly", "read-only", "query", "safe", "get", "lookup", "search", "fetch"}
_WRITE_WORDS = {"write", "state_modifying", "state-modifying", "mutating", "modify", "side_effect",
                "side_effects", "action", "create", "update", "delete", "mutation", "command"}
_IDEMPOTENT_KEYS = ("idempotent", "is_idempotent", "idempotentHint", "idempotency")
_IDEM_PARAM_NAMES = {"idempotencykey", "idempotency", "idemkey", "idemtoken", "requestid", "clienttoken",
                     "clientrequestid", "dedupkey", "dedupekey", "dedupid", "dedupeid", "operationid",
                     "requesttoken", "nonce"}
_SCHEMA_KEYS = ("parameters", "params", "input_schema", "inputSchema", "args", "arguments", "schema")
_OUTPUT_KEYS = ("output_schema", "outputSchema", "returns", "response_schema", "result_schema", "output")
_TYPE_ALIASES = {
    "str": "string", "text": "string", "string": "string",
    "int": "integer", "integer": "integer", "long": "integer",
    "float": "number", "double": "number", "number": "number", "decimal": "number",
    "bool": "boolean", "boolean": "boolean",
    "list": "array", "array": "array", "tuple": "array",
    "dict": "object", "object": "object", "map": "object",
    "date": "string", "datetime": "string", "date-time": "string", "time": "string",
    "any": "any", "null": "any",
}


@dataclass(frozen=True)
class ParamDef:
    name: str
    type: str = "string"  # string|integer|number|boolean|array|object|any
    required: bool = False
    enum: tuple[Any, ...] | None = None
    description: str = ""
    default: Any = _NO_DEFAULT
    format: str | None = None  # e.g. "date", "date-time", "email"
    items_type: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    aliases: tuple[str, ...] = ()

    @property
    def has_default(self) -> bool:
        return self.default is not _NO_DEFAULT


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    params: tuple[ParamDef, ...]
    read_only: bool
    idempotent: bool = False  # manifest guarantees repeat calls are safe
    idempotency_param: str | None = None  # param carrying a dedup key
    output_fields: tuple[str, ...] = ()
    decided_by: str = "default"  # which marker set read_only (for traces)

    @property
    def state_modifying(self) -> bool:
        return not self.read_only

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params if p.required)

    def param(self, name: str) -> ParamDef | None:
        return next((p for p in self.params if p.name == name), None)

    @property
    def spec(self) -> ToolSpec:
        """Down-convert to protocol.ToolSpec for code that only needs the basics."""
        return ToolSpec(
            name=self.name, description=self.description, state_modifying=self.state_modifying,
            params=tuple(ParamSpec(p.name, p.type, p.required, p.enum, p.description, p.pattern) for p in self.params),
        )


def from_toolspec(t: ToolSpec) -> ToolDef:
    return ToolDef(
        name=t.name, description=t.description, read_only=not t.state_modifying, decided_by="toolspec",
        params=tuple(ParamDef(p.name, _norm_type(p.type)[0], p.required, p.enum, p.description) for p in t.params),
    )


def _flag(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "1", "y"):
            return True
        if s in ("false", "no", "0", "n", "none"):
            return False
    return None


def _norm_type(t: Any) -> tuple[str, str | None]:
    """Returns (json type, implied format)."""
    if isinstance(t, (list, tuple)):
        t = next((x for x in t if str(x).lower() != "null"), "any")
    s = str(t or "string").strip().lower()
    fmt = s if s in ("date", "datetime", "date-time", "time") else None
    if fmt == "datetime":
        fmt = "date-time"
    return _TYPE_ALIASES.get(s, "any"), fmt


def _num(v: Any) -> float | None:
    try:
        return None if v is None or isinstance(v, bool) else float(v)
    except (TypeError, ValueError):
        return None


def _int(v: Any) -> int | None:
    n = _num(v)
    return None if n is None else int(n)


def _param_def(name: str, ps: Any, required: bool) -> ParamDef:
    if not isinstance(ps, Mapping):
        ps = {"type": ps}
    typ, fmt = _norm_type(ps.get("type", "string"))
    enum = ps.get("enum", ps.get("choices", ps.get("options")))
    items = ps.get("items")
    items_type = _norm_type(items.get("type"))[0] if isinstance(items, Mapping) else None
    aliases = ps.get("aliases", ps.get("x-aliases", ps.get("synonyms", ())))
    req = _flag(ps.get("required"))
    return ParamDef(
        name=str(name), type=typ, required=required or bool(req),
        enum=tuple(enum) if isinstance(enum, (list, tuple)) and enum else None,
        description=str(ps.get("description", ps.get("desc", "")) or ""),
        default=ps["default"] if "default" in ps else _NO_DEFAULT,
        format=str(ps["format"]).lower() if ps.get("format") else fmt,
        items_type=items_type,
        minimum=_num(ps.get("minimum", ps.get("min"))), maximum=_num(ps.get("maximum", ps.get("max"))),
        min_length=_int(ps.get("minLength", ps.get("min_length"))),
        max_length=_int(ps.get("maxLength", ps.get("max_length"))),
        pattern=str(ps["pattern"]) if ps.get("pattern") else None,
        aliases=tuple(str(a) for a in aliases) if isinstance(aliases, (list, tuple)) else (),
    )


def parse_params(schema: Any) -> tuple[ParamDef, ...]:
    if isinstance(schema, str):
        import json
        try:
            schema = json.loads(schema)
        except ValueError:
            return ()
    out: list[ParamDef] = []
    if isinstance(schema, Mapping) and isinstance(schema.get("properties"), Mapping):
        req = schema.get("required", [])
        req = set(req) if isinstance(req, (list, tuple)) else set()
        for name, ps in schema["properties"].items():
            out.append(_param_def(name, ps or {}, name in req))
    elif isinstance(schema, Mapping) and "type" in schema and schema.get("type") == "object":
        pass  # object schema with no properties: no params
    elif isinstance(schema, Mapping):
        for name, ps in schema.items():
            out.append(_param_def(name, ps, False))
    elif isinstance(schema, (list, tuple)):
        for ps in schema:
            if isinstance(ps, Mapping) and ps.get("name"):
                out.append(_param_def(ps["name"], ps, False))
            elif isinstance(ps, str):
                out.append(_param_def(ps, {}, False))
    return tuple(out)


def _classify(t: Mapping[str, Any]) -> tuple[bool, str]:
    """Returns (read_only, deciding marker)."""
    sources: list[Mapping[str, Any]] = [t]
    ann = t.get("annotations") or t.get("hints") or t.get("meta")
    if isinstance(ann, Mapping):
        sources.append(ann)
    read_votes: list[str] = []
    write_votes: list[str] = []
    for src in sources:
        for k in _READ_BOOL_KEYS:
            if k in src:
                f = _flag(src[k])
                if f is True:
                    read_votes.append(k)
                elif f is False:
                    write_votes.append(k)
        for k in _WRITE_BOOL_KEYS:
            if k in src:
                v = src[k]
                f = _flag(v)
                if f is None and isinstance(v, str):  # e.g. "side_effects": "write"
                    f = v.strip().lower() not in _READ_WORDS
                if isinstance(v, (list, tuple)):  # e.g. "side_effects": ["db"]
                    f = len(v) > 0
                if f is True:
                    write_votes.append(k)
                elif f is False and k not in ("destructive", "destructiveHint"):
                    # destructive=false does not mean read-only (MCP semantics)
                    read_votes.append(k)
        for k in _KIND_KEYS:
            v = src.get(k)
            if isinstance(v, str):
                s = v.strip().lower()
                if s in _READ_WORDS:
                    read_votes.append(f"{k}={s}")
                elif s in _WRITE_WORDS:
                    write_votes.append(f"{k}={s}")
    if write_votes:
        return False, write_votes[0]
    if read_votes:
        return True, read_votes[0]
    return False, "default"  # ASSUMPTION (SPEC U11): unmarked -> state-modifying


def _idempotent(t: Mapping[str, Any]) -> bool:
    for src in (t, t.get("annotations") if isinstance(t.get("annotations"), Mapping) else {}):
        for k in _IDEMPOTENT_KEYS:
            if k in src and _flag(src[k]) is True:
                return True
    return False



def _idem_param(params: tuple[ParamDef, ...]) -> str | None:
    for p in params:
        if _squash(p.name) in _IDEM_PARAM_NAMES or "idempot" in p.description.lower():
            return p.name
    return None


def _output_fields(t: Mapping[str, Any]) -> tuple[str, ...]:
    for k in _OUTPUT_KEYS:
        o = t.get(k)
        if isinstance(o, Mapping):
            props = o.get("properties", o)
            return tuple(str(x) for x in props) if isinstance(props, Mapping) else ()
        if isinstance(o, (list, tuple)):
            return tuple(str(x) for x in o)
    return ()


def parse_tool_def(raw: Mapping[str, Any]) -> ToolDef:
    """Raises ValueError if the entry has no usable name."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"tool entry is not an object: {type(raw).__name__}")
    t: dict[str, Any] = dict(raw)
    fn = t.get("function")
    if isinstance(fn, Mapping):  # OpenAI shape: flags may sit outside "function"
        outer = {k: v for k, v in t.items() if k != "function" and not (k == "type" and v == "function")}
        t = {**outer, **fn}
    elif t.get("type") == "function":
        t.pop("type")
    name = t.get("name") or t.get("tool") or t.get("id")
    if not name or not isinstance(name, str):
        raise ValueError("tool entry without a name")
    schema = next((t[k] for k in _SCHEMA_KEYS if k in t and t[k] is not None), {})
    params = parse_params(schema)
    read_only, why = _classify(t)
    idem_param = _idem_param(params)
    return ToolDef(
        name=name.strip(), description=str(t.get("description", "") or ""), params=params,
        read_only=read_only, idempotent=read_only or _idempotent(t),
        idempotency_param=None if read_only else idem_param,
        output_fields=_output_fields(t), decided_by=why,
    )


def parse_manifest_defs(payload: Any) -> tuple[list[ToolDef], list[str]]:
    """Never raises. Returns (tools, errors for skipped entries)."""
    errors: list[str] = []
    entries: Any = payload
    if isinstance(payload, Mapping):
        entries = next((payload[k] for k in ("tools", "manifest", "functions", "tool_manifest")
                        if k in payload and payload[k] is not None), None)
        if entries is None:  # {tool_name: def}
            entries = [{"name": k, **v} if isinstance(v, Mapping) else v for k, v in payload.items()]
        elif isinstance(entries, Mapping):
            entries = [{"name": k, **v} if isinstance(v, Mapping) else v for k, v in entries.items()]
    if not isinstance(entries, (list, tuple)):
        return [], [f"manifest has no tool list ({type(payload).__name__})"]
    tools: list[ToolDef] = []
    for i, e in enumerate(entries):
        try:
            tools.append(parse_tool_def(e))
        except Exception as exc:  # malformed entry: skip, never crash
            errors.append(f"tool[{i}]: {exc}")
    return tools, errors


def parse_tool(t: Mapping[str, Any]) -> ToolSpec:
    """One manifest entry as a ToolSpec. Raises ValueError without a name."""
    return parse_tool_def(t).spec


def parse_manifest(ev_or_payload: Event | Mapping[str, Any]) -> list[ToolSpec]:
    """Never raises. Malformed entries are skipped."""
    p = ev_or_payload.payload if isinstance(ev_or_payload, Event) else ev_or_payload
    return [d.spec for d in parse_manifest_defs(p)[0]]


# ---- actions ----------------------------------------------------------------

class IdGen:
    """Deterministic per-session ids (U13, U26): call-0001, act-0001, ..."""

    def __init__(self) -> None:
        self._n: dict[str, int] = {}

    def __call__(self, prefix: str) -> str:
        n = self._n.get(prefix, 0) + 1
        self._n[prefix] = n
        return f"{prefix}-{n:04d}"


_fallback_ids = IdGen()  # only for callers that do not pass their own IdGen


def new_id(prefix: str) -> str:
    """Compat helper. Sessions should own an IdGen instead."""
    return _fallback_ids(prefix)


def snapshot_payload(intent: str | None, slots: Mapping[str, Any]) -> dict[str, Any]:
    """U16: {"intent": str|None, "slots": {name: json value}}."""
    return {"intent": intent, "slots": {k: v for k, v in slots.items() if v is not None}}


def make_action(type_: str, t: float, snapshot: dict[str, Any] | None = None, *,
                ids: IdGen | None = None, **fields: Any) -> dict[str, Any]:
    """Internal action. Every action carries an action_id, a timestamp and the
    current snapshot. ASSUMPTION: attaching the snapshot to every action (not
    just finals) is allowed and serves 'updated state snapshots' (G§5)."""
    act: dict[str, Any] = {"type": type_, "action_id": (ids or _fallback_ids)("act"), "t": round(float(t), 3)}
    act.update(fields)
    if snapshot is not None:
        act["state_snapshot"] = snapshot
    return act


def validate_action(a: Mapping[str, Any]) -> list[str]:
    """Validates an internal action. Empty list == valid (U24)."""
    errs: list[str] = []
    typ = a.get("type")
    if typ not in ACTION_TYPES:
        errs.append(f"unknown type {typ!r}")
    if not isinstance(a.get("action_id"), str) or not a.get("action_id"):
        errs.append("missing action_id")
    if not isinstance(a.get("t"), (int, float)):
        errs.append("missing t")
    if typ in (ACT_SPEAK, ACT_CLARIFY, ACT_FINAL) and not str(a.get("text", "")).strip():
        errs.append("empty text")
    if typ == ACT_SPEAK and a.get("kind") not in SPEAK_KINDS:
        errs.append(f"bad speak kind {a.get('kind')!r}")
    if typ in (ACT_TOOL_CALL, ACT_CANCEL) and not a.get("call_id"):
        errs.append("missing call_id")
    if typ == ACT_TOOL_CALL:
        if not a.get("tool"):
            errs.append("missing tool")
        if not isinstance(a.get("args"), dict):
            errs.append("args must be an object")
    snap = a.get("state_snapshot")
    if typ == ACT_FINAL and snap is None:
        errs.append("final_response requires state_snapshot")
    if snap is not None:
        if not isinstance(snap, dict) or "intent" not in snap or not isinstance(snap.get("slots"), dict):
            errs.append("malformed state_snapshot")
    try:
        json.dumps(a, allow_nan=False)
    except (TypeError, ValueError):
        errs.append("not JSON-serialisable")
    return errs


def to_wire(a: Mapping[str, Any], fmt: WireFormat | None = None) -> dict[str, Any]:
    """Internal action -> kit payload (U05, U12, U18)."""
    fmt = fmt or WireFormat()
    out: dict[str, Any] = {}
    for k, v in a.items():
        if k == "t":
            out[fmt.ts_key] = fmt.out_time(float(v))
        elif k == "type":
            out["type"] = WIRE_TYPES.get(v, v)
        else:
            out[WIRE_FIELDS.get(k, k)] = v
    return out


def from_wire(a: Mapping[str, Any], fmt: WireFormat | None = None) -> dict[str, Any]:
    """Kit payload -> internal action (local harness and tests)."""
    fmt = fmt or WireFormat()
    back_f = {v: k for k, v in WIRE_FIELDS.items()}
    back_t = {v: k for k, v in WIRE_TYPES.items()}
    out: dict[str, Any] = {}
    for k, v in a.items():
        if k == fmt.ts_key:
            out["t"] = to_ms(v, fmt.unit)
        elif k == "type":
            out["type"] = back_t.get(v, v)
        else:
            out[back_f.get(k, k)] = v
    return out


def dumps(a: Mapping[str, Any]) -> str:
    return json.dumps(a, separators=(",", ":"), sort_keys=True)


# ---- clock (U06): implementations live in clock.py -------------------------

from .clock import Clock, EventClock, wall_clock_ms  # noqa: E402,F401  (re-export)
