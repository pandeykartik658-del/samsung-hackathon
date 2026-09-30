# /mnt/project-files/theme5/theme5/fastpath.py
"""Fast path: rule-based floor management. No LLM, no I/O, no awaits.

What it does
- Classifies every user text chunk: new request, correction, cancellation,
  clarification answer, backchannel ("uh-huh"). Built on nlu.py so the fast
  path and the planner agree on cancel/repair cues.
- Speaks first, quickly, and only what is true. Narration is derived from
  actions the agent actually emitted (tool_call, cancel) and results it
  actually received. "Checking flights to Mumbai" exists only while a call
  with destination=Mumbai is in flight. No template here claims completion,
  and a guard drops any text that would (claims_completion).
- Rate-limits fillers/progress: min gap, no repeated text, per-turn cap.
  Silent on backchannels and while the user holds the floor.
- Narrates calls that stay in flight past configurable thresholds.

How the engine drives it (all synchronous, microseconds):
    cls = fp.on_user_text(chunk, t, end_of_turn, awaiting_clarification=..., has_active_task=...)
    fp.on_interrupt(t)                     # barge-in signal
    fp.on_media("audio" | "frame", t)      # -> speech actions
    fp.on_slots_changed({slot: (old, new)})  # optional, improves correction acks
    fp.observe(action)                     # EVERY action the engine emits -> speech to emit next
    fp.on_result(tool_result, t)           # every tool result
    fp.flush(t)                            # end of each event handler -> speech
    fp.tick(t)                             # timer; fp.next_deadline() says when (FastPathTicker does this)
Speech is returned as protocol actions (dicts); the engine puts them on the
outbox. Actions the fast path created are ignored when passed back to observe().

The trace audits at the bottom (audit_*) read sim/trace.py records and are
used by the tests; the harness scorer may reuse them.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping

from . import nlu
from .protocol import (
    ACK_ON_INTERRUPT, ACT_CANCEL, ACT_CLARIFY, ACT_FINAL, ACT_SPEAK, ACT_TOOL_CALL, EV_INTERRUPT,
    EV_MANIFEST, EV_TEXT, EV_TOOL_RESULT, SPEAK_ACK, SPEAK_FILLER, SPEAK_INFO, SPEAK_PROGRESS,
    ToolResult, is_end_of_turn, make_action, parse_event, parse_manifest, parse_tool_result, text_of,
)
from .tools import canonical_args


# --------------------------------------------------------------------------
# Config. ASSUMPTION: every value is a tuning choice, not given by the guide.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FastPathConfig:
    latency_budget_ms: float = 300.0  # guide G§1 "a few hundred milliseconds"; ASSUMPTION: 300
    ack_deadline_ms: float = 150.0  # generic ack if nothing concrete was said by then
    filler_min_gap_ms: float = 3000.0  # at most 1 filler/progress line per 3 s
    max_fillers_per_turn: int = 2  # filler/progress lines between two user turns
    progress_after_ms: tuple[float, ...] = (2000.0, 6000.0)  # "still ..." thresholds per call
    narrate_chained: bool = True  # "Now booking ..." for calls not tied to a user turn
    ack_on_interrupt: bool = ACK_ON_INTERRUPT
    max_value_chars: int = 32  # longest arg value quoted in narration


FILLER_KINDS = (SPEAK_FILLER, SPEAK_PROGRESS)
SUBSTANTIVE_SPEAK = (SPEAK_ACK, SPEAK_PROGRESS, SPEAK_INFO)  # ASSUMPTION (SPEC U21): plain fillers don't count


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
class InputKind(str, Enum):
    NEW_REQUEST = "new_request"
    CORRECTION = "correction"
    CANCELLATION = "cancellation"
    CLARIFICATION_ANSWER = "clarification_answer"
    BACKCHANNEL = "backchannel"
    EMPTY = "empty"


@dataclass(frozen=True)
class Classification:
    kind: InputKind
    text: str  # cleaned text the kind was decided on
    confidence: float
    cue: str | None = None  # matched cue, e.g. "actually", "never mind"
    self_repair: bool = False  # repair inside a fresh request ("to Delhi, no, Mumbai")
    final: bool = True  # decided on a complete turn (False for partial chunks)

    @property
    def takes_floor(self) -> bool:
        """False for backchannels/empty: they must not interrupt, cancel or trigger speech."""
        return self.kind not in (InputKind.BACKCHANNEL, InputKind.EMPTY)


_HESITATION_RE = re.compile(r"^(?:u+h+|u+m+|e+r+m*|e+h+|h+m+|a+h+|m+)$")
_BACKCHANNEL_PHRASES = {
    ("uh", "huh"), ("uh-huh",), ("uhhuh",), ("mhm",), ("mhmm",), ("mm", "hmm"), ("mm-hmm",), ("hmm",),
    ("ok",), ("okay",), ("k",), ("yeah",), ("yea",), ("ya",), ("yep",), ("yup",), ("right",), ("sure",),
    ("alright",), ("all", "right"), ("cool",), ("nice",), ("great",), ("fine",), ("good",), ("i", "see"),
    ("got", "it"), ("oh",), ("ah",), ("aha",), ("go", "on"), ("thanks",), ("thank", "you"), ("wow",),
    ("sounds", "good"), ("makes", "sense"), ("yes",), ("uh",), ("um",), ("oh", "okay"), ("so",), ("and",),
}
_ANSWER_WORDS = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "right", "correct", "mhm", "uh-huh",
                 "no", "nope", "nah", "alright", "exactly", "please"}
_REQUEST_RE = re.compile(
    r"\b(?:book|find|search|get|show|create|open|check|look|need|want|can you|could you|please|help|"
    r"what|how|where|when|why|which|who|is|are|tell|give|make|reserve|navigate|take me|raise|file|report)\b", re.I)
_BARE_CANCEL_OBJECTS = re.compile(r"\b(?:booking|reservation|ticket|order|flight)\b", re.I)


def _norm(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9'\- ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _covered(tokens: list[str], phrases: set[tuple[str, ...]]) -> bool:
    i = 0
    while i < len(tokens):
        for n in (3, 2, 1):
            if tuple(tokens[i:i + n]) in phrases:
                i += n
                break
        else:
            return False
    return True


def classify(text: str, *, awaiting_clarification: bool = False, has_active_task: bool = False,
             final: bool = True) -> Classification:
    """Pure rule classifier over one (possibly partial) user turn."""
    norm = _norm(text)
    if not norm:
        return Classification(InputKind.EMPTY, "", 1.0, final=final)
    toks = norm.split()
    if all(_HESITATION_RE.match(t) for t in toks):
        return Classification(InputKind.BACKCHANNEL, norm, 0.95, cue=norm, final=final)

    p = nlu.parse(text)
    clean = p.clean or norm

    if p.cancel:
        m = nlu.CANCEL_RE.search(clean)
        cue = m.group(0).lower() if m else None
        # "cancel the booking" with nothing running is a request for the cancel tool
        if not has_active_task and not awaiting_clarification and _BARE_CANCEL_OBJECTS.search(clean):
            return Classification(InputKind.NEW_REQUEST, clean, 0.6, cue=cue, final=final)
        return Classification(InputKind.CANCELLATION, clean, 0.9, cue=cue, final=final)

    if awaiting_clarification:
        short = len(toks) <= 3 or toks[0] in _ANSWER_WORDS
        return Classification(InputKind.CLARIFICATION_ANSWER, clean, 0.9 if short else 0.75,
                              cue=toks[0] if toks[0] in _ANSWER_WORDS else None, final=final)

    if len(toks) <= 5 and _covered(toks, _BACKCHANNEL_PHRASES):
        return Classification(InputKind.BACKCHANNEL, norm, 0.85, cue=norm, final=final)

    m = nlu.REPAIR_RE.search(clean)
    cue = m.group(0).strip(" ,").lower() if m else None
    if p.correction:
        if has_active_task:
            return Classification(InputKind.CORRECTION, clean, 0.85, cue=cue, final=final)
        return Classification(InputKind.NEW_REQUEST, clean, 0.75, cue=cue, self_repair=m is not None, final=final)

    if has_active_task:
        if p.deny and not p.slots and p.intent is None:
            return Classification(InputKind.CORRECTION, clean, 0.5, cue=toks[0], final=final)
        if p.intent is None and len(toks) <= 3 and not _REQUEST_RE.search(clean):
            return Classification(InputKind.CORRECTION, clean, 0.5, final=final)  # bare value: "Mumbai"

    return Classification(InputKind.NEW_REQUEST, clean, 0.8 if p.intent else 0.6, final=final)


# --------------------------------------------------------------------------
# Honest narration of a real tool call
# --------------------------------------------------------------------------
_COMPLETION_RE = re.compile(
    r"\b(?:booked|confirmed|reserved|done|completed?|finished|created|cancell?ed|scheduled|submitted|"
    r"succeeded|successful(?:ly)?|found|placed|issued|sorted|all set|went through|here(?:'s| is| are)|"
    r"(?:is|are) ready|has been|have been|i(?:'ve| have) (?:booked|made|created|found|placed|sent|opened|"
    r"filed|reserved|scheduled|submitted|updated|changed|fixed))\b", re.I)
_NEGATED_RE = re.compile(r"(?:\bnot|n't|\bnever|\bno)\s+(?:\w+\s+)?$", re.I)


def claims_completion(text: str) -> bool:
    """True if the text asserts that work finished. Negated uses ("wasn't booked") do not count."""
    for m in _COMPLETION_RE.finditer(text):
        if not _NEGATED_RE.search(text[:m.start()]):
            return True
    return False


_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NOISE = {"api", "tool", "v1", "v2", "v3", "by", "for", "the", "a", "an", "of", "info", "data", "request"}
_TO_KEYS = ("destination", "dest", "to", "to_city", "destination_city", "arrival", "arrival_city",
            "arrival_airport", "destination_airport")
_FROM_KEYS = ("origin", "from", "from_city", "origin_city", "source", "departure_city", "departure_airport",
              "origin_airport")
_DATE_KEYS = ("date", "departure_date", "travel_date", "depart_date", "day", "when")
_PLACE_KEYS = ("city", "location", "address")
_ABOUT_KEYS = ("city", "location", "address", "device", "model", "product", "appliance", "error_code", "code", "topic", "subject", "issue", "query")

# verb -> (gerund, article for a singular noun, plural-noun gerund or None)
_VERBS: dict[str, tuple[str, str, str | None]] = {
    "search": ("looking for", "a", "checking"), "find": ("looking for", "a", "checking"),
    "list": ("checking", "", "checking"), "query": ("looking for", "a", "checking"),
    "check": ("checking", "the", "checking"), "get": ("checking", "the", "checking"),
    "fetch": ("checking", "the", "checking"), "retrieve": ("checking", "the", "checking"),
    "read": ("checking", "the", "checking"), "show": ("checking", "the", "checking"),
    "view": ("checking", "the", "checking"), "lookup": ("looking up", "the", "looking up"),
    "look": ("looking up", "the", "looking up"), "book": ("booking", "a", "booking"),
    "reserve": ("reserving", "a", "reserving"), "create": ("creating", "a", "creating"),
    "open": ("opening", "a", "opening"), "file": ("filing", "a", "filing"), "raise": ("raising", "a", "raising"),
    "log": ("logging", "a", "logging"), "add": ("adding", "a", "adding"), "make": ("making", "a", "making"),
    "submit": ("submitting", "the", "submitting"), "send": ("sending", "the", "sending"),
    "notify": ("sending", "the", "sending"), "cancel": ("cancelling", "the", "cancelling"),
    "delete": ("removing", "the", "removing"), "remove": ("removing", "the", "removing"),
    "update": ("updating", "the", "updating"), "modify": ("updating", "the", "updating"),
    "change": ("changing", "the", "changing"), "edit": ("updating", "the", "updating"),
    "reschedule": ("rescheduling", "the", "rescheduling"), "confirm": ("confirming", "the", "confirming"),
    "pay": ("processing payment for", "the", "processing payment for"),
    "navigate": ("working out the route", "", None), "route": ("working out the route", "", None),
    "compute": ("working out", "the", "working out"), "calculate": ("working out", "the", "working out"),
    "plan": ("planning", "the", "planning"), "analyze": ("looking at", "the", "looking at"),
    "analyse": ("looking at", "the", "looking at"), "describe": ("looking at", "the", "looking at"),
    "detect": ("looking at", "the", "looking at"), "identify": ("identifying", "the", "identifying"),
    "transcribe": ("transcribing", "the", "transcribing"), "translate": ("translating", "the", "translating"),
}


def _words(name: str) -> list[str]:
    return [w for w in re.split(r"[_\-\s.]+", _CAMEL_RE.sub("_", name).lower()) if w]


def _plural(noun: str) -> bool:
    last = noun.split()[-1] if noun else ""
    return last.endswith("s") and not last.endswith(("ss", "us", "is", "status"))


def _value(v: Any, limit: int, title: bool) -> str | None:
    if isinstance(v, bool) or not isinstance(v, (str, int, float)):
        return None
    s = str(v).strip()
    if not s or len(s) > limit:
        return None
    if title and s.islower():
        s = " ".join(w.capitalize() for w in s.split())
    return s


def _arg_phrase(args: Mapping[str, Any], limit: int, with_to: bool = True) -> str:
    low = {str(k).lower(): v for k, v in args.items()}
    parts: list[str] = []

    def first(keys: tuple[str, ...], title: bool) -> str | None:
        for k in keys:
            if k in low:
                val = _value(low[k], limit, title)
                if val is not None:
                    return val
        return None

    origin = first(_FROM_KEYS, True)
    dest = first(_TO_KEYS, True) if with_to else None
    if origin:
        parts.append(f"from {origin}")
    if dest:
        parts.append(f"to {dest}")
    d = first(_DATE_KEYS, False)
    if d:
        parts.append(f"on {d}" if re.match(r"\d{4}-\d{2}-\d{2}$", d) else f"for {d}")
    if not parts:
        for k in _ABOUT_KEYS:
            about = _value(low[k], limit, k in _PLACE_KEYS) if k in low else None
            if about and len(about.split()) <= 4:
                parts.append(f"{'in' if k in _PLACE_KEYS else 'for'} {about}")
                break
    return " ".join(parts)


def describe_call(tool: str, args: Mapping[str, Any] | None = None, max_value_chars: int = 32) -> str:
    """Lower-case gerund phrase for an in-flight call, e.g.
    ('search_flights', {'destination': 'Mumbai'}) -> 'checking flights to Mumbai'.
    Works for unseen tools; falls back to 'working on that'."""
    args = args or {}
    words = _words(tool)
    vi = next((i for i, w in enumerate(words) if w in _VERBS), None)
    noun_words = [w for i, w in enumerate(words) if i != vi and w not in _NOISE]
    noun = " ".join(noun_words)
    if vi is None:
        head = f"working on the {noun}" if noun else "working on that"
    else:
        gerund, article, plural_gerund = _VERBS[words[vi]]
        if plural_gerund is None:  # route-style verbs carry their own noun
            head = gerund
        elif not noun:
            head = {"looking for": "looking that up", "checking": "checking that"}.get(gerund, f"{gerund} that")
        elif _plural(noun):
            head = f"{plural_gerund} {noun}"
        else:
            art = ("an" if noun[0] in "aeiou" else "a") if article == "a" else article
            head = f"{gerund} {art + ' ' if art else ''}{noun}"
    phrase = _arg_phrase(args, max_value_chars)
    text = f"{head} {phrase}".strip()
    return text if not claims_completion(text) else "working on that"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


# --------------------------------------------------------------------------
# Rate limiter for filler/progress speech
# --------------------------------------------------------------------------
@dataclass
class FillerLimiter:
    min_gap_ms: float
    max_per_turn: int
    last_t: float | None = None
    turn_count: int = 0
    said: set[str] = field(default_factory=set)

    def allows(self, text: str, t: float) -> bool:
        if _key(text) in self.said or self.turn_count >= self.max_per_turn:
            return False
        return self.last_t is None or t - self.last_t >= self.min_gap_ms

    def blocked_forever(self, text: str) -> bool:
        return _key(text) in self.said or self.turn_count >= self.max_per_turn

    def next_allowed(self) -> float:
        return float("-inf") if self.last_t is None else self.last_t + self.min_gap_ms

    def record(self, text: str, t: float) -> None:
        self.said.add(_key(text))
        self.turn_count += 1
        self.last_t = t

    def new_turn(self) -> None:
        self.turn_count = 0


# --------------------------------------------------------------------------
# The fast path
# --------------------------------------------------------------------------
_ACKS: dict[str, tuple[str, ...]] = {
    InputKind.NEW_REQUEST.value: ("Sure, one moment.", "Okay, let me look into that.", "Got it, one second."),
    InputKind.CORRECTION.value: ("Got it, let me change that.", "Okay, updating that.", "Sure, changing that."),
    InputKind.CLARIFICATION_ANSWER.value: ("Got it, thanks.", "Thanks, got it.", "Okay, thanks."),
    InputKind.CANCELLATION.value: ("Okay, no problem.", "Sure, I'll leave it there.", "Okay, never mind then."),
    "interrupt": ("Sorry, go ahead.", "Yes?"),
    "audio": ("Let me listen to that.", "One moment, listening."),
    "frame": ("Let me take a look at the picture.", "Looking at the image now."),
}


@dataclass
class _Flight:
    call_id: str
    tool: str
    phrase: str
    start: float
    turn: int
    stage: int = 0


@dataclass
class _Pending:
    kind: str  # InputKind value, "interrupt", "audio" or "frame"
    since: float
    turn: int
    cancels: list[str] = field(default_factory=list)
    change: str | None = None  # "Mumbai instead of Delhi"


class FastPath:
    def __init__(self, config: FastPathConfig | None = None,
                 snapshot: Callable[[], dict[str, Any]] | None = None) -> None:
        self.cfg = config or FastPathConfig()
        self._snapshot = snapshot
        self.limiter = FillerLimiter(self.cfg.filler_min_gap_ms, self.cfg.max_fillers_per_turn)
        self.inflight: dict[str, _Flight] = {}
        self.last: Classification | None = None
        self.user_speaking = False
        self._buf = ""
        self._turn = 0
        self._pending: _Pending | None = None
        self._failed_keys: set[str] = set()
        self._keys: dict[str, str] = {}  # call_id -> tool|args key
        self._own: set[str] = set()  # action_ids we produced
        self._ack_i: dict[str, int] = {}
        self._last_ack: str | None = None

    # ---------------------------------------------------------------- input
    def on_user_text(self, text: str, t: float, end_of_turn: bool = False, *,
                     awaiting_clarification: bool = False, has_active_task: bool = False) -> Classification:
        """Feed one chunk; returns the classification of the turn so far."""
        chunk = (text or "").strip()
        if chunk and self._buf and chunk.lower().startswith(self._buf.lower()):
            self._buf = chunk  # cumulative chunk (ASSUMPTION SPEC U07: both styles occur)
        elif chunk:
            self._buf = f"{self._buf} {chunk}".strip()
        has_task = has_active_task or bool(self.inflight)
        cls = classify(self._buf, awaiting_clarification=awaiting_clarification,
                       has_active_task=has_task, final=end_of_turn)
        self.last = cls
        if not end_of_turn:
            if cls.takes_floor:
                self.user_speaking = True
            return cls
        self._buf = ""
        self.user_speaking = False
        if cls.takes_floor:
            self._turn += 1
            self.limiter.new_turn()
            self._pending = _Pending(cls.kind.value, t, self._turn)
        return cls

    def on_interrupt(self, t: float) -> None:
        """Barge-in. The user now holds the floor; queued narration waits for their turn."""
        self.user_speaking = True
        if self.cfg.ack_on_interrupt:
            self._pending = _Pending("interrupt", t, self._turn)

    def on_media(self, kind: str, t: float) -> list[dict[str, Any]]:
        """Audio/frame received and handed to the slow path: acknowledge it."""
        k = "audio" if kind.lower().startswith(("aud", "wav")) else "frame"
        self._turn += 1
        self.limiter.new_turn()
        self._pending = _Pending(k, t, self._turn)
        if k == "frame" or not self.user_speaking:
            return self._ack(self._next_ack(k), t)
        return []

    def on_slots_changed(self, changes: Mapping[str, tuple[Any, Any]]) -> None:
        """Optional: lets correction acks name the change ('Mumbai instead of Delhi')."""
        if self._pending is None:
            return
        for _, (old, new) in changes.items():
            o, n = _value(old, self.cfg.max_value_chars, False), _value(new, self.cfg.max_value_chars, False)
            if n is not None:
                self._pending.change = f"{n} instead of {o}" if o is not None else str(n)
                return

    # ---------------------------------------------------------------- engine actions / results
    def observe(self, action: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Call for every action the engine emits, right after emitting it."""
        if action.get("action_id") in self._own:
            return []
        typ = action.get("type")
        t = float(action.get("t", 0.0))
        if typ == ACT_TOOL_CALL:
            return self._on_call(action, t)
        if typ == ACT_CANCEL:
            f = self.inflight.pop(str(action.get("call_id")), None)
            if f is not None and self._pending is not None:
                self._pending.cancels.append(f.phrase)
            return []
        if typ == ACT_SPEAK:
            kind = action.get("kind")
            if kind in FILLER_KINDS:
                self.limiter.record(str(action.get("text", "")), t)
            if kind in SUBSTANTIVE_SPEAK:
                self._pending = None
            return []
        if typ in (ACT_CLARIFY, ACT_FINAL):
            self._pending = None
        return []

    def on_result(self, result: ToolResult, t: float) -> list[dict[str, Any]]:
        """Tool result arrived. The fast path never announces results; the engine's final does."""
        self.inflight.pop(result.call_id, None)
        key = self._keys.get(result.call_id)
        if key is not None and not result.ok:
            self._failed_keys.add(key)
        return []

    # ---------------------------------------------------------------- timing
    def flush(self, t: float) -> list[dict[str, Any]]:
        """End of an event handler: say the concrete thing now if there is one."""
        p = self._pending
        if p is None or self.user_speaking:
            return []
        f = self._turn_flight(p.turn)
        if f is not None:
            return self._ack(self._call_ack(f, p), t)
        if p.kind == InputKind.CANCELLATION.value:
            if p.cancels:
                what = p.cancels[0] if len(p.cancels) == 1 else "everything I was working on"
                return self._ack(f"Okay, I've stopped {what}.", t)
            return self._ack(self._next_ack(p.kind), t)
        if p.kind == InputKind.CLARIFICATION_ANSWER.value:
            return self._ack(self._next_ack(p.kind), t)
        return []

    def tick(self, t: float) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if self.user_speaking:
            return out
        p = self._pending
        if p is not None and t >= p.since + self.cfg.ack_deadline_ms:
            out += self.flush(t)
            if self._pending is not None:
                text = self._next_ack(p.kind)
                if p.kind == InputKind.CORRECTION.value and p.change:
                    text = f"Got it, {p.change}."
                out += self._ack(text, t)
        for f in sorted(self.inflight.values(), key=lambda x: (x.start, x.call_id)):
            while f.stage < len(self.cfg.progress_after_ms) and t >= f.start + self.cfg.progress_after_ms[f.stage]:
                text = self._progress_text(f)
                if self.limiter.blocked_forever(text):
                    f.stage += 1
                    continue
                if self.limiter.allows(text, t):
                    f.stage += 1
                    out += self._say(text, SPEAK_PROGRESS, t)
                break
        return out

    def next_deadline(self) -> float | None:
        """Earliest virtual time at which tick() may speak; None if nothing is due."""
        if self.user_speaking:
            return None
        cands: list[float] = []
        if self._pending is not None:
            cands.append(self._pending.since + self.cfg.ack_deadline_ms)
        if self.limiter.turn_count < self.limiter.max_per_turn:
            for f in self.inflight.values():
                if f.stage < len(self.cfg.progress_after_ms):
                    cands.append(max(f.start + self.cfg.progress_after_ms[f.stage], self.limiter.next_allowed()))
        return min(cands) if cands else None

    # ---------------------------------------------------------------- internals
    def _on_call(self, a: Mapping[str, Any], t: float) -> list[dict[str, Any]]:
        call_id = str(a.get("call_id"))
        tool = str(a.get("tool") or a.get("name") or "")
        args = a.get("args") if isinstance(a.get("args"), Mapping) else {}
        key = f"{tool}|{canonical_args(dict(args))}"
        retry = key in self._failed_keys
        self._failed_keys.discard(key)
        self._keys[call_id] = key
        f = _Flight(call_id, tool, describe_call(tool, args, self.cfg.max_value_chars), t,
                    self._turn + (1 if self.user_speaking else 0))
        self.inflight[call_id] = f
        if self.user_speaking:
            return []  # speculative call mid-utterance: narrate at end of turn
        p = self._pending
        if p is not None and p.turn == f.turn:
            return self._ack(self._call_ack(f, p), t)
        if retry:
            text = f"That didn't go through, so I'm {f.phrase} again."
        elif self.cfg.narrate_chained:
            text = f"Now {f.phrase}."
        else:
            return []
        return self._say(text, SPEAK_PROGRESS, t) if self.limiter.allows(text, t) else []

    def _turn_flight(self, turn: int) -> _Flight | None:
        cands = [f for f in self.inflight.values() if f.turn == turn]
        return max(cands, key=lambda f: (f.start, f.call_id)) if cands else None

    def _call_ack(self, f: _Flight, p: _Pending) -> str:
        if p.kind == InputKind.CORRECTION.value or p.cancels:
            return f"Okay, {f.phrase} instead."
        return f"{_cap(f.phrase)}."

    def _progress_text(self, f: _Flight) -> str:
        if f.stage == 0:
            return f"Still {f.phrase}."
        return f"Still on it; {f.phrase} is taking a little longer than usual."

    def _next_ack(self, kind: str) -> str:
        opts = _ACKS.get(kind, _ACKS[InputKind.NEW_REQUEST.value])
        i = self._ack_i.get(kind, 0)
        text = opts[i % len(opts)]
        if text == self._last_ack and len(opts) > 1:
            i += 1
            text = opts[i % len(opts)]
        self._ack_i[kind] = i + 1
        return text

    def _ack(self, text: str, t: float) -> list[dict[str, Any]]:
        self._pending = None
        self._last_ack = text
        return self._say(text, SPEAK_ACK, t)

    def _say(self, text: str, kind: str, t: float) -> list[dict[str, Any]]:
        if claims_completion(text):  # never from the fast path
            if kind != SPEAK_ACK:
                return []
            text = "One moment."
        a = make_action(ACT_SPEAK, t, snapshot=self._snapshot() if self._snapshot else None, text=text, kind=kind)
        self._own.add(a["action_id"])
        if kind in FILLER_KINDS:
            self.limiter.record(text, t)
        return [a]


# --------------------------------------------------------------------------
# Timer driver: runs tick() at next_deadline() on the running loop's clock.
# --------------------------------------------------------------------------
class FastPathTicker:
    """Background task. `now()` returns ms on the same clock the fast path uses;
    `emit(action)` puts an action on the outbox. Call poke() after every
    handled event so a new deadline is picked up."""

    def __init__(self, fp: FastPath, emit: Callable[[dict[str, Any]], None], now: Callable[[], float]) -> None:
        self.fp, self.emit, self.now = fp, emit, now
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._run())

    def poke(self) -> None:
        self._wake.set()

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        while True:
            d = self.fp.next_deadline()
            self._wake.clear()
            if d is None:
                await self._wake.wait()
                continue
            delay = (d - self.now()) / 1000.0
            if delay > 0:
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=delay)
                    continue  # poked: recompute
                except asyncio.TimeoutError:
                    pass
            for a in self.fp.tick(self.now()):
                self.emit(a)
            if self.fp.next_deadline() == d:  # nothing progressed: avoid a hot loop
                await self._wake.wait()


# --------------------------------------------------------------------------
# Trace audits over sim/trace.py records:
#   {"seq", "t_ms", "dir": "in"|"out"|"sys", "kind", "data"}
# Each returns a list of violations (empty == pass).
# --------------------------------------------------------------------------
_NARRATION_RE = re.compile(
    r"^(?:okay, |now |still (?:on it; )?)?(?:looking (?:for|up|at)|checking|booking|reserving|creating|opening|filing|"
    r"raising|logging|submitting|sending|cancelling|removing|updating|changing|rescheduling|confirming|"
    r"working out|planning|identifying|transcribing|translating|processing)\b"
    r"|that didn't go through, so i'm", re.I)


def _recs(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(records, key=lambda r: (r.get("t_ms", 0.0), r.get("seq", 0)))


def _speech(r: Mapping[str, Any]) -> str | None:
    if r.get("dir") == "out" and r.get("kind") in (ACT_SPEAK, ACT_CLARIFY, ACT_FINAL):
        return str(r.get("data", {}).get("text", ""))
    return None


def audit_false_completion(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Completion wording is allowed only after an ok result of a live (not cancelled) call.
    Write-style claims (booked, created, ...) need an ok result from a state-modifying tool."""
    writes: set[str] = set()
    tool_of: dict[str, str] = {}
    cancelled: set[str] = set()
    ok_any = ok_write = False
    bad: list[dict[str, Any]] = []
    write_words = re.compile(r"\b(?:booked|reserved|created|cancell?ed|scheduled|submitted|placed|issued|"
                             r"updated|changed|confirmed|succeeded|went through)\b", re.I)
    for r in _recs(records):
        d = r.get("data", {})
        if r.get("dir") == "in" and r.get("kind") == EV_MANIFEST:
            writes |= {s.name for s in parse_manifest(parse_event(dict(d, type=EV_MANIFEST))) if s.state_modifying}
        elif r.get("dir") == "out" and r.get("kind") == ACT_TOOL_CALL:
            tool_of[str(d.get("call_id"))] = str(d.get("tool") or d.get("name"))
        elif r.get("dir") == "out" and r.get("kind") == ACT_CANCEL:
            cancelled.add(str(d.get("call_id")))
        elif r.get("dir") == "in" and r.get("kind") == EV_TOOL_RESULT:
            tr = parse_tool_result(parse_event(dict(d, type=EV_TOOL_RESULT)))
            if tr.ok and tr.call_id in tool_of and tr.call_id not in cancelled:
                ok_any = True
                ok_write = ok_write or tool_of[tr.call_id] in writes or not writes
        text = _speech(r)
        if text and claims_completion(text):
            need_write = bool(write_words.search(text))
            if not ok_any or (need_write and not ok_write):
                bad.append({"t_ms": r.get("t_ms"), "text": text, "why": "completion claimed before a supporting result"})
    return bad


def audit_unbacked_narration(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """'Checking flights ...' style speech needs a call actually in flight at that moment."""
    live: set[str] = set()
    bad: list[dict[str, Any]] = []
    for r in _recs(records):
        d = r.get("data", {})
        if r.get("dir") == "out" and r.get("kind") == ACT_TOOL_CALL:
            live.add(str(d.get("call_id")))
        elif r.get("dir") == "out" and r.get("kind") == ACT_CANCEL:
            live.discard(str(d.get("call_id")))
        elif r.get("dir") == "in" and r.get("kind") == EV_TOOL_RESULT:
            live.discard(parse_tool_result(parse_event(dict(d, type=EV_TOOL_RESULT))).call_id)
        text = _speech(r)
        if text and r.get("kind") == ACT_SPEAK and _NARRATION_RE.search(text) and "stopped" not in text and not live:
            bad.append({"t_ms": r.get("t_ms"), "text": text, "why": "narrates work with no call in flight"})
    return bad


def audit_fillers(records: Iterable[Mapping[str, Any]], cfg: FastPathConfig | None = None) -> list[dict[str, Any]]:
    """Filler/progress lines: min gap, no repeats, per-turn cap."""
    cfg = cfg or FastPathConfig()
    bad: list[dict[str, Any]] = []
    last: float | None = None
    seen: set[str] = set()
    per_turn = 0
    for r in _recs(records):
        if r.get("dir") == "in" and r.get("kind") == EV_TEXT and is_end_of_turn(parse_event(dict(r.get("data", {}), type=EV_TEXT))):
            if classify(text_of(parse_event(dict(r["data"], type=EV_TEXT)))).takes_floor:
                per_turn = 0
            continue
        if not (r.get("dir") == "out" and r.get("kind") == ACT_SPEAK and r.get("data", {}).get("kind") in FILLER_KINDS):
            continue
        t, text = float(r.get("t_ms", 0.0)), str(r["data"].get("text", ""))
        if last is not None and t - last < cfg.filler_min_gap_ms:
            bad.append({"t_ms": t, "text": text, "why": f"filler gap {t - last:.0f} ms < {cfg.filler_min_gap_ms:.0f} ms"})
        if _key(text) in seen:
            bad.append({"t_ms": t, "text": text, "why": "repeated filler"})
        per_turn += 1
        if per_turn > cfg.max_fillers_per_turn:
            bad.append({"t_ms": t, "text": text, "why": "too many fillers in one turn"})
        seen.add(_key(text))
        last = t
    return bad


def audit_latency(records: Iterable[Mapping[str, Any]], budget_ms: float = 300.0,
                  include_interrupts: bool = True) -> list[dict[str, Any]]:
    """Time from each user input (end-of-turn text that takes the floor, or an
    interrupt) to the first substantive spoken action. A trigger followed by
    another trigger before any response is measured from the later one.
    ASSUMPTION: substantive = speak(ack|progress|info), clarify, final_response."""
    bad: list[dict[str, Any]] = []
    trigger: tuple[float, str] | None = None
    for r in _recs(records):
        d = r.get("data", {})
        t = float(r.get("t_ms", 0.0))
        if r.get("dir") == "in" and r.get("kind") == EV_TEXT:
            ev = parse_event(dict(d, type=EV_TEXT))
            if is_end_of_turn(ev) and classify(text_of(ev)).takes_floor:
                trigger = (t, text_of(ev))
            continue
        if r.get("dir") == "in" and r.get("kind") == EV_INTERRUPT and include_interrupts:
            trigger = (t, "<interrupt>")
            continue
        if trigger is None or r.get("dir") != "out":
            continue
        k = r.get("kind")
        if k in (ACT_CLARIFY, ACT_FINAL) or (k == ACT_SPEAK and d.get("kind") in SUBSTANTIVE_SPEAK):
            if t - trigger[0] > budget_ms:
                bad.append({"t_ms": t, "input": trigger[1], "latency_ms": t - trigger[0]})
            trigger = None
    if trigger is not None:
        bad.append({"t_ms": None, "input": trigger[1], "latency_ms": None, "why": "no response"})
    return bad
