# /mnt/project-files/theme5/theme5/slots.py
"""Session-scoped slot memory with localized corrections.

Pipeline (all rule-based, synchronous, sub-millisecond per utterance):
  text chunks -> clean hesitations -> split at self-repair markers ->
  typed mentions (city, date, count, ...) with cues -> resolve each mention to
  one slot -> ordered events -> SlotMemory applies them with provenance.

A correction only touches the slot it targets: a bare value after a repair
marker ("no wait, Pune") replaces the most recent slot of the same type, a
negated value ("not Mumbai, Pune") replaces whichever slot holds it, and a
slot noun ("change the origin to Chennai") names the slot outright.

Optional LLM refinement goes through plugins.LLMPlugin, is only tried on the
slow path when the rules found nothing, and is bounded by LLM_TIMEOUT_S.

Nothing here is module-level mutable state: every SlotTracker owns its memory,
so there is no cross-session persistence (guide section 6).

ASSUMPTION: slot names follow planner.SLOT_SYNONYMS keys; value formats follow
nlu (dates ISO when a reference date is given, else surface form such as
"5th", "friday", "march 5").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from . import nlu
from .planner import slot_for_param
from .plugins import LLMPlugin, llm_parse
from .protocol import ASR_MIN_CONFIDENCE, LLM_TIMEOUT_S, ToolSpec, snapshot_payload

# --------------------------------------------------------------------------
# Schema. ASSUMPTION: names and relevance sets are ours, not the guide's.
# --------------------------------------------------------------------------
SLOT_TYPES: dict[str, str] = {
    "origin": "city", "destination": "city", "location": "place",
    "date": "date", "return_date": "date", "time": "time",
    "passengers": "count", "cabin": "cabin", "choice": "choice",
    "name": "name", "email": "email", "phone": "phone", "booking_ref": "ref",
    "issue": "text", "query": "text", "priority": "priority",
    "device": "device", "error_code": "code", "frame": "frame",
}
_FLIGHT = {"origin", "destination", "date", "return_date", "time", "passengers", "cabin", "choice"}
_CONTACT = {"name", "email", "phone"}
INTENT_SLOTS: dict[str, frozenset[str]] = {
    "search_flights": frozenset(_FLIGHT),
    "book_flight": frozenset(_FLIGHT | _CONTACT),
    "cancel_booking": frozenset({"booking_ref", "origin", "destination", "date"} | _CONTACT),
    "create_ticket": frozenset({"booking_ref", "issue", "priority", "device", "error_code"} | _CONTACT),
    "lookup_manual": frozenset({"device", "error_code", "query"}),
    "navigate": frozenset({"origin", "destination", "location", "time"}),
}
ALWAYS_KEEP = frozenset({"frame"})  # perception context survives goal changes

CONF_SLOT_NOUN = 0.95  # "change the origin to X"
CONF_CUE = 0.9  # "from X", "5 passengers"
CONF_ANCHOR = 0.9  # "not X, Y"
CONF_REPAIR = 0.85  # "no wait, Y"
CONF_BACKREF = 0.8  # "the earlier date" resolved uniquely
CONF_DEFAULT = 0.6  # bare value, role guessed
CONF_CLARIFIED = 0.95  # user answered our clarification
LLM_MAX_CONF = 0.7  # LLM output is never trusted above this

# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotValue:
    value: Any
    confidence: float
    source: str | None  # chunk id that set it
    t: float  # harness time of that chunk
    turn: int
    via: str  # "rule" | "llm" | "clarify" | "external"
    seq: int  # global order of assignment within the session


@dataclass(frozen=True)
class Assign:
    slot: str
    value: Any
    confidence: float
    surface: str = ""
    via: str = "rule"
    seg: int = 0


@dataclass(frozen=True)
class SetIntent:
    intent: str
    seg: int = 0


@dataclass(frozen=True)
class Clarification:
    kind: str  # "value" (pick among options) | "role" (which slot) | "missing"
    slot: str | None
    question: str
    options: tuple[Any, ...] = ()
    value: Any = None  # for "role": the value whose slot is unclear
    current: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "slot": self.slot, "question": self.question,
                "options": list(self.options), "value": self.value}


@dataclass
class ParseResult:
    events: list[Assign | SetIntent] = field(default_factory=list)
    clarification: Clarification | None = None
    answered_pending: bool = False

    @property
    def empty(self) -> bool:
        return not self.events and self.clarification is None and not self.answered_pending


@dataclass
class SlotUpdate:
    intent: str | None
    intent_changed: bool
    changed: dict[str, Any]  # slot -> new value (None == removed)
    corrected: set[str]  # slots that had a value before this turn and now differ
    dropped: set[str]  # parked by a goal change
    clarification: Clarification | None
    snapshot: dict[str, Any]
    final: bool
    previous: dict[str, Any] = field(default_factory=dict)  # slot -> value before this update

    @property
    def transitions(self) -> dict[str, tuple[Any, Any]]:
        """{slot: (old, new)}, the shape fastpath.FastPath.on_slots_changed takes."""
        return {k: (self.previous.get(k), v) for k, v in self.changed.items()}

    @property
    def any(self) -> bool:
        return bool(self.changed or self.intent_changed or self.clarification)

    def apply_to(self, state: Any) -> set[str]:
        """Mirror into state.SessionState so its version counter stays the
        single staleness signal for the planner."""
        changed = state.update(dict(self.changed), correction=bool(self.corrected))
        if self.intent_changed:
            state.set_intent(self.intent)
        return changed


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------


class SlotMemory:
    """One session's slots. Also the read-only view the parser resolves against."""

    def __init__(self, intent_slots: Mapping[str, Iterable[str]] | None = None) -> None:
        self.intent: str | None = None
        self.slots: dict[str, SlotValue] = {}
        self.parked: dict[str, SlotValue] = {}  # dropped by a goal change, restorable
        self.history: dict[str, list[SlotValue]] = {}
        self.corrections: list[tuple[str, Any, Any, str | None]] = []
        self.pending: Clarification | None = None
        self.turn = 0
        self._seq = 0
        self.intent_slots: dict[str, frozenset[str]] = dict(INTENT_SLOTS)
        for k, v in (intent_slots or {}).items():
            self.intent_slots[k] = frozenset(v)

    def clone(self) -> "SlotMemory":
        """Cheap copy: SlotValue and Clarification are immutable, so only the
        containers are duplicated (deepcopy was the hot spot)."""
        c = SlotMemory.__new__(SlotMemory)
        c.intent, c.pending, c.turn, c._seq = self.intent, self.pending, self.turn, self._seq
        c.slots, c.parked = dict(self.slots), dict(self.parked)
        c.history = {k: list(v) for k, v in self.history.items()}
        c.corrections = list(self.corrections)
        c.intent_slots = dict(self.intent_slots)
        return c

    # ---- read side -----------------------------------------------------------
    def value(self, slot: str) -> Any:
        sv = self.slots.get(slot)
        return None if sv is None else sv.value

    def values(self) -> dict[str, Any]:
        return {k: v.value for k, v in self.slots.items()}

    def last_slot(self, typ: str | None = None) -> str | None:
        cands = [(sv.seq, k) for k, sv in self.slots.items() if typ is None or SLOT_TYPES.get(k) == typ]
        return max(cands)[1] if cands else None

    def holder(self, value: Any, typ: str) -> str | None:
        for k, sv in sorted(self.slots.items(), key=lambda kv: -kv[1].seq):
            if SLOT_TYPES.get(k) == typ and _same(sv.value, value):
                return k
        return None

    def past_values(self, slot: str) -> list[Any]:
        """Distinct values in the order first said, current one last."""
        out: list[Any] = []
        for sv in self.history.get(slot, []):
            if out and _same(out[-1], sv.value):
                continue
            out.append(sv.value)
        return out

    def snapshot(self) -> dict[str, Any]:
        return snapshot_payload(self.intent, self.values())

    def provenance(self) -> dict[str, dict[str, Any]]:
        return {k: {"value": v.value, "confidence": v.confidence, "source": v.source,
                    "t": v.t, "turn": v.turn, "via": v.via} for k, v in self.slots.items()}

    def low_confidence(self, threshold: float = ASR_MIN_CONFIDENCE) -> list[str]:
        return [k for k, v in self.slots.items() if v.confidence < threshold]

    def context(self) -> dict[str, Any]:
        return {"intent": self.intent, "slots": self.values(),
                "pending": self.pending.to_dict() if self.pending else None}

    # ---- write side ----------------------------------------------------------
    def set(self, slot: str, value: Any, confidence: float, source: str | None = None,
            t: float = 0.0, via: str = "rule") -> bool:
        old = self.slots.get(slot)
        if value is None:
            if old is None:
                return False
            del self.slots[slot]
            return True
        if old is not None and _same(old.value, value):
            if confidence > old.confidence:  # repeated: more certain, same provenance
                self.slots[slot] = SlotValue(old.value, round(confidence, 3), old.source, old.t, old.turn, old.via, old.seq)
            return False
        self._seq += 1
        sv = SlotValue(value, round(confidence, 3), source, t, self.turn, via, self._seq)
        self.slots[slot] = sv
        self.parked.pop(slot, None)
        hist = self.history.setdefault(slot, [])
        if not hist or not _same(hist[-1].value, value):
            hist.append(sv)
        if old is not None:
            self.corrections.append((slot, old.value, value, source))
        if self.pending is not None and self.pending.slot == slot:
            self.pending = None
        return True

    def set_intent(self, intent: str | None) -> set[str]:
        """Goal change keeps slots relevant to the new goal, parks the rest.
        Returns the parked slot names."""
        if intent == self.intent:
            return set()
        old, self.intent = self.intent, intent
        relevant = self.intent_slots.get(intent) if intent else None
        if old is None or relevant is None:
            return set()  # first goal, or unknown schema: keep everything
        keep = relevant | ALWAYS_KEEP
        parked = {k for k in self.slots if k not in keep}
        for k in parked:
            self.parked[k] = self.slots.pop(k)
        for k in [k for k in self.parked if k in keep and k not in self.slots]:
            self.slots[k] = self.parked.pop(k)
        if self.pending is not None and self.pending.slot in parked:
            self.pending = None
        return parked

    def register_tool(self, tool: ToolSpec, intent: str | None = None) -> None:
        """Unseen tool: its parameters (mapped to our slot names) become the
        relevance set for `intent` (default: the tool name)."""
        names = {slot_for_param(p) or p.name for p in tool.params}
        self.intent_slots[intent or tool.name] = frozenset(names)

    def apply(self, res: ParseResult, source_of: Callable[[str], tuple[str | None, float]],
              t: float = 0.0, final: bool = True) -> set[str]:
        """Apply parse events in order. Returns slots parked by goal changes."""
        parked: set[str] = set()
        if res.answered_pending:
            self.pending = None
        for ev in res.events:
            if isinstance(ev, SetIntent):
                parked |= self.set_intent(ev.intent)
            else:
                src, asr_conf = source_of(ev.surface or str(ev.value))
                self.set(ev.slot, ev.value, ev.confidence * asr_conf, src, t, ev.via)
                parked.discard(ev.slot)
        if final and res.clarification is not None:
            self.pending = res.clarification
        return parked


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b


# --------------------------------------------------------------------------
# Lexicons
# --------------------------------------------------------------------------
_CITY_ALIASES = {"bombay": "Mumbai", "madras": "Chennai", "calcutta": "Kolkata", "cochin": "Kochi",
                 "bangalore": "Bengaluru", "vizag": "Visakhapatnam", "nyc": "New York"}
_CITIES = (
    "delhi|new delhi|mumbai|pune|goa|bengaluru|chennai|kolkata|hyderabad|ahmedabad|jaipur|lucknow|kochi|"
    "trivandrum|thiruvananthapuram|srinagar|amritsar|chandigarh|indore|bhopal|nagpur|patna|varanasi|guwahati|"
    "bhubaneswar|coimbatore|mangalore|visakhapatnam|udaipur|leh|dehradun|ranchi|raipur|surat|vadodara|madurai|"
    "port blair|agra|shimla|jammu|london|paris|new york|dubai|singapore|tokyo|seoul|busan|bangkok|hong kong|"
    "sydney|frankfurt|amsterdam|san francisco|los angeles|chicago|toronto|abu dhabi|doha|kathmandu|colombo|"
    "kuala lumpur|berlin|rome|madrid|boston|seattle|beijing|shanghai|istanbul|zurich|munich|melbourne"
).split("|")
_CITY_WORDS = sorted(set(_CITIES) | set(_CITY_ALIASES), key=len, reverse=True)
CITY_RE = re.compile(r"\b(?:" + "|".join(re.escape(c) for c in _CITY_WORDS) + r")\b", re.I)

NUMS: dict[str, int] = {**{k: v for k, v in nlu.NUM_WORDS.items() if " " not in k},
                        "eleven": 11, "twelve": 12}
NUM = r"(?:\d{1,2}|" + "|".join(sorted(NUMS, key=len, reverse=True)) + r")"
_ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
         "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
         "eighteenth", "nineteenth", "twentieth"]
ORDINAL_WORDS: dict[str, int] = {w: i + 1 for i, w in enumerate(_ORDW)}
ORDINAL_WORDS.update({f"twenty {w}": 20 + i + 1 for i, w in enumerate(_ORDW[:9])})
ORDINAL_WORDS.update({f"twenty-{w}": 20 + i + 1 for i, w in enumerate(_ORDW[:9])})
ORDINAL_WORDS.update({"thirtieth": 30, "thirty first": 31, "thirty-first": 31})
_ORDW_RE = "|".join(sorted((re.escape(w) for w in ORDINAL_WORDS), key=len, reverse=True))
_MON = "|".join(list(nlu.MONTHS) + [m for m in nlu.MONTH_ABBR if m != "may"])
_WEEK = "|".join(nlu.WEEKDAYS)
_NOT_A_NAME = nlu.ENTITY_STOP | nlu.NOT_A_PLACE | {"calling", "trying", "looking", "flying", "going",
                                                     "travelling", "traveling", "here", "booking", "fine"}

# --------------------------------------------------------------------------
# Hesitation cleanup and repair segmentation
# --------------------------------------------------------------------------
_FILLER_RE = re.compile(r"(?<![\w'])(?:u+m+|u+h+m*|e+r+m+|e+r+|h+m+|a+h+|m{2,}|you know)(?![\w'])[,.]?\s*", re.I)
_TRUNC_RE = re.compile(r"\b\w+-(?=\s|,|$)")
_PAUSE_RE = re.compile(r"\.{2,}|…|\s-{1,2}\s|—")
_REPEAT_RE = re.compile(r"\b(\w+(?:\s+\w+){0,2})(?:\s*,?\s+\1\b)+", re.I)

_MARKERS: list[tuple[str, str]] = [
    ("repair", r"no\s*,?\s*wait"), ("repair", r"wait\s*,?\s*no"), ("repair", r"no\s*,?\s*no"),
    ("repair", r"i\s+meant?"), ("repair", r"or\s+rather"), ("repair", r"let\s+me\s+rephrase"),
    ("repair", r"on\s+second\s+thought"),
    ("discard", r"scratch\s+that"), ("discard", r"never\s*mind"), ("discard", r"forget\s+(?:it|that)"),
    ("make", r"make\s+(?:it|that)"), ("make", r"change\s+(?:it|that)\s+to"),
    ("repair", r"rather"), ("repair", r"sorry"), ("repair", r"actually"), ("repair", r"correction"),
    ("repair", r"(?:hang|hold)\s+on"), ("repair", r"oops"), ("repair", r"wait"),
]
_MARKER_RE = re.compile(
    "|".join(rf"(?P<m{i}>\b{rx}\b)" for i, (_, rx) in enumerate(_MARKERS))
    + r"|(?P<no>(?:^|(?<=[,.;!?])|(?<=[,.;!?]\s))\s*no\b(?!\s+(?:more|longer|need|thanks|problem|one|passengers?)\b))",
    re.I,
)


def clean(text: str) -> str:
    t = _PAUSE_RE.sub(", ", text)
    t = _FILLER_RE.sub("", t)
    t = _TRUNC_RE.sub("", t)
    t = re.sub(r"\s+", " ", t)
    t = _REPEAT_RE.sub(r"\1", t)
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    t = re.sub(r"([,.;])(?:\s*[,.;])+", r"\1", t)
    return t.strip(" ,;")


@dataclass
class Segment:
    kind: str | None  # None (first) | "repair" | "discard" | "make"
    text: str


def split_segments(text: str) -> list[Segment]:
    segs: list[Segment] = []
    pos, kind = 0, None
    for m in _MARKER_RE.finditer(text):
        segs.append(Segment(kind, text[pos:m.start()]))
        name = m.lastgroup or "no"
        kind = "repair" if name == "no" else _MARKERS[int(name[1:])][0]
        pos = m.end()
    segs.append(Segment(kind, text[pos:]))
    return segs


# --------------------------------------------------------------------------
# Mentions
# --------------------------------------------------------------------------


@dataclass
class Mention:
    type: str
    value: Any
    start: int
    end: int
    surface: str
    slot: str | None = None
    conf: float = CONF_CUE
    negated: bool = False
    delta: int | None = None  # relative passenger change
    backref: tuple[str, str] | None = None  # (kind, noun)


_SLOT_NOUNS: list[tuple[str, str]] = [
    ("origin", r"origin|departure city|departing city|source city|source|starting point|from city"),
    ("destination", r"destination|arrival city|to city"),
    ("return_date", r"return date|return"),
    ("date", r"departure date|travel date|flight date|date|day"),
    ("time", r"departure time|time"),
    ("passengers", r"number of passengers|passenger count|passengers|headcount"),
    ("name", r"passenger name|name"),
]
_NOUN_RE = [(s, re.compile(rf"\b(?:{rx})\b", re.I)) for s, rx in _SLOT_NOUNS]
_ORIGIN_CUES = {"from", "leaving", "departing", "ex"}
_DEST_CUES = {"to", "into", "towards", "for", "reach", "reaching"}
_NEG_RE = re.compile(r"(?:\bnot|\binstead\s+of|\brather\s+than)\s*(?:the\s+|to\s+|from\s+|on\s+)?$", re.I)
_BACKREF_RE = re.compile(
    r"\b(?:the\s+|my\s+)?(earlier|previous|prior|original|initial|old|older|other|former|first)\s+"
    r"(date|day|destination|origin|city|time|one|value|option)\b", re.I)
_BACKREF_NOUN = {"date": "date", "day": "date", "destination": "destination", "origin": "origin",
                 "time": "time", "city": "city"}


def _noun_slot(pre: str, typ: str, floor: int) -> str | None:
    """Nearest slot noun of the right type in `pre` after position `floor`."""
    best: tuple[int, str] | None = None
    for slot, rx in _NOUN_RE:
        if SLOT_TYPES[slot] != typ:
            continue
        for m in rx.finditer(pre):
            if m.start() >= floor and len(pre) - m.end() <= 30 and (best is None or m.start() > best[0]):
                best = (m.start(), slot)
    return best[1] if best else None


def _day_value(d: int, ref: date | None, month: str | None = None) -> str | None:
    if not 1 <= d <= 31:
        return None
    if month:
        return nlu.extract_date(f"{d} {month}", ref)
    if ref is None:
        suf = "th" if 10 <= d % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
        return f"{d}{suf}"
    y, m = ref.year, ref.month
    if d < ref.day:
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


_DATE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "nlu"),
    (re.compile(r"\bday after tomorrow\b|\btomorrow\b|\btoday\b|\btonight\b", re.I), "nlu"),
    (re.compile(rf"\b(?:(?:next|this|coming)\s+)?(?:{_WEEK})\b", re.I), "nlu"),
    (re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MON}|may)\b", re.I), "nlu"),
    (re.compile(rf"\b(?:{_MON}|may)\s+\d{{1,2}}(?:st|nd|rd|th)?\b", re.I), "nlu"),
    (re.compile(rf"\bthe\s+({_ORDW_RE})(?:\s+of\s+({_MON}|may))?\b"
                r"(?!\s+(?:one|option|flight|result|class|date|day|destination|origin|city|time|passenger))", re.I), "word"),
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\b", re.I), "digit"),
    (re.compile(r"\bon\s+the\s+(\d{1,2})\b(?!\s*(?:am|pm|a\.m|p\.m|:|st|nd|rd|th|passengers?|people))", re.I), "digit"),
]


def _dates(text: str, ref: date | None) -> list[Mention]:
    out: list[Mention] = []
    for rx, how in _DATE_PATTERNS:
        for m in rx.finditer(text):
            if how == "nlu":
                v = nlu.extract_date(m.group(0), ref)
            elif how == "word":
                v = _day_value(ORDINAL_WORDS[m.group(1).lower()], ref, m.group(2))
            else:
                v = _day_value(int(m.group(1)), ref)
            if v is not None:
                out.append(Mention("date", v, m.start(), m.end(), m.group(0)))
    return out


def _date_key(v: Any) -> tuple[Any, ...] | None:
    s = str(v).lower()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return ("iso", s)
    m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)", s)
    if m:
        return ("dom", int(m.group(1)))
    m = re.fullmatch(rf"({'|'.join(nlu.MONTHS)}) (\d{{1,2}})", s)
    if m:
        return ("md", nlu.MONTHS.index(m.group(1)), int(m.group(2)))
    rel = {"today": 0, "tomorrow": 1, "day after tomorrow": 2}
    return ("rel", rel[s]) if s in rel else None


def _earlier(a: Any, b: Any) -> bool | None:
    ka, kb = _date_key(a), _date_key(b)
    if ka is None or kb is None or ka[0] != kb[0]:
        return None
    return ka < kb


def _cities(seg: str, final: bool) -> list[Mention]:
    out: list[Mention] = []
    low = seg.lower()
    for m in re.finditer(r"\b(from|to|into|towards|leaving|departing|via|for)\s+", low):
        rest = seg[m.end():]
        words = nlu._words(rest)
        ent = nlu._entity_after(words, 0)
        if not ent:
            continue
        span = re.match(r"\W*" + r"\W+".join(re.escape(w) for w in ent.split()), rest, re.I)
        if span is None or span.start() != 0 or span.group(0)[:1] in (",", "."):
            continue
        s, e = m.end(), m.end() + span.end()
        known = CITY_RE.fullmatch(ent.lower()) is not None
        cue = m.group(1)
        if not known and (cue == "for" or (not final and e >= len(seg.rstrip()))):
            continue  # "for Rahul" is not a city; an unfinished chunk may be a partial word
        value = _CITY_ALIASES.get(ent.lower(), ent)
        slot = None if cue == "via" else ("origin" if cue in _ORIGIN_CUES else "destination")
        out.append(Mention("city", value, s, e, seg[s:e], slot=slot, conf=CONF_CUE,
                           negated=cue == "via"))  # "via X": consumed, never assigned
    for m in CITY_RE.finditer(seg):
        if any(x.start <= m.start() < x.end for x in out):
            continue
        name = m.group(0).lower()
        value = _CITY_ALIASES.get(name, nlu._title(name))
        slot = None
        if re.match(r"\s*(?:to|-|->|→)\s+\S", seg[m.end():]):
            slot = "origin"  # "Delhi to Mumbai"
        out.append(Mention("city", value, m.start(), m.end(), m.group(0), slot=slot,
                           conf=CONF_CUE if slot else CONF_DEFAULT))
    return out


def _counts(seg: str) -> list[Mention]:
    low = seg.lower()
    out: list[Mention] = []
    unit = r"(?:passengers?|people|persons?|adults?|travell?ers?|tickets?|seats?|pax)"

    def n(tok: str) -> int | None:
        tok = tok.lower()
        return int(tok) if tok.isdigit() else NUMS.get(tok, 1 if tok in ("a", "an", "another") else None)

    for m in re.finditer(rf"\b({NUM}|a|an|another)\s+more\s+{unit}\b|\badd\s+({NUM}|a|an|another)\s+(?:more\s+)?{unit}\b", low):
        out.append(Mention("count", None, m.start(), m.end(), m.group(0), "passengers", delta=n(m.group(1) or m.group(2))))
    for m in re.finditer(rf"\b(?:remove|drop|minus)\s+({NUM}|a|an)\s+{unit}\b|\b({NUM})\s+(?:less|fewer)\s+{unit}\b", low):
        out.append(Mention("count", None, m.start(), m.end(), m.group(0), "passengers", delta=-(n(m.group(1) or m.group(2)) or 1)))
    pats = [rf"\b({NUM})\s+(?:adult\s+)?{unit}\b", rf"\b({NUM})\s+of\s+us\b",
            rf"\bfor\s+({NUM})\b(?!\s*(?:am|pm|o'?clock|:|hours?|minutes?|days?|nights?|weeks?))",
            rf"\bpassengers?\s+(?:to|is|should be|as)\s+({NUM})\b"]
    for p in pats:
        for m in re.finditer(p, low):
            if any(x.start <= m.start() < x.end for x in out):
                continue
            v = n(m.group(1))
            if v:
                out.append(Mention("count", v, m.start(), m.end(), m.group(0), "passengers"))
    m = re.search(r"\b(?:just|only)\s+(?:me|myself)\b|\bsolo\b|\bby myself\b", low)
    if m:
        out.append(Mention("count", 1, m.start(), m.end(), m.group(0), "passengers"))
    return out


def _simple(seg: str) -> list[Mention]:
    """Single-slot types."""
    low = seg.lower()
    out: list[Mention] = []

    def add(typ: str, slot: str, v: Any, m: re.Match[str], g: int = 0) -> None:
        out.append(Mention(typ, v, m.start(g), m.end(g), m.group(g), slot))

    m = re.search(r"\b(?:my name is|my name's|name is|under the name(?: of)?|passenger name is|"
                  r"for passenger|this is|i am|i'm)\s+([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,2})", seg, re.I)
    if m:
        words: list[str] = []
        for w in m.group(1).split():
            if w.lower() in _NOT_A_NAME or CITY_RE.fullmatch(w):
                break
            words.append(w)
        cue = m.group(0)[: m.start(1) - m.start()].lower()
        strict = cue.strip() in ("this is", "i am", "i'm")
        if words and (not strict or words[0][0].isupper()):
            out.append(Mention("name", nlu._title(" ".join(words)), m.start(1), m.start(1) + len(" ".join(words)),
                               " ".join(words), "name"))
    m = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", seg)
    if m:
        add("email", "email", m.group(0).lower(), m)
    m = re.search(r"(?<![\w-])\+?\d[\d -]{8,}\d\b", seg)
    if m and len(re.sub(r"\D", "", m.group(0))) >= 10:
        add("phone", "phone", re.sub(r"[ -]", "", m.group(0)), m)
    m = re.search(r"\b(?:pnr|booking\s+(?:ref(?:erence)?|id|number|code)|confirmation\s+(?:code|number)|reference(?:\s+number)?)"
                  r"(?:\s+(?:is|number))?\s*:?\s*([A-Za-z0-9]{5,8})\b", seg, re.I)
    if m and m.group(1).lower() not in ("number", "please"):
        add("ref", "booking_ref", m.group(1).upper(), m, 1)
    m = re.search(r"\b(premium economy|economy|business|first)\s+class\b|\b(?:in|fly|flying)\s+(premium economy|economy|business)\b"
                  r"|\b(premium economy|economy)\b", low)
    if m:
        add("cabin", "cabin", (m.group(1) or m.group(2) or m.group(3)).replace(" ", "_"), m)
    m = re.search(r"\b(?:at|around|by|about)?\s*\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)", low)
    if m:
        h = int(m.group(1)) % 12 + (12 if m.group(3).startswith("p") else 0)
        add("time", "time", f"{h:02d}:{m.group(2) or '00'}", m)
    else:
        m = re.search(r"\bin\s+the\s+(morning|afternoon|evening)\b|\b(morning|afternoon|evening|night|red-eye)\s+flights?\b"
                      r"|\b(noon|midnight)\b", low)
        if m:
            v = m.group(1) or m.group(2) or {"noon": "12:00", "midnight": "00:00"}[m.group(3)]
            add("time", "time", v, m)
    m = re.search(r"\b(urgent|high|medium|low)\s+priority\b|\b(urgent(?:ly)?|asap|critical)\b", low)
    if m:
        add("priority", "priority", "high" if (m.group(2) or m.group(1) == "urgent") else m.group(1), m)
    for dev in sorted(nlu.DEVICES, key=len, reverse=True):
        m = re.search(r"\b" + re.escape(dev) + r"\b", low)
        if m:
            add("device", "device", dev, m)
            break
    m = re.search(r"\b(?:error|code)\s+([a-z]?\d+[a-z0-9]*|[a-z]{1,3}\d+)\b", low)
    if m:
        add("code", "error_code", m.group(1).upper(), m, 1)
    m = re.search(r"\b(?:the\s+)?(first|second|third|fourth|fifth|last|cheapest|earliest|fastest|latest)\s+(?:one|option|flight|result)\b"
                  rf"|\boption\s+({NUM})\b", low)
    if m:
        v = nlu.ORDINALS.get(m.group(1)) if m.group(1) else NUMS.get(m.group(2), int(m.group(2)) if m.group(2).isdigit() else None)
        if v is not None:
            add("choice", "choice", v, m)
    m = re.search(r"\b(?:ticket|complaint|case|request|issue)\s+(?:about|regarding|for|because|saying)\s+(.+?)\s*(?:[.!?]|$)"
                  r"|\b(?:problem|issue)\s+(?:is|with)\s+(.+?)\s*(?:[.!?]|$)", seg, re.I)
    if m:
        g = 1 if m.group(1) else 2
        v = re.sub(r"(?:[\s,]*\b(?:please|thanks|thank you|instead))+$", "", m.group(g), flags=re.I).strip(" ,")
        if v:
            add("text", "issue", v, m, g)
    return out


def _backrefs(seg: str) -> list[Mention]:
    out = []
    for m in _BACKREF_RE.finditer(seg):
        kind, noun = m.group(1).lower(), m.group(2).lower()
        if kind == "first" and noun in ("one", "option", "value"):
            continue  # "the first one": a result choice (or a clarification answer)
        out.append(Mention("backref", None, m.start(), m.end(), m.group(0), backref=(kind, noun)))
    return out


def extract(seg: str, ref: date | None = None, final: bool = True) -> list[Mention]:
    """Typed mentions in text order; overlapping lower-priority spans are dropped."""
    groups = [_backrefs(seg), _simple(seg), _counts(seg), _dates(seg, ref), _cities(seg, final)]
    out: list[Mention] = []
    for g in groups:  # priority order
        for m in g:
            if any(m.start < x.end and x.start < m.end for x in out):
                continue
            out.append(m)
    out.sort(key=lambda m: m.start)
    low = seg.lower()
    for i, m in enumerate(out):
        floor = out[i - 1].end if i else 0
        pre = low[:m.start]
        if _NEG_RE.search(pre[floor:]) or re.search(r"\binstead\s+of\s*$", pre):
            m.negated = True
        if m.type == "city":
            noun = _noun_slot(pre, "city", floor)
            if noun:
                if re.search(r"\bfrom\s*$", pre) and re.search(r"\b(?:origin|destination|city)\s+from\s*$", pre):
                    m.negated = True  # "change the destination from Mumbai to Pune": Mumbai is the old value
                m.slot, m.conf = noun, CONF_SLOT_NOUN
        elif m.type == "date":
            noun = _noun_slot(pre, "date", floor)
            if noun:
                m.slot, m.conf = noun, CONF_SLOT_NOUN
            elif re.search(r"\b(?:return|returning|back|coming back|come back)\b[^,.;]{0,12}$", pre):
                m.slot = "return_date"
            elif re.search(r"\b(?:depart|departing|departure|leaving|leave|outbound)\b[^,.;]{0,12}$", pre):
                m.slot = "date"
    return out


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


class SlotParser(Protocol):
    def parse(self, text: str, view: SlotMemory, final: bool = True) -> ParseResult: ...

    async def parse_async(self, text: str, view: SlotMemory, final: bool = True) -> ParseResult: ...


def _say(v: Any) -> str:
    s = str(v)
    return f"the {s}" if re.fullmatch(r"\d{1,2}(?:st|nd|rd|th)", s) else s


def _or(opts: Sequence[Any]) -> str:
    words = [_say(o) for o in opts]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " or " + words[-1]


class RuleParser:
    """Deterministic, synchronous. ref_date turns relative dates into ISO."""

    def __init__(self, ref_date: date | None = None) -> None:
        self.ref_date = ref_date

    async def parse_async(self, text: str, view: SlotMemory, final: bool = True) -> ParseResult:
        return self.parse(text, view, final)

    def parse(self, text: str, view: SlotMemory, final: bool = True) -> ParseResult:
        res = ParseResult()
        segs = split_segments(clean(text))
        mentions = [extract(s.text, self.ref_date, final) for s in segs]
        local: list[Assign] = []

        def emit(a: Assign) -> None:
            local.append(a)
            res.events.append(a)

        def current(slot: str) -> Any:
            for a in reversed(local):
                if a.slot == slot:
                    return a.value
            return view.value(slot)

        # 1. an answer to our pending clarification
        if view.pending is not None:
            ans = self._answer(view.pending, segs, mentions)
            if ans is not None:
                res.answered_pending = True
                emit(ans)
                mentions = [[m for m in ms if m.type not in ("backref", "choice")] for ms in mentions]

        anchors = [m for ms in mentions for m in ms if m.negated]
        for i, (seg, ms) in enumerate(zip(segs, mentions)):
            live = [m for m in ms if not m.negated]
            if seg.kind == "discard" and not live and not seg.text.strip(" ,.!?"):
                dropped = [a for a in local if a.seg == i - 1]
                res.events = [e for e in res.events if not (isinstance(e, Assign) and e in dropped)]
                local[:] = [a for a in local if a not in dropped]
                continue
            intent = nlu.detect_intent(seg.text)
            if intent:
                res.events.append(SetIntent(intent, i))
            repair = seg.kind in ("repair", "make", "discard")
            if repair and not live:
                fb = self._bare_fallback(seg, local, view, i)
                if fb is not None:
                    live = [fb]
            for m in live:
                if m.type == "backref":
                    out = self._backref(m, local, view, current)
                    if isinstance(out, Clarification):
                        if final:
                            res.clarification = out
                    elif out is not None:
                        emit(Assign(out[0], out[1], CONF_BACKREF, m.surface, seg=i))
                    continue
                if m.type == "count" and m.delta is not None:
                    base = current("passengers") or 1
                    emit(Assign("passengers", max(1, int(base) + m.delta), CONF_CUE, m.surface, seg=i))
                    continue
                slot, conf = m.slot, m.conf
                if slot is None:
                    slot, conf, clar = self._resolve(m, i, repair, anchors, local, view, current)
                    if clar is not None:
                        if final:
                            res.clarification = clar
                        continue
                    if slot is None:
                        continue
                emit(Assign(slot, m.value, conf, m.surface, seg=i))
        if res.clarification is None and not res.answered_pending and view.pending is not None and final:
            res.clarification = None  # leave the old question pending
        return res

    # ---- helpers ------------------------------------------------------------
    def _resolve(self, m: Mention, seg_i: int, repair: bool, anchors: list[Mention], local: list[Assign],
                 view: SlotMemory, current: Callable[[str], Any]) -> tuple[str | None, float, Clarification | None]:
        typ = m.type
        slots_of_type = [s for s, t in SLOT_TYPES.items() if t == typ]
        # value-anchored: "not Mumbai, Pune" / "Pune instead of Mumbai"
        for a in anchors:
            if a.type == typ and not _same(a.value, m.value):
                for a2 in reversed(local):
                    if SLOT_TYPES.get(a2.slot) == typ and _same(a2.value, a.value):
                        return a2.slot, CONF_ANCHOR, None
                h = view.holder(a.value, typ)
                if h:
                    return h, CONF_ANCHOR, None
        if repair:
            for a in reversed(local):
                if a.seg < seg_i and SLOT_TYPES.get(a.slot) == typ:
                    return a.slot, CONF_REPAIR, None
            last = view.last_slot(typ)
            if last:
                return last, CONF_REPAIR, None
        if len(slots_of_type) == 1:
            return slots_of_type[0], m.conf, None
        if typ == "date":
            return "date", 0.8, None
        if typ == "city":
            o, d = current("origin"), current("destination")
            if (o is not None and _same(o, m.value)) or (d is not None and _same(d, m.value)):
                return None, 0.0, None  # restated, nothing to change
            if d is None:
                return "destination", CONF_DEFAULT, None
            if o is None:
                return "origin", CONF_DEFAULT, None
            q = f"Should {m.value} be where you're flying from, or where you're going?"
            return None, 0.0, Clarification("role", None, q, ("origin", "destination"), value=m.value)
        return None, 0.0, None

    def _bare_fallback(self, seg: Segment, local: list[Assign], view: SlotMemory, seg_i: int) -> Mention | None:
        """'no, Shillong' / 'make it three' / 'sorry, Rohan': a bare value typed by
        whatever was said last."""
        text = seg.text.strip(" ,.!?")
        if not text or len(text.split()) > 3:
            return None
        prev = next((a.slot for a in reversed(local) if a.seg < seg_i), None) or view.last_slot()
        if seg.kind == "make" and re.fullmatch(NUM, text.lower()):
            prev = "passengers"
        typ = SLOT_TYPES.get(prev or "")
        if typ == "count" and re.fullmatch(NUM, text.lower()):
            v = int(text) if text.isdigit() else NUMS[text.lower()]
            return Mention("count", v, 0, len(text), text, "passengers", CONF_REPAIR)
        if typ in ("city", "name"):
            ent = nlu._entity_after(nlu._words(text), 0)
            if ent and (typ == "city" or ent.split()[0].lower() not in _NOT_A_NAME):
                return Mention(typ, _CITY_ALIASES.get(ent.lower(), ent), 0, len(text), text, None, CONF_REPAIR)
        return None

    def _backref(self, m: Mention, local: list[Assign], view: SlotMemory,
                 current: Callable[[str], Any]) -> tuple[str, Any] | Clarification | None:
        kind, noun = m.backref or ("", "")
        slot = _BACKREF_NOUN.get(noun)
        if noun in ("one", "value", "option"):
            slot = view.pending.slot if view.pending else (local[-1].slot if local else view.last_slot())
        if slot == "city":
            moved = [s for s in ("destination", "origin") if len(self._past(s, local, view)) > 1]
            if len(moved) != 1:
                q = "Which city do you want to change: where you're flying from, or where you're going?"
                return Clarification("role", None, q, ("origin", "destination"))
            slot = moved[0]
        if slot is None:
            return Clarification("missing", None, "Sorry, which detail do you want to change back?")
        past = self._past(slot, local, view)
        cur = current(slot)
        prior = [v for v in past if not _same(v, cur)]
        label = slot.replace("_", " ")
        if not prior:
            return Clarification("missing", slot, f"I don't have an earlier {label}. Which {label} would you like?",
                                 current=cur)
        if kind in ("previous", "prior", "former", "old", "older"):
            return slot, prior[-1]
        if kind in ("original", "initial", "first"):
            return (slot, past[0]) if not _same(past[0], cur) else None
        if kind == "other" and len(prior) == 1:
            return slot, prior[0]
        if kind == "earlier" and len(prior) == 1:
            if SLOT_TYPES[slot] != "date":
                return slot, prior[0]
            if cur is None or _earlier(prior[0], cur):
                return slot, prior[0]
            # said earlier, but not earlier in time: both readings are live
            q = f"Do you want {_say(prior[0])}, or keep {_say(cur)}?"
            return Clarification("value", slot, q, (prior[0], cur), current=cur)
        opts = tuple(dict.fromkeys(prior + ([cur] if cur is not None else [])))
        return Clarification("value", slot, f"Which {label} do you mean: {_or(opts)}?", opts, current=cur)

    @staticmethod
    def _past(slot: str, local: list[Assign], view: SlotMemory) -> list[Any]:
        past = view.past_values(slot)
        for a in local:
            if a.slot == slot and (not past or not _same(past[-1], a.value)):
                past.append(a.value)
        return past

    @staticmethod
    def _answer(p: Clarification, segs: list[Segment], mentions: list[list[Mention]]) -> Assign | None:
        text = " ".join(s.text for s in segs).lower()
        if p.kind == "role":
            if p.value is None:
                return None
            o = re.search(r"\b(?:origin|from|departure|departing|leaving|source|starting)\b", text)
            d = re.search(r"\b(?:destination|to|going|arrival|arriving|heading)\b", text)
            if o and (not d or o.start() < d.start()):
                return Assign("origin", p.value, CONF_CLARIFIED, str(p.value), "clarify")
            if d:
                return Assign("destination", p.value, CONF_CLARIFIED, str(p.value), "clarify")
            return None
        if p.kind != "value" or not p.options or p.slot is None:
            return None
        typ = SLOT_TYPES.get(p.slot)
        for ms in mentions:
            for m in ms:
                if m.type == typ and any(_same(m.value, o) for o in p.options):
                    return Assign(p.slot, m.value, CONF_CLARIFIED, m.surface, "clarify")
        pick: Any = None
        if re.search(r"\b(?:first|former|1st)\b", text):
            pick = p.options[0]
        elif re.search(r"\b(?:second|latter|2nd)\b", text) and len(p.options) > 1:
            pick = p.options[1]
        elif re.search(r"\b(?:last|third|3rd)\b", text) and len(p.options) > 2:
            pick = p.options[-1] if "last" in text else p.options[2]
        elif re.search(r"\b(?:keep|current|same|as is|no change|leave it)\b", text) and p.current is not None:
            pick = p.current
        elif re.search(r"\b(?:earlier|previous|older|old one|change it back)\b", text) and len(p.options) == 2 \
                and p.current is not None:
            pick = next(o for o in p.options if not _same(o, p.current))
        elif re.match(r"\s*(?:yes|yeah|yep|correct|right|sure)\b", text) and len(p.options) == 1:
            pick = p.options[0]
        if pick is None:
            return None
        return Assign(p.slot, pick, CONF_CLARIFIED, str(pick), "clarify")


class HybridParser(RuleParser):
    """Rules first. The LLM (plugins.LLMPlugin) is consulted only on the slow
    path, only when the rules produced nothing, and only within its timeout."""

    def __init__(self, llm: LLMPlugin | None = None, ref_date: date | None = None,
                 tools: Sequence[ToolSpec] = (), timeout_s: float = LLM_TIMEOUT_S) -> None:
        super().__init__(ref_date)
        self.llm = llm
        self.tools = list(tools)
        self.timeout_s = timeout_s

    async def parse_async(self, text: str, view: SlotMemory, final: bool = True) -> ParseResult:
        res = self.parse(text, view, final)
        if self.llm is None or not final or not res.empty or not clean(text):
            return res
        out = await llm_parse(self.llm, text, view.snapshot(), self.tools, self.timeout_s)
        return self._from_llm(out, view) or res

    @staticmethod
    def _from_llm(out: Mapping[str, Any] | None, view: SlotMemory) -> ParseResult | None:
        if not out:
            return None
        res = ParseResult()
        intent = out.get("intent")
        if isinstance(intent, str) and intent:
            res.events.append(SetIntent(intent))
        known = set(SLOT_TYPES) | {s for v in view.intent_slots.values() for s in v}
        confs = out.get("confidence") if isinstance(out.get("confidence"), Mapping) else {}
        for k, v in (out.get("slots") or {}).items():
            if k not in known or not isinstance(v, (str, int, float, bool)) or v == "":
                continue
            c = confs.get(k, LLM_MAX_CONF) if isinstance(confs.get(k, 0), (int, float)) else LLM_MAX_CONF
            res.events.append(Assign(k, v, min(float(c), LLM_MAX_CONF), str(v), "llm"))
        return res if res.events else None


# --------------------------------------------------------------------------
# Session facade: chunks in, updates out
# --------------------------------------------------------------------------


class SlotTracker:
    """Feed transcript chunks; each call re-parses the turn so far against the
    state at turn start, so a repair in a later chunk ("...no wait, Pune")
    rewrites only what it targets, and partial chunks never commit guesses."""

    def __init__(self, parser: SlotParser | None = None, ref_date: date | None = None,
                 intent_slots: Mapping[str, Iterable[str]] | None = None) -> None:
        self.parser: SlotParser = parser or RuleParser(ref_date)
        self.memory = SlotMemory(intent_slots)
        self._baseline: SlotMemory | None = None
        self._buf = ""
        self._chunks: list[tuple[str | None, str, float]] = []  # (id, text, asr confidence)

    # ---- properties -------------------------------------------------------
    @property
    def intent(self) -> str | None:
        return self.memory.intent

    @property
    def pending(self) -> Clarification | None:
        return self.memory.pending

    def snapshot(self) -> dict[str, Any]:
        return self.memory.snapshot()

    # ---- input --------------------------------------------------------------
    def feed(self, text: str, chunk_id: str | None = None, t: float = 0.0, end_of_turn: bool = True,
             confidence: float = 1.0) -> SlotUpdate:
        self._start(text, chunk_id, confidence)
        res = self.parser.parse(self._buf, self._baseline, final=end_of_turn)  # type: ignore[arg-type]
        return self._finish(res, t, end_of_turn)

    async def feed_async(self, text: str, chunk_id: str | None = None, t: float = 0.0, end_of_turn: bool = True,
                         confidence: float = 1.0) -> SlotUpdate:
        """Slow-path variant: lets a HybridParser fall back to its LLM."""
        self._start(text, chunk_id, confidence)
        res = await self.parser.parse_async(self._buf, self._baseline, final=end_of_turn)  # type: ignore[arg-type]
        return self._finish(res, t, end_of_turn)

    def end_turn(self, t: float = 0.0) -> SlotUpdate:
        """End-of-turn marker without new text."""
        if self._baseline is None:
            return self._diff(self.memory, self.memory, self.memory, set(), True)
        return self.feed("", None, t, True)

    def set_intent(self, intent: str | None) -> SlotUpdate:
        """External goal decision (e.g. manifest-driven unseen tool)."""
        before = self.memory.clone()
        parked = self.memory.set_intent(intent)
        if self._baseline is not None:
            self._baseline.set_intent(intent)
        return self._diff(before, self.memory, before, parked, True)

    def set_slot(self, slot: str, value: Any, confidence: float = 1.0, source: str | None = None,
                 t: float = 0.0) -> SlotUpdate:
        """External write (tool-derived id, perception). Survives the open turn."""
        before = self.memory.clone()
        self.memory.set(slot, value, confidence, source, t, "external")
        if self._baseline is not None:
            self._baseline.set(slot, value, confidence, source, t, "external")
        return self._diff(before, self.memory, before, set(), True)

    def register_tools(self, tools: Iterable[ToolSpec]) -> None:
        for tool in tools:
            for mem in filter(None, (self.memory, self._baseline)):
                if tool.name not in mem.intent_slots:
                    mem.register_tool(tool)

    def reset(self) -> None:
        """Session end. Nothing survives."""
        self.memory = SlotMemory({k: v for k, v in self.memory.intent_slots.items()})
        self._baseline, self._buf, self._chunks = None, "", []

    # ---- internals ------------------------------------------------------------
    def _start(self, text: str, chunk_id: str | None, confidence: float) -> None:
        if self._baseline is None:
            self._baseline = self.memory.clone()
            self._buf, self._chunks = "", []
        if text:
            if self._buf and not self._buf[-1].isspace() and not text[0].isspace():
                self._buf += " "
            self._buf += text
            self._chunks.append((chunk_id, text, max(0.0, min(1.0, confidence))))

    def _source_of(self, surface: str) -> tuple[str | None, float]:
        s = surface.lower().strip()
        for cid, txt, conf in reversed(self._chunks):
            if s and s in txt.lower():
                return cid, conf
        if self._chunks:
            return self._chunks[-1][0], self._chunks[-1][2]
        return None, 1.0

    def _finish(self, res: ParseResult, t: float, final: bool) -> SlotUpdate:
        base = self._baseline
        assert base is not None
        work = base.clone()
        parked = work.apply(res, self._source_of, t, final)
        before, self.memory = self.memory, work
        upd = self._diff(before, work, base, parked, final)
        if final:
            self.memory.turn += 1
            self._baseline, self._buf, self._chunks = None, "", []
        return upd

    @staticmethod
    def _diff(before: SlotMemory, after: SlotMemory, base: SlotMemory, parked: set[str], final: bool) -> SlotUpdate:
        keys = set(before.slots) | set(after.slots)
        changed = {k: after.value(k) for k in keys if not _same(before.value(k), after.value(k))
                   or (k in before.slots) != (k in after.slots)}
        corrected = {k for k in changed if base.value(k) is not None and after.value(k) is not None
                     and not _same(base.value(k), after.value(k))}
        clar = after.pending if after.pending is not None and after.pending is not before.pending \
            and after.pending != before.pending else None
        return SlotUpdate(after.intent, before.intent != after.intent, changed, corrected,
                          {k for k in parked if k in base.slots or k in before.slots}, clar, after.snapshot(), final,
                          {k: before.value(k) for k in changed})
