# /mnt/project-files/theme5/theme5/planner.py
"""Schema-driven planner. Maps intent + slots onto whatever tools the scenario
manifest declares (including unseen ones), chains read-only prerequisites to
obtain ids, and asks for missing required slots. Deterministic and fast."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .protocol import DEFAULT_CHOICE, ParamSpec, ToolSpec
from .state import SessionState
from .tools import ToolRegistry, canonical_args

SLOT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "origin": ("origin", "from", "source", "src", "departure", "departure_city", "from_city", "origin_city",
               "start", "start_location", "departure_airport", "from_airport", "origin_airport"),
    "destination": ("destination", "to", "dest", "dst", "arrival", "arrival_city", "to_city", "destination_city",
                    "end", "end_location", "target", "arrival_airport", "to_airport", "destination_airport"),
    "location": ("location", "city", "place", "address", "area", "region", "where"),
    "date": ("date", "departure_date", "travel_date", "day", "when", "depart_date", "flight_date"),
    "passengers": ("passengers", "pax", "num_passengers", "passenger_count", "travelers", "travellers",
                   "count", "seats", "num_seats", "adults", "party_size", "num_travelers"),
    "cabin": ("cabin", "class", "cabin_class", "seat_class", "fare_class", "travel_class"),
    "time": ("time", "departure_time", "hour", "preferred_time"),
    "name": ("name", "passenger_name", "customer_name", "full_name", "user_name", "contact_name", "passenger"),
    "email": ("email", "contact_email", "email_address"),
    "phone": ("phone", "phone_number", "mobile", "contact_number"),
    "issue": ("issue", "description", "problem", "summary", "details", "title", "subject", "complaint",
              "issue_description"),
    "query": ("query", "question", "q", "search", "text", "topic", "search_query"),
    "device": ("device", "product", "model", "appliance", "device_type", "product_name", "device_model"),
    "error_code": ("error_code", "code", "error"),
    "frame": ("frame", "frame_id", "image", "image_id", "image_ref", "photo", "frame_ref", "picture", "image_path"),
    "priority": ("priority", "severity", "urgency"),
    "booking_ref": ("booking_ref", "booking_id", "booking_reference", "pnr", "reference", "confirmation",
                    "confirmation_code", "confirmation_id", "reservation_id"),
    "flight_id": ("flight_id", "flight", "flight_number", "flight_no"),
}
_PARAM_TO_SLOT = {syn: slot for slot, syns in SLOT_SYNONYMS.items() for syn in syns}
SLOT_FALLBACK = {"location": "destination", "destination": "location"}

INTENT_HINTS: dict[str, tuple[tuple[str, ...], bool | None]] = {
    # intent: (name tokens, expects state-modifying?)
    "search_flights": (("search", "find", "flight", "flights", "list", "query"), False),
    "book_flight": (("book", "booking", "reserve", "reservation", "flight"), True),
    "cancel_booking": (("cancel", "booking", "reservation"), True),
    "create_ticket": (("ticket", "create", "support", "issue", "case", "complaint"), True),
    "lookup_manual": (("manual", "lookup", "guide", "frame", "docs", "troubleshoot"), False),
    "navigate": (("route", "navigate", "navigation", "directions", "calculate"), None),
}

CLARIFY_TEXT = {
    "origin": "Where are you flying from?",
    "destination": "Where would you like to go?",
    "date": "What date should I look at?",
    "passengers": "How many passengers?",
    "name": "What name should I put it under?",
    "email": "What email should I use?",
    "phone": "What phone number should I use?",
    "issue": "Can you describe the problem?",
    "device": "Which device is this about?",
    "query": "What would you like to know?",
    "frame": "Could you show me the device on camera?",
    "location": "Which city?",
}

_STOP = {"a", "an", "the", "to", "of", "for", "and", "or", "in", "on", "by", "with", "get", "tool", "api", "given", "user"}


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def tokens(s: str) -> set[str]:
    return {_stem(w) for w in re.split(r"[^a-z0-9]+", s.lower()) if w and w not in _STOP}


# Word-level cues for parameter names/descriptions the synonym table misses
# (unseen tools: venueSlug, party_sz, when_iso, "Number of guests").
_ABBR = {"sz": "size", "qty": "quantity", "num": "number", "no": "number", "dt": "date", "ts": "time",
         "loc": "location", "addr": "address", "tel": "phone", "mob": "mobile", "desc": "description",
         "msg": "message", "cnt": "count", "ppl": "people"}
_EXTRA_WORDS = {
    "passengers": ("guest", "people", "person", "party", "headcount", "attendee", "diner", "occupant"),
    "location": ("venue", "restaurant", "hotel", "property", "store", "shop", "branch", "clinic", "outlet"),
    "date": ("day", "when", "checkin"),
    "time": ("hour", "clock"),
    "issue": ("message", "complaint"),
    "query": ("keyword", "term"),
}
_WORD_TO_SLOT: dict[str, str] = {}
for _slot, _syns in SLOT_SYNONYMS.items():
    for _w in _syns:
        if "_" not in _w and _w not in _STOP:
            _WORD_TO_SLOT.setdefault(_stem(_w), _slot)
for _slot, _ws in _EXTRA_WORDS.items():
    for _w in _ws:
        _WORD_TO_SLOT.setdefault(_stem(_w), _slot)


def _words(s: str) -> list[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    ws = [w for w in re.split(r"[^a-z0-9]+", s.lower()) if w]
    return [_ABBR.get(w, w) for w in ws]


def slot_for_param(p: ParamSpec) -> str | None:
    n = p.name.lower()
    if n in SLOT_SYNONYMS:
        return n
    if n in _PARAM_TO_SLOT:
        return _PARAM_TO_SLOT[n]
    if n.endswith("_id") or p.name.endswith("Id"):
        return None
    ws = _words(p.name)
    joined = "_".join(ws)
    if joined in SLOT_SYNONYMS:
        return joined
    if joined in _PARAM_TO_SLOT:
        return _PARAM_TO_SLOT[joined]
    for w in reversed(ws):  # head noun last: guest_name -> name, venue_slug -> location
        if _stem(w) in _WORD_TO_SLOT:
            return _WORD_TO_SLOT[_stem(w)]
    for w in _words(p.description or ""):  # "Restaurant name or slug", "Number of guests"
        if w not in _STOP and _stem(w) in _WORD_TO_SLOT:
            return _WORD_TO_SLOT[_stem(w)]
    return None


def _wants_datetime(p: ParamSpec) -> bool:
    text = f"{p.name} {p.description or ''}".lower()
    return "datetime" in text.replace("_", "").replace("-", "") or "iso" in text or "time" in text \
        or p.name.lower() in ("when", "at", "start")


def coerce(p: ParamSpec, v: Any) -> Any:
    if v is None:
        return None
    if p.type in ("integer", "int"):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    if p.type in ("number", "float"):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return int(f) if f.is_integer() and not isinstance(v, float) else f
    if p.type in ("boolean", "bool"):
        return bool(v)
    if p.enum:
        for e in p.enum:
            if str(e).lower() == str(v).lower() or str(v).lower() in str(e).lower():
                return e
        return None
    return v if isinstance(v, (str, list, dict)) else str(v)


def fill_args(tool: ToolSpec, slots: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    args: dict[str, Any] = {}
    missing: list[str] = []
    mapped = {p.name: (p.name if p.name in slots else slot_for_param(p)) for p in tool.params}
    time_param = any(s == "time" for s in mapped.values())
    for p in tool.params:
        v = slots.get(p.name)
        s = mapped[p.name]
        if v is None:
            v = slots.get(s) if s else None
            if v is None and s in SLOT_FALLBACK:
                v = slots.get(SLOT_FALLBACK[s])
        if s == "date" and not time_param and slots.get("time") and isinstance(v, str) \
                and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v) and _wants_datetime(p):
            v = f"{v}T{slots['time']}"  # one ISO date-time param carries both slots
        v = coerce(p, v)
        if v is None:
            if p.required:
                missing.append(p.name)
        else:
            args[p.name] = v
    return args, missing


# ---- schema-guided extraction (unseen tools) ---------------------------------
# Values an unseen tool needs that no known slot carries (awb_no with a
# pattern, an enum zone, a currency pair, a licence number) are read straight
# from the utterance using the parameter's own schema.

_CONCEPTS = {
    "dollar": "currency", "rupee": "currency", "euro": "currency", "pound": "currency", "yen": "currency",
    "dirham": "currency", "exchange": "currency", "convert": "conversion", "parcel": "courier",
    "package": "courier", "shipment": "courier", "degree": "temperature", "warmer": "temperature",
    "cooler": "temperature", "thermostat": "temperature", "heating": "temperature", "ac": "temperature",
}
_CURRENCY_RE = re.compile(r"\b(us dollars?|dollars?|usd|\$|rupees?|rs\.?|inr|euros?|eur|pounds?|gbp|yen|jpy|"
                          r"dirhams?|aed|singapore dollars?|sgd)\b", re.I)
_CURRENCY = {"us dollar": "USD", "dollar": "USD", "usd": "USD", "$": "USD", "rupee": "INR", "rs": "INR",
             "inr": "INR", "euro": "EUR", "eur": "EUR", "pound": "GBP", "gbp": "GBP", "yen": "JPY", "jpy": "JPY",
             "dirham": "AED", "aed": "AED", "singapore dollar": "SGD", "sgd": "SGD"}
_SOURCE_WORDS = {"frm", "from", "source", "src", "base", "sell"}
_TARGET_WORDS = {"to", "target", "dest", "destination", "quote", "buy", "into"}
_ID_WORDS = {"number", "no", "id", "code", "licence", "license", "ref", "reference", "awb", "waybill", "account",
             "serial", "pnr", "plate", "tracking"}
_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "a": 1}
_NUM_RE = re.compile(r"(?<![\w.:-])(\d+(?:\.\d+)?)(?![\w:-]|\.\d)(?!\s*(?:am|pm|st|nd|rd|th)\b)", re.I)
_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"


def concept_tokens(text: str) -> set[str]:
    toks = tokens(text)
    return toks | {_CONCEPTS[t] for t in toks if t in _CONCEPTS}


def _collapse_digits(text: str) -> str:
    return re.sub(r"(?<=\d)[ -](?=\d)", "", text)


def _currency_key(m: str) -> str:
    k = m.lower().rstrip(".")
    return k[:-1] if k.endswith("s") and k != "rs" else k


def _currency_param(p: ParamSpec) -> bool:
    t = f"{p.name} {p.description or ''}".lower()
    return "ccy" in t or "currency" in t or "4217" in t


def _numbers(text: str) -> list[float]:
    text = re.sub(rf"\b{_MONTHS}\s+\d+|\d+\s+{_MONTHS}", " ", text, flags=re.I)  # dates are not amounts
    return [float(m) for m in _NUM_RE.findall(text)]


def schema_values(tool: ToolSpec, text: str, taken: set[str] | None = None) -> dict[str, Any]:
    """Values for `tool`'s parameters found in `text` by schema shape alone.
    Skips params listed in `taken`. Returns {param_name: value}."""
    taken = set(taken or ())
    out: dict[str, Any] = {}
    flat = _collapse_digits(text)
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", flat)
    used: set[str] = set()
    # currencies, by role
    cps = [p for p in tool.params if p.name not in taken and _currency_param(p)]
    if cps:
        codes = [c for c in (_CURRENCY.get(_currency_key(m)) for m in _CURRENCY_RE.findall(text)) if c]
        role = {p.name: (0 if set(_words(p.name + " " + (p.description or ""))) & _SOURCE_WORDS
                         else 1 if set(_words(p.name + " " + (p.description or ""))) & _TARGET_WORDS else None)
                for p in cps}
        for i, p in enumerate(cps):
            idx = role[p.name] if role[p.name] is not None else i
            if idx < len(codes):
                out[p.name] = codes[idx]
    nums = _numbers(flat)
    for p in tool.params:
        if p.name in taken or p.name in out:
            continue
        pat = p.pattern
        if p.enum:
            if all(isinstance(e, (int, float)) and not isinstance(e, bool) for e in p.enum):
                cand = nums + [_WORD_NUM[w.lower()] for w in words if w.lower() in _WORD_NUM and w.lower() != "a"]
                v = next((e for n in cand for e in p.enum if float(e) == float(n)), None)
            else:
                v = next((e for e in p.enum if isinstance(e, str)
                          and re.search(rf"\b{re.escape(e)}\b", text, re.I)), None)
            if v is not None:
                out[p.name] = v
            continue
        if pat:
            try:
                rx = re.compile(pat)
            except re.error:
                rx = None
            v = next((w for w in words if rx and rx.fullmatch(w) and w not in used), None) if rx else None
            if v is not None:
                out[p.name] = v
                used.add(v)
            continue
        if p.type in ("integer", "int", "number", "float") and not _currency_param(p):
            if nums:
                n = nums.pop(0)
                out[p.name] = int(n) if p.type in ("integer", "int") or n.is_integer() else n
            continue
        if p.type in ("string", "str") and set(_words(p.name)) & _ID_WORDS:
            ids = [w for w in words if w not in used and len(w) >= 6 and re.search(r"\d", w)
                   and (re.fullmatch(r"\d+", w) or re.fullmatch(r"[A-Z0-9-]+", w))]
            if len(ids) == 1:
                out[p.name] = ids[0]
                used.add(ids[0])
    return out



def score_tool(tool: ToolSpec, hint_tokens: set[str], expect_write: bool | None) -> float:
    name_t = tokens(tool.name)
    desc_t = tokens(tool.description)
    s = 3.0 * len(hint_tokens & name_t) + 1.0 * len(hint_tokens & desc_t)
    if s and expect_write is not None:
        s += 1.5 if tool.state_modifying == expect_write else -3.0
    return s


def goal_tool(intent: str | None, utterance: str, registry: ToolRegistry) -> ToolSpec | None:
    if intent and registry.get(intent):
        return registry.get(intent)
    tools = registry.all()
    if not tools:
        return None
    if intent in INTENT_HINTS:
        hint, expect = INTENT_HINTS[intent]
        htoks = {_stem(h) for h in hint}
        best = max(tools, key=lambda t: score_tool(t, htoks, expect))
        if score_tool(best, htoks, expect) >= 3.0:
            return best
    # unseen tools: match the utterance (plus concept words) against names and
    # descriptions; a tool whose required params the utterance fully supplies
    # by schema shape gets extra evidence.
    utoks = concept_tokens(utterance)

    def total(t: ToolSpec) -> float:
        req = [p.name for p in t.params if p.required]
        bonus = 2.0 if req and set(req) <= set(schema_values(t, utterance)) else 0.0
        return score_tool(t, utoks, None) + bonus
    best = max(tools, key=total)
    return best if total(best) >= 3.0 else None


def result_items(result: Any) -> list[Any]:
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for v in result.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        return [result]
    return []


def _num_field(item: dict[str, Any], keys: tuple[str, ...]) -> float:
    for k, v in item.items():
        if any(x in k.lower() for x in keys) and isinstance(v, (int, float)):
            return float(v)
    return float("inf")


def choose(items: list[Any], choice: Any) -> Any:
    if not items:
        return None
    if choice == "cheapest":
        return min(items, key=lambda i: _num_field(i, ("price", "fare", "cost", "amount")) if isinstance(i, dict) else 0)
    if choice in ("earliest", "fastest"):
        keys = ("depart", "time", "start") if choice == "earliest" else ("duration",)
        return min(items, key=lambda i: str(next((v for k, v in i.items() if any(x in k.lower() for x in keys)), "~")) if isinstance(i, dict) else "")
    if choice == "latest":
        return items[-1]
    idx = choice if isinstance(choice, int) else DEFAULT_CHOICE
    if idx == -1:
        return items[-1]
    return items[idx - 1] if 1 <= idx <= len(items) else None


def extract_value(item: Any, param: str) -> Any:
    if not isinstance(item, dict):
        return None
    if param in item:
        return item[param]
    if param.endswith("_id") and "id" in item:
        return item["id"]
    base = param[:-3] if param.endswith("_id") else param
    for k, v in item.items():
        if k.lower().endswith("_id") and base in k.lower():
            return v
    return None


@dataclass
class Step:
    kind: str  # "call" | "clarify" | "done" | "idle" | "unsupported" | "choose_failed"
    tool: ToolSpec | None = None
    args: dict[str, Any] = field(default_factory=dict)
    purpose: str = "goal"
    slot: str | None = None
    text: str = ""
    derived: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResultStore:
    """Latest successful, non-stale result per tool, keyed by the args it ran with."""
    by_tool: dict[str, tuple[str, Any]] = field(default_factory=dict)

    def put(self, tool: str, args: dict[str, Any], result: Any) -> None:
        self.by_tool[tool] = (canonical_args(args), result)

    def get_valid(self, tool: ToolSpec, slots: dict[str, Any]) -> Any:
        ent = self.by_tool.get(tool.name)
        if ent is None:
            return None
        args, missing = fill_args(tool, slots)
        if missing or canonical_args(args) != ent[0]:
            return None
        return ent[1]

    def has(self, tool: str, args: dict[str, Any]) -> bool:
        ent = self.by_tool.get(tool)
        return ent is not None and ent[0] == canonical_args(args)


def next_step(state: SessionState, registry: ToolRegistry, results: ResultStore, utterance: str = "") -> Step:
    if state.intent is None:
        return Step("idle")
    goal = goal_tool(state.intent, utterance, registry)
    if goal is None:
        return Step("unsupported", text="I can't do that with the tools available here.")
    slots = dict(state.slots)
    derived: dict[str, Any] = {}

    args, missing = fill_args(goal, slots)
    if missing:
        # try to derive missing ids from a valid prerequisite result
        for r in registry.all():
            if r.name == goal.name or r.state_modifying:
                continue
            res = results.get_valid(r, slots)
            if res is None:
                continue
            item = choose(result_items(res), slots.get("choice"))
            if item is None:
                return Step("choose_failed", text="I couldn't find that option in the results. Which one would you like?", slot="choice")
            for m in missing:
                v = extract_value(item, m)
                if v is not None:
                    derived[m] = v
        slots.update(derived)
        args, missing = fill_args(goal, slots)

    if not missing:
        if results.has(goal.name, args):
            return Step("done", tool=goal, args=args, derived=derived)
        return Step("call", tool=goal, args=args, purpose="goal", derived=derived)

    # chain: a related read-only tool we can run now to obtain the missing values
    gtoks = tokens(goal.name) | tokens(goal.description)
    cands = []
    for r in registry.all():
        if r.name == goal.name or r.state_modifying:
            continue
        rargs, rmissing = fill_args(r, slots)
        if rmissing or results.has(r.name, rargs):
            continue
        rel = len(gtoks & (tokens(r.name) | tokens(r.description)))
        if rel:
            cands.append((rel, r, rargs))
    if cands:
        cands.sort(key=lambda c: -c[0])
        _, r, rargs = cands[0]
        return Step("call", tool=r, args=rargs, purpose="prereq", derived=derived)

    # ask for the first missing slot that a user can actually supply
    for m in missing:
        p = goal.param(m)
        slot = slot_for_param(p) if p else None
        if slot or not m.endswith("_id"):
            key = slot or m
            return Step("clarify", slot=key, text=CLARIFY_TEXT.get(key, f"What {m.replace('_', ' ')} should I use?"), derived=derived)
    return Step("clarify", slot=missing[0], text=f"I need the {missing[0].replace('_', ' ')} to continue.", derived=derived)
