# /mnt/project-files/theme5/theme5/nlu.py
"""Rule-based fast-path understanding: disfluency cleanup, self-repair,
cancellation, intent keywords and slot extraction. Pure functions, no I/O,
microseconds per utterance."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "a couple of": 2, "single": 1,
}
ORDINALS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "last": -1,
    "cheapest": "cheapest", "earliest": "earliest", "fastest": "fastest", "latest": "latest",
}
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december")
MONTH_ABBR = {m[:3]: i + 1 for i, m in enumerate(MONTHS)}
DEVICES = ("tv", "television", "washing machine", "washer", "dryer", "fridge", "refrigerator",
           "phone", "ac", "air conditioner", "oven", "microwave", "dishwasher", "router",
           "printer", "car", "laptop", "monitor", "soundbar", "vacuum", "watch")

FILLERS_RE = re.compile(r"\b(?:u+m+|u+h+|e+r+m*|h+m+|a+h+|you know|kind of|sort of)\b[,.]?\s*", re.I)
STUTTER_RE = re.compile(r"\b((?!\d+\b)\w+)(?:\s+\1\b)+", re.I)  # digits exempt: "5 3 3 5" is a number
REPAIR_RE = re.compile(
    r"(?:\bno\s*,?\s*wait\b|\bwait\s*,?\s*no\b|\bno\s*,\s*no\b|\bactually\b|\bsorry\b|\bi\s+mean\b|\bi\s+meant\b"
    r"|\brather\b|\bscratch\s+that\b|\bcorrection\b|\bmake\s+(?:it|that)\b|\bchange\s+(?:it|that)\s+to\b"
    r"|\binstead\b|\bon\s+second\s+thought\b|\bno\s*,)",
    re.I,
)
CANCEL_RE = re.compile(
    r"\b(?:cancel(?:\s+(?:that|it|this|everything|the\s+search|the\s+request))?|never\s*mind|forget\s+(?:it|that|about\s+it)"
    r"|stop(?:\s+(?:that|it))?|abort|don'?t\s+bother|skip\s+it)\b",
    re.I,
)
THANKS_RE = re.compile(r"\b(?:thanks|thank you|cheers|great|perfect)\b", re.I)
AFFIRM_RE = re.compile(r"^\s*(?:yes|yeah|yep|yup|sure|correct|right|ok(?:ay)?|go ahead|confirm(?:ed)?|please do|do it)\b", re.I)
DENY_RE = re.compile(r"^\s*(?:no|nope|nah|not really|don'?t)\b", re.I)
NEGATED_RE = re.compile(r"\b(?:not|instead\s+of)\s+(?:to\s+|from\s+|on\s+)?([a-z][a-z'-]*(?:\s+[a-z][a-z'-]*)?)", re.I)

ENTITY_STOP = {
    "on", "for", "at", "tomorrow", "today", "tonight", "next", "this", "in", "with", "and", "please",
    "by", "around", "departing", "leaving", "returning", "then", "via", "passengers", "passenger",
    "people", "adults", "i", "but", "not", "economy", "business", "first", "premium", "is", "are",
    "flight", "flights", "instead", "rather", "actually", "sorry", "so", "the", "a", "an", "it",
    "that", "me", "we", "us", "my", "from", "to", "no", "wait", "or", "departure", "date",
    "morning", "evening", "afternoon", "night", "one", "two", "three", "four", "five", "class",
} | set(WEEKDAYS) | set(MONTHS)
NOT_A_PLACE = {
    "book", "find", "search", "go", "fly", "travel", "change", "make", "be", "get", "see", "know",
    "do", "have", "help", "check", "look", "cancel", "reset", "fix", "use", "turn", "set", "open",
    "create", "file", "raise", "report", "talk", "speak", "ask", "buy", "pay", "leave", "come",
    "start", "stop", "reserve", "navigate", "drive", "head", "show", "tell", "keep", "want",
}

DOMAIN_OF_INTENT = {
    "search_flights": "flight", "book_flight": "flight", "cancel_booking": "flight",
    "create_ticket": "support", "lookup_manual": "manual", "navigate": "navigation",
}


@dataclass
class Parse:
    raw: str
    clean: str
    intent: str | None = None
    slots: dict[str, Any] = field(default_factory=dict)
    correction: bool = False
    cancel: bool = False
    affirm: bool = False
    deny: bool = False


def clean_text(text: str) -> str:
    t = FILLERS_RE.sub("", text)
    t = re.sub(r"\b(\w+)-\s+", r"\1 ", t)  # "Del- Delhi" style cut-offs
    t = STUTTER_RE.sub(r"\1", t)
    # A-B-A reverts: "Kolkata was right", "Goa is correct after all" -> the value alone
    t = re.sub(r"\s+(?:was|is|'s)\s+(?:right|correct|fine|good|ok|okay)(?:\s+after\s+all)?\b", "", t, flags=re.I)
    t = re.sub(r"\s+after\s+all\b", "", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


def _num(tok: str) -> int | None:
    tok = tok.lower()
    if tok.isdigit():
        return int(tok)
    return NUM_WORDS.get(tok)


def _title(s: str) -> str:
    return " ".join(w if w.isupper() and len(w) <= 4 else w.capitalize() for w in s.split())


def _entity_after(words: list[str], i: int, allow_the: bool = False) -> str | None:
    out: list[str] = []
    j = i
    if allow_the and j < len(words) and words[j].lower() == "the":
        out.append("the")
        j += 1
    while j < len(words) and len(out) < 4:
        w = words[j]
        wl = w.lower()
        if wl in ENTITY_STOP or _num(wl) is not None or re.fullmatch(r"\d.*", wl):
            break
        out.append(w)
        j += 1
    if not out or out == ["the"]:
        return None
    if out[0].lower() in NOT_A_PLACE:
        return None
    return _title(" ".join(out))


def _words(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'&.-]*", s)


def extract_date(s: str, ref: date | None = None) -> str | None:
    sl = s.lower()
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", sl)
    if m:
        return m.group(1)
    if re.search(r"\bday after tomorrow\b", sl):
        return _resolve(2, "day after tomorrow", ref)
    if re.search(r"\btomorrow\b", sl):
        return _resolve(1, "tomorrow", ref)
    if re.search(r"\b(today|tonight)\b", sl):
        return _resolve(0, "today", ref)
    m = re.search(r"\b(next|this|coming)?\s*(" + "|".join(WEEKDAYS) + r")\b", sl)
    if m:
        surface = (m.group(1) + " " if m.group(1) else "") + m.group(2)
        if ref is not None:
            # ASSUMPTION: "next friday" == the upcoming friday, never today
            delta = (WEEKDAYS.index(m.group(2)) - ref.weekday()) % 7 or 7
            return (ref + timedelta(days=delta)).isoformat()
        return surface
    mon = "|".join(list(MONTHS) + list(MONTH_ABBR))
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + mon + r")\b", sl) or \
        re.search(r"\b(" + mon + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b", sl)
    if m:
        a, b = m.groups()
        day, mname = (a, b) if a.isdigit() else (b, a)
        month = MONTH_ABBR[mname[:3]]
        if ref is not None:
            year = ref.year + (1 if (month, int(day)) < (ref.month, ref.day) else 0)
            return f"{year:04d}-{month:02d}-{int(day):02d}"
        return f"{MONTHS[month - 1]} {int(day)}"
    return None


def _resolve(days: int, surface: str, ref: date | None) -> str:
    return (ref + timedelta(days=days)).isoformat() if ref is not None else surface


def extract_slots(seg: str, ref: date | None = None) -> tuple[dict[str, Any], list[str]]:
    """Returns (slots, order) where order lists slot names in mention order."""
    slots: dict[str, Any] = {}
    order: list[str] = []

    def put(k: str, v: Any) -> None:
        if v is None:
            return
        slots[k] = v
        if k in order:
            order.remove(k)
        order.append(k)

    s = NEGATED_RE.sub(" ", seg)  # "to Mumbai, not Delhi" -> drop Delhi
    sl = s.lower()
    words = _words(s)
    lw = [w.lower() for w in words]

    # explicit "<slot> to/is <value>"
    m = re.search(r"\b(destination|origin|source|date|passengers?|name|departure city|arrival city)\s+(?:to|is|should be|as|=)\s+(.+)$", s, re.I)
    if m:
        key = m.group(1).lower()
        val = m.group(2).strip(" .,!?")
        key = {"source": "origin", "departure city": "origin", "arrival city": "destination", "passenger": "passengers"}.get(key, key)
        if key == "passengers":
            n = _num(val.split()[0]) if val.split() else None
            put("passengers", n)
        elif key == "date":
            put("date", extract_date(val, ref) or val.lower())
        else:
            put(key, _title(val))

    for i, w in enumerate(lw):
        if w == "from" and "origin" not in slots:
            put("origin", _entity_after(words, i + 1))
        elif w in ("to", "into") and "destination" not in slots:
            ent = _entity_after(words, i + 1, allow_the=True)
            if ent and ent.lower().startswith("the ") and not re.search(r"\b(navigate|drive|take me|directions|route|head|go)\b", sl):
                ent = None
            put("destination", ent)

    d = extract_date(s, ref)
    if d:
        put("date", d)

    m = re.search(r"\b(?:in|at|near|around)\s+((?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+){0,2})\b", s)
    if m and m.group(1).lower() not in ENTITY_STOP and "destination" not in slots and "origin" not in slots:
        put("location", m.group(1))

    m = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:passengers?|people|persons?|adults?|tickets?|seats?|travell?ers?|of us)\b", sl) \
        or re.search(r"\bfor\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b(?!\s*(?:am|pm|o'?clock|:|hours?|minutes?|days?))", sl)
    if m:
        put("passengers", _num(m.group(1)))
    elif re.search(r"\bjust me\b|\bonly me\b|\bmyself\b", sl):
        put("passengers", 1)

    m = re.search(r"\b(premium economy|economy|business|first class)\b", sl)
    if m:
        put("cabin", m.group(1).replace(" class", ""))

    m = re.search(r"\b(?:at|around|by)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", sl)
    if m and (m.group(3) or m.group(2)):
        h = int(m.group(1)) % 12 + (12 if m.group(3) == "pm" else 0) if m.group(3) else int(m.group(1))
        put("time", f"{h:02d}:{m.group(2) or '00'}")
    else:
        m = re.search(r"\b(morning|afternoon|evening|night)\b", sl)
        if m:
            put("time", m.group(1))

    m = re.search(r"\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|last|cheapest|earliest|fastest|latest)\b(?!\s*(?:of\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))\s*(?:one|option|flight|result)?", sl)
    if m and not re.search(r"\bfirst class\b", sl):
        put("choice", ORDINALS[m.group(1)])
    m = re.search(r"\boption\s+(\d+|one|two|three|four|five)\b", sl)
    if m:
        put("choice", _num(m.group(1)))

    m = re.search(r"\b(?:my name is|name is|under the name|for passenger|passenger name is|this is)\s+([A-Za-z][A-Za-z' -]{1,40}?)(?=$|[,.!?]| and | on | for )", s, re.I)
    if m:
        put("name", _title(m.group(1).strip()))

    if "name" not in slots:
        m = re.search(r"\bfor\s+(?:(?:Mr|Ms|Mrs|Dr)\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", s)
        if m and not ({w.lower() for w in m.group(1).split()} & (ENTITY_STOP | set(MONTHS))):
            put("name", m.group(1))

    m = re.search(r"\b([A-Z0-9]{2}-\d{2,4})\b", s) or re.search(r"\bflight\s+([A-Z]{2}\d{2,4})\b", s)
    if m:
        put("flight_id", m.group(1))
    m = re.search(r"\b(?=[A-Z0-9]{6}\b)(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6}\b", s)
    if m and m.group(0) != slots.get("flight_id"):
        put("booking_ref", m.group(0))

    m = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", s)
    if m:
        put("email", m.group(0).lower())
    m = re.search(r"\b(?:\+?\d[\d -]{7,}\d)\b", s)
    if m:
        put("phone", re.sub(r"[ -]", "", m.group(0)))

    m = re.search(r"\b(urgent|high|medium|low)\s+priority\b|\b(urgent(?:ly)?|asap)\b", sl)
    if m:
        put("priority", "high" if (m.group(2) or m.group(1) in ("urgent",)) else m.group(1))

    for dev in sorted(DEVICES, key=len, reverse=True):
        if re.search(r"\b" + re.escape(dev) + r"\b", sl):
            put("device", dev)
            break

    code = r"((?=[A-Za-z0-9-]*\d)[A-Za-z0-9]{1,4}(?:-[A-Za-z0-9]{1,4})?)"
    m = re.search(r"\b(?:error|code)\s+(?:code\s+)?" + code + r"(?![\w-])", s, re.I) or \
        re.search(r"\b(?:showing|shows|displays|says|reads)\s+(?:error\s+)?(?:code\s+)?" + code + r"(?![\w-])", s, re.I)
    m = m or re.search(r"\b(?i:showing|shows|displays|says|reads)\s+(?i:error\s+|code\s+)?(?!OK\b)([A-Z]{2,4})\b(?![a-z])", s)
    if m:
        put("error_code", m.group(1))  # the display's own case: "C-d0", "4C", "LC"

    m = re.search(r"\bmodel\s+(?:number\s+|no\.?\s+)?((?=[A-Za-z0-9-]*\d)[A-Za-z0-9][A-Za-z0-9-]{2,})\b", s, re.I)
    if m:
        put("device_model", m.group(1).upper())

    return slots, order


def _bare_value(seg: str, last: str | None, ref: date | None) -> dict[str, Any]:
    """Repair segment with no slot keyword: 'actually Pune', 'I mean three'."""
    words = _words(seg)
    if not words:
        return {}
    n = _num(words[0])
    if n is not None and len(words) <= 3:
        return {"passengers": n} if last in (None, "passengers", "origin", "destination", "date") else {last: n}
    d = extract_date(seg, ref)
    if d:
        return {"date": d}
    ent = _entity_after(words, 0)
    if ent:
        target = last if last in ("origin", "destination", "name", "device") else "destination"
        return {target: ent}
    return {}


def bare_value(text: str, slot: str, ref: date | None = None) -> Any:
    """Answer to a clarification for `slot`: 'Chennai', 'three', 'tomorrow'."""
    seg = clean_text(text).strip(" .!?")
    words = _words(seg)
    if not words:
        return None
    if slot in ("passengers", "count", "quantity"):
        return next((n for n in (_num(w) for w in words) if n is not None), None)
    if slot == "date":
        return extract_date(seg, ref)
    if slot in ("issue", "query", "description"):
        return seg
    if "?" in text or len(words) > 5:
        return None
    lead = re.sub(r"^(?:it'?s|its|to|from|in|at|the city is|my name is|i'?m)\s+", "", seg, flags=re.I)
    return _title(lead) if lead else None


_COMPOUND_RE = re.compile(r",?\s+and\s+(?:also|then)\s+|;\s*|,\s+(?:also|then)\s+", re.I)
_ITEM_SLOTS = ("name", "flight_id", "booking_ref", "destination", "location")


def split_compound(text: str, ref: date | None = None) -> list[str]:
    """'Book UK-1 for A, and also UK-1 for B' -> two requests. Splits only when
    every part names its own item (a person, flight, booking or place)."""
    parts = [x.strip(" ,.") for x in _COMPOUND_RE.split(text) if x.strip(" ,.")]
    if len(parts) < 2 or REPAIR_RE.search(text):
        return [text]
    if all(any(k in extract_slots(clean_text(x), ref)[0] for k in _ITEM_SLOTS) for x in parts):
        return parts
    return [text]


def split_repairs(text: str) -> tuple[list[str], bool]:
    parts = REPAIR_RE.split(text)
    segs = [p.strip(" ,.;-") for p in parts]
    had = len(parts) > 1
    return [s for s in segs if s], had


INTENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("cancel_booking", re.compile(r"\bcancel\b.*\b(booking|reservation|flight|ticket|order)\b", re.I)),
    ("create_ticket", re.compile(
        r"\b(support ticket|raise a ticket|open a ticket|create a ticket|file a (?:ticket|complaint)|log a (?:ticket|complaint)"
        r"|complaint|report (?:a|an|the) (?:issue|problem|bug)|not working|stopped working|broken|won'?t (?:turn|start|work)|technician|service request)\b", re.I)),
    ("book_flight", re.compile(r"\b(book|reserve)\b", re.I)),
    ("search_flights", re.compile(r"\b(flights?|fly|plane)\b", re.I)),
    ("lookup_manual", re.compile(
        r"\b(manual|what does (?:this|that|the) .*mean|what(?:'s| is) this (?:light|icon|symbol|button|code)|blinking|flashing"
        r"|error code|how (?:do|can) i|how to|reset|fix|troubleshoot|repair)\b", re.I)),
    ("navigate", re.compile(r"\b(navigate|directions?|route|take me|drive (?:me )?to|head(?:ing)? to|reroute|get me to)\b", re.I)),
]


def detect_intent(text: str) -> str | None:
    for name, rx in INTENT_RULES:
        if rx.search(text):
            return name
    return None


def parse(text: str, last_slot: str | None = None, ref: date | None = None) -> Parse:
    clean = clean_text(text)
    p = Parse(raw=text, clean=clean)
    p.affirm = bool(AFFIRM_RE.search(clean))
    p.deny = bool(DENY_RE.search(clean)) and not REPAIR_RE.match(clean)
    p.intent = detect_intent(clean)
    if p.intent != "cancel_booking" and CANCEL_RE.search(clean):
        # "cancel that" with nothing after it cancels; "cancel that, search Pune" re-plans
        rest = CANCEL_RE.split(clean, maxsplit=1)[-1]
        rest_slots, _ = extract_slots(rest, ref)
        if not rest_slots and detect_intent(rest) is None:
            p.cancel = True
            return p
        clean = rest
        p.correction = True
    segs, had_repair = split_repairs(clean)
    p.correction = p.correction or had_repair or bool(re.search(r"\b(change|update|switch)\b", clean, re.I))
    last = last_slot
    for i, seg in enumerate(segs):
        found, order = extract_slots(seg, ref)
        repair_seg = i > 0 or (had_repair and len(segs) == 1)
        if not found and repair_seg:
            found = _bare_value(seg, last, ref)
            order = list(found)
        elif repair_seg:
            # "I mean Chennai, on October 12th": a bare value ahead of other slots
            lead = re.match(r"\s*([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,2}?)\s*(?:,|\b(?:on|for|at|from|in|tomorrow|today)\b)", seg)
            if lead and not extract_slots(lead.group(1), ref)[0]:
                bv = _bare_value(lead.group(1), last, ref)
                for k, v in bv.items():
                    if k not in found:
                        found[k] = v
                        order.insert(0, k)
        p.slots.update(found)
        if order:
            last = order[-1]
    if p.intent in ("create_ticket", "lookup_manual"):
        p.slots.setdefault("issue" if p.intent == "create_ticket" else "query", clean)
    return p
