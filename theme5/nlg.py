# /mnt/project-files/theme5/theme5/nlg.py
"""Template responses. Rules: short, grounded in tool results, never claim
completion before a state-modifying call has returned ok."""
from __future__ import annotations

import re
from typing import Any

from .planner import result_items
from .protocol import ToolSpec

_LABEL_KEYS = ("section", "airline", "carrier", "flight_number", "flight_no", "name", "title", "summary", "answer",
               "text", "step", "eta", "duration", "distance", "route", "via", "departure_time", "depart",
               "arrival_time", "price", "fare", "cost", "status")
_CONFIRM_KEYS = ("booking_ref", "confirmation", "confirmation_id", "confirmation_code", "booking_id", "pnr", "reference",
                 "ticket_id", "case_id", "id", "status")


GERUND = {"upgrade": "Upgrading", "search": "Searching", "find": "Finding", "get": "Getting", "lookup": "Looking up", "look": "Looking up",
          "book": "Booking", "create": "Creating", "cancel": "Cancelling", "update": "Updating", "modify": "Updating",
          "check": "Checking", "list": "Listing", "calculate": "Calculating", "compute": "Calculating",
          "reserve": "Reserving", "renew": "Renewing", "submit": "Submitting", "send": "Sending", "schedule": "Scheduling", "open": "Opening"}
PAST = {"book": "booked", "create": "created", "cancel": "cancelled", "update": "updated", "modify": "updated",
        "reserve": "reserved", "renew": "renewed", "submit": "submitted", "send": "sent", "schedule": "scheduled", "open": "opened",
        "change": "changed", "reschedule": "rescheduled", "set": "set", "upgrade": "upgraded"}


def _words(name: str) -> str:
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ").replace("-", " ").lower()


def human(tool: ToolSpec) -> str:
    return _words(tool.name)


def _split(tool: ToolSpec) -> tuple[str | None, str]:
    """('search', 'flights') for search_flights and flight_search alike."""
    words = human(tool).split()
    verbs = GERUND.keys() | PAST.keys()
    i = next((i for i, w in enumerate(words) if w.lower() in verbs), None)
    if i is None:
        return None, " ".join(words)
    rest = words[:i] + words[i + 1:]
    noun = " ".join(rest) or "request"
    if words[i].lower() in ("search", "find", "list") and not noun.endswith("s"):
        noun += "s"
    return words[i].lower(), noun


def _args_phrase(args: dict[str, Any]) -> str:
    parts = []
    if "origin" in args or "from" in args:
        parts.append(f"from {args.get('origin', args.get('from'))}")
    for k in ("destination", "to"):
        if k in args:
            parts.append(f"to {args[k]}")
            break
    for k in ("date", "departure_date", "travel_date"):
        if k in args:
            parts.append(f"on {args[k]}" if "-" in str(args[k]) or " " in str(args[k]) else f"for {args[k]}")
            break
    for k in ("passengers", "pax", "num_passengers"):
        if k in args:
            n = args[k]
            parts.append(f"for {n} passenger{'s' if n != 1 else ''}")
            break
    for k in ("cabin", "cabin_class", "class"):
        if k in args:
            parts.append(f"to {str(args[k]).replace('_', ' ')}")
            break
    for k in ("flight_id", "booking_ref"):
        if k in args:
            parts.append(f"{'on' if k == 'flight_id' else 'for booking'} {args[k]}")
    for k in ("passenger_name", "name", "customer_name"):
        if k in args:
            parts.append(f"for {args[k]}")
            break
    return " ".join(parts)


def _key(item: dict[str, Any], subs: tuple[str, ...]) -> Any:
    return next((v for k, v in item.items() if any(x in k.lower() for x in subs) and isinstance(v, (str, int, float))), None)


def _option(item: Any) -> str:
    """'6E-2041 at 06:40 for 4500' when the item looks like an option."""
    if not isinstance(item, dict):
        return str(item)
    ident = _key(item, ("flight_id", "flight_no", "flight_number", "_id")) or item.get("id")
    when = _key(item, ("depart", "time", "start"))
    price = _key(item, ("price", "fare", "cost", "amount"))
    if ident is None and when is None:
        return _fmt_item(item)
    bits = [str(ident)] if ident is not None else []
    if when is not None:
        bits.append(f"at {when}")
    if price is not None:
        bits.append(f"for {price}")
    return " ".join(bits)


def progress(tool: ToolSpec, args: dict[str, Any], retry: bool = False, revised: bool = False) -> str:
    phrase = _args_phrase(args)
    if retry:
        return "That didn't go through, trying again."
    verb, noun = _split(tool)
    lead = GERUND.get(verb or "", "working on" if tool.state_modifying else "checking")
    body = f"{lead.lower() if verb else lead} {noun if verb else human(tool)}{' ' + phrase if phrase else ''}"
    if revised:
        return f"Okay, updating: {body}."
    return f"{body[0].upper()}{body[1:]}."


def _fmt_item(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item)
    bits = []
    for k in _LABEL_KEYS:
        if k in item and isinstance(item[k], (str, int, float)):
            v = item[k]
            bits.append(f"{v}" if k in ("airline", "carrier", "name", "title", "summary", "answer", "text", "step")
                        else f"{k.replace('_', ' ')} {v}")
        if len(bits) >= 3:
            break
    if not bits:
        bits = [f"{v}" if k in ("section", "title") else f"{k.replace('_', ' ')} {v}"
                for k, v in list(item.items())[:3] if isinstance(v, (str, int, float)) and k not in ("device_model", "page")]
    steps = next((v for v in item.values() if isinstance(v, list) and v and all(isinstance(x, str) for x in v)), None)
    if steps:
        bits.append(" ".join(x.rstrip(".") + "." for x in steps[:4]).rstrip("."))
    if steps and len(bits) > 1:
        return f"{', '.join(bits[:-1])}. {bits[-1]}"
    return ", ".join(bits)


def _ref_of(result: dict[str, Any]) -> Any:
    ref = next((result[k] for k in _CONFIRM_KEYS if k in result and k != "status"), None)
    if ref is None:  # unseen tools: reservation_id, receipt, order_no, ...
        ref = next((v for k, v in result.items() if isinstance(v, (str, int)) and not isinstance(v, bool)
                    and re.search(r"(^|_)(id|ref|no|number)$|receipt|confirmation|reference", k.lower())), None)
    return ref


def _generic_args(args: dict[str, Any]) -> str:
    vals = [f"{_words(k)} {v}" for k, v in list(args.items())[:3] if isinstance(v, (str, int, float))]
    return f"with {', '.join(vals)}" if vals else ""


def _facts(result: dict[str, Any], args: dict[str, Any], skip: Any) -> str:
    """Up to three returned values the user has not already said."""
    def norm(v: Any) -> str:
        return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v).lower()
    echoed = {norm(v) for v in args.values()} | {norm(skip)}
    out = [f"{_words(k)} {v}" for k, v in result.items()
           if isinstance(v, (str, int, float)) and not isinstance(v, bool) and norm(v) not in echoed
           and k.lower() not in ("status", "ok", "applied", "success")]
    return ", ".join(out[:3])


def final(tool: ToolSpec, args: dict[str, Any], result: Any, intent: str | None) -> str:
    if tool.state_modifying:
        ref = _ref_of(result) if isinstance(result, dict) else None
        facts = _facts(result, args, ref) if isinstance(result, dict) else ""
        verb, noun = _split(tool)
        phrase = _args_phrase(args) or _generic_args(args)
        if verb in PAST:
            msg = f"Done, your {noun} is {PAST[verb]}{' ' + phrase if phrase else ''}."
        else:
            msg = f"Done, {human(tool)} succeeded{' ' + phrase if phrase else ''}."
        msg += f" Your reference is {ref}." if ref is not None else ""
        return msg + (f" {facts[0].upper()}{facts[1:]}." if facts else "")
    items = result_items(result)
    if isinstance(result, (str, int, float)):
        return str(result)
    if not items:
        return "I didn't find anything for that."
    if len(items) == 1:
        return f"Here's what I found: {_option(items[0]).rstrip('.')}."
    opts = [_option(i) for i in items[:3]]
    listed = ", ".join(opts[:-1]) + f", and {opts[-1]}" if len(opts) > 2 else " and ".join(opts)
    more = f" and {len(items) - 3} more" if len(items) > 3 else ""
    return f"I found {len(items)} options: {listed}{more}."


def failure(tool: ToolSpec, error: str | None, uncertain: bool = False) -> str:
    if tool.state_modifying and uncertain:
        return (f"Sorry, {human(tool)} timed out, so I can't confirm whether it went through. "
                "I haven't retried, to avoid doing it twice.")
    if tool.state_modifying:
        return f"Sorry, {human(tool)} didn't go through, so nothing was changed. Want me to try again?"
    return f"Sorry, I couldn't complete {human(tool)} right now."


def cancelled(had_commits: bool = False) -> str:
    if had_commits:
        return "Okay, I've stopped. Anything finished earlier stays as it is."
    return "Okay, I've stopped everything. No changes were made."


def repeat_done(tool: ToolSpec, args: dict[str, Any], result: Any) -> str:
    """A request to repeat a write that already went through."""
    text = final(tool, args, result, None)
    text = text.replace("Done, ", "That's already done: ", 1) if text.startswith("Done, ") else text
    return f"{text} I haven't done it a second time."


def already_done(tool: ToolSpec) -> str:
    return f"That's already done with the earlier details; {human(tool)} can't be changed with the tools I have."
