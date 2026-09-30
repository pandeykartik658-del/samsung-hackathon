# bench/gen_scenarios.py
"""Generate an adversarial 60-scenario suite in the harness scenario format
(sim/scenario.py), mirroring the hidden set's mix (guide section 4):
50% text (30), 30% audio (18), 20% visual (12).

Adversarial families (ids g<NN>_<modality>_<family>):
  near_simul      interrupt + correction within 0-40 ms, or two corrections back to back
  post_result     correction after the first search already returned
  fault           read-only error/timeout (retry expected); write error marked
                  "not executed" (retry allowed); write timeout (no blind retry,
                  report honestly); manual_lookup error
  dup_trap        slow booking + user repeats; confirm-while-booking; two
                  legitimate different bookings (ledger must not over-block);
                  "book it" again after success
  unseen          tools the agent has never seen, with unusual parameter names,
                  a chained unseen pair, and a missing required parameter
  hesitation      multi-second pauses and fillers before end of turn
  contradict      A -> B -> "no, A was right"
  clarify         a required slot is missing
  cancel_all      user abandons the task while a call is in flight
  frame_clear     visual: clear frame, question arrives around the frame
  frame_ambig     visual: two near-equal detections or low confidence -> clarify
  frame_switch    visual: user swaps device mid-lookup
  frame_ticket    visual: lookup then ticket, user repeats the ticket request

Everything is deterministic for a given --seed. Audio events carry an oracle
`transcript`, frames carry oracle `labels` (ASSUMPTION: the real kit ships raw
media only; the harness can strip oracles with --strip-oracle). Media files are
placeholders written by sim.assets.

    python -m bench.gen_scenarios --out bench/scenarios60 [--seed 5] [--n 60]
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REF_DATE = date(2026, 9, 29)
SESSION = {"reference_date": REF_DATE.isoformat(), "locale": "en-IN", "timezone": "Asia/Kolkata"}
CITIES = [("DEL", "Delhi"), ("BOM", "Mumbai"), ("BLR", "Bangalore"), ("MAA", "Chennai"), ("HYD", "Hyderabad"),
          ("CCU", "Kolkata"), ("GOI", "Goa"), ("PNQ", "Pune"), ("AMD", "Ahmedabad"), ("COK", "Kochi"),
          ("JAI", "Jaipur"), ("LKO", "Lucknow")]
NAMES = ["Rahul Verma", "Priya Nair", "Meera Iyer", "Arjun Rao", "Kavya Menon", "Vikram Shah", "Ananya Das",
         "Rohan Gupta", "Sneha Pillai", "Aditya Kulkarni"]
AIRLINES = ["6E", "AI", "UK", "SG", "QP"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
DEVICES = [  # (spoken name, model, error code, fixture section, step keywords)
    ("washing machine", "WW90T", "4C", "Error 4C: water supply", ["tap", "inlet hose", "filter"]),
    ("dryer", "DV90T", "HE", "Error HE: heating", ["lint filter", "vent", "restart"]),
    ("refrigerator", "RF28T", "22E", "Error 22E: fan", ["defrost", "fan", "unplug"]),
    ("air conditioner", "AR18B", "E4", "Error E4: sensor", ["sensor", "reset", "remote"]),
    ("TV", "QN55Q80", "NO SIGNAL", "No signal", ["input", "HDMI", "cable"]),
    ("dishwasher", "DW60M", "LC", "Error LC: leak", ["leak", "drain", "tray"]),
    ("microwave", "MS23K", "C-d0", "Error C-d0: button", ["button", "door", "power"]),
]
FLIGHT_INTENTS = {"any_of": ["search_flights", "flight_search"]}
BOOK_INTENTS = {"any_of": ["book_flight", "flight_booking"]}
DEVICE_INTENTS = {"any_of": ["troubleshoot", "troubleshoot_device", "manual_lookup", "lookup_manual"]}
TICKET_INTENTS = {"any_of": ["create_ticket", "support_ticket"]}
NULL_INTENT = {"any_of": [None, "none", "cancelled", "cancel"]}


def city(c: Tuple[str, str]) -> Dict[str, Any]:
    return {"any_of": [c[0], c[1]]}


def spoken_date(d: date) -> str:
    if d == REF_DATE + timedelta(days=1):
        return "tomorrow"
    suffix = "th" if 11 <= d.day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d.day % 10, "th")
    return f"{MONTHS[d.month - 1]} {d.day}{suffix}"


class B:
    """Builds one scenario: modality-aware user events plus expected block."""

    def __init__(self, sid: str, modality: str, family: str, title: str, rng: random.Random):
        self.sid, self.modality, self.family, self.rng = sid, modality, family, rng
        self.d: Dict[str, Any] = {
            "id": sid, "modality": modality, "title": title, "family": family, "description": "",
            "duration_ms": 30000, "session": dict(SESSION), "cancel_grace_ms": 300,
            "tools": {"enabled": [], "config": {}}, "events": [],
            "expected": {"required_calls": [], "forbidden_calls": [], "final_snapshot": None},
        }
        self._n = 0
        self._clip = 0

    # -- events --------------------------------------------------------------------
    def _id(self) -> str:
        self._n += 1
        return f"e{self._n}"

    def say(self, t: int, text: str, eot: bool = True, audio: Optional[bool] = None) -> str:
        eid = self._id()
        if (self.modality == "audio") if audio is None else audio:
            self._clip += 1
            clip = f"{self.sid}_c{self._clip}"
            dur = max(600, 55 * len(text))
            self.d["events"].append({"id": eid, "t_ms": t, "type": "audio_clip", "clip_id": clip,
                                     "path": f"assets/{clip}.wav", "sample_rate": 16000, "duration_ms": dur,
                                     "end_of_turn": eot, "transcript": text})
        else:
            self.d["events"].append({"id": eid, "t_ms": t, "type": "text_chunk", "text": text, "end_of_turn": eot})
        return eid

    def interrupt(self, t: int) -> str:
        eid = self._id()
        self.d["events"].append({"id": eid, "t_ms": t, "type": "interrupt", "reason": "barge_in"})
        return eid

    def frame(self, t: int, labels: List[Dict[str, Any]]) -> str:
        eid = self._id()
        fid = f"{self.sid}_f{eid[1:]}"
        self.d["events"].append({"id": eid, "t_ms": t, "type": "video_frame", "frame_id": fid,
                                 "path": f"assets/{fid}.png", "width": 160, "height": 120, "labels": labels})
        return eid

    # -- tools ---------------------------------------------------------------------
    def enable(self, *names: str) -> "B":
        for n in names:
            if n not in self.d["tools"]["enabled"]:
                self.d["tools"]["enabled"].append(n)
        return self

    def cfg(self, tool: str) -> Dict[str, Any]:
        return self.d["tools"]["config"].setdefault(tool, {})

    def extra(self, tool_def: Dict[str, Any]) -> None:
        self.d["tools"].setdefault("extra", []).append(tool_def)

    def flights(self, o, dst, day: date, latency: int = 1500) -> List[str]:
        ids = []
        for _ in range(2):
            ids.append(f"{self.rng.choice(AIRLINES)}-{self.rng.randint(100, 9899)}")
        c = self.cfg("flight_search")
        c.setdefault("latency_ms", latency)
        c.setdefault("fixtures", []).append({
            "match": {"destination": city(dst), "date": day.isoformat()},
            "response": {"flights": [
                {"flight_id": fid, "origin": o[0], "destination": dst[0], "date": day.isoformat(),
                 "depart": f"{7 + 5 * i:02d}:{15 * self.rng.randint(0, 3):02d}", "price_inr": self.rng.randint(2800, 9800)}
                for i, fid in enumerate(ids)]},
        })
        return ids

    def booking(self, fid: str, name: str, latency: int = 900) -> str:
        ref = "".join(self.rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
        c = self.cfg("book_flight")
        c.setdefault("latency_ms", latency)
        c.setdefault("fixtures", []).append({"match": {"flight_id": fid, "passenger_name": name},
                                             "response": {"booking_ref": ref, "flight_id": fid, "status": "confirmed"}})
        return ref

    def manual(self, dev, latency: int = 900) -> None:
        name, model, code, section, keys = dev
        c = self.cfg("manual_lookup")
        c.setdefault("latency_ms", latency)
        c.setdefault("fixtures", []).append({"match": {"device_model": {"any_of": [model, name]}},
                                             "response": {"device_model": model, "section": section,
                                                          "steps": [f"Check the {k}." for k in keys]}})

    # -- expected ------------------------------------------------------------------
    @property
    def exp(self) -> Dict[str, Any]:
        return self.d["expected"]

    def require(self, tool: str, args: Optional[Dict[str, Any]] = None, **kw: Any) -> None:
        self.exp["required_calls"].append({"tool": tool, "args": args or {}, **kw})

    def forbid(self, tool: str, args: Optional[Dict[str, Any]] = None, **kw: Any) -> None:
        f = {"tool": tool, **kw}
        if args:
            f["args"] = args
        self.exp["forbidden_calls"].append(f)

    def done(self, desc: str) -> Dict[str, Any]:
        self.d["description"] = desc
        last = max(e["t_ms"] for e in self.d["events"])
        self.d["duration_ms"] = min(60000, int(last) + 20000)
        return self.d


# ------------------------------------------------------------------------------
# families
# ------------------------------------------------------------------------------
def _trip(rng: random.Random, k: int = 2):
    cs = rng.sample(CITIES, k + 1)
    day = REF_DATE + timedelta(days=rng.randint(1, 40))
    return cs[0], cs[1:], day


def fam_near_simul(b: B, v: int) -> Dict[str, Any]:
    rng = b.rng
    o, (d1, d2, d3), day = _trip(rng, 3)
    b.enable("flight_search", "book_flight")
    b.flights(o, d1, day, latency=2500)
    b.say(0, f"Search flights from {o[1]} to {d1[1]} on {spoken_date(day)}.")
    ti = rng.randint(500, 1500)
    ei = b.interrupt(ti)
    if v % 2:
        b.flights(o, d2, day, latency=2500)
        ids = b.flights(o, d3, day, latency=2500)
        b.say(ti + rng.randint(0, 20), f"Make it {d2[1]}.")
        last = b.say(ti + rng.randint(25, 60), f"Sorry, no, {d3[1]}.")
        final, stale = d3, [d1, d2]
    else:
        ids = b.flights(o, d2, day, latency=2500)
        last = b.say(ti + rng.randint(0, 40), f"No, {d2[1]} instead.")
        final, stale = d2, [d1]
    b.require("flight_search", {"origin": city(o), "destination": city(final), "date": day.isoformat()})
    for s in stale:
        b.forbid("flight_search", {"destination": city(s)}, after_event=last)
    b.exp["must_cancel"] = [{"tool": "flight_search", "args": {"destination": city(d1)}, "anchor_event": ei}]
    b.exp["max_write_calls"] = {"book_flight": 0}
    b.exp["final_snapshot"] = {"intent": FLIGHT_INTENTS, "slots": {
        "origin": city(o), "destination": city(final), "date": day.isoformat()}}
    b.exp["final_mentions"] = [{"any_of": ids}]
    return b.done(f"Barge-in {len(stale)} correction(s) within ~60 ms of the interrupt while the {d1[1]} search runs.")


def fam_post_result(b: B, v: int) -> Dict[str, Any]:
    rng = b.rng
    o, (dst,), day = _trip(rng, 1)
    day2 = day + timedelta(days=rng.randint(1, 5))
    b.enable("flight_search", "book_flight")
    old = b.flights(o, dst, day, latency=1200)
    new = b.flights(o, dst, day2, latency=1200)
    e1 = b.say(0, f"Find flights from {o[1]} to {dst[1]} on {spoken_date(day)}.")
    e2 = b.say(rng.randint(2600, 4000), f"Oh wait, actually I need {spoken_date(day2)}, not {spoken_date(day)}.")
    b.require("flight_search", {"destination": city(dst), "date": day.isoformat()}, status="any")
    b.require("flight_search", {"origin": city(o), "destination": city(dst), "date": day2.isoformat()})
    b.exp["ordered_calls"] = True
    b.forbid("flight_search", {"date": day.isoformat()}, after_event=e2)
    b.exp["max_write_calls"] = {"book_flight": 0}
    b.exp["final_snapshot"] = {"intent": FLIGHT_INTENTS, "slots": {
        "origin": city(o), "destination": city(dst), "date": day2.isoformat()}}
    b.exp["final_mentions"] = [{"any_of": new}]
    b.exp["final_text"] = {"excludes": old}
    return b.done("Date correction after the first search already returned; only the date slot changes.")


def fam_fault(b: B, v: int) -> Dict[str, Any]:
    rng = b.rng
    kind = ["read_error", "read_timeout", "write_not_executed", "write_timeout", "manual_error"][v % 5]
    name = rng.choice(NAMES)
    if kind.startswith("read"):
        o, (dst,), day = _trip(rng, 1)
        b.enable("flight_search", "book_flight")
        ids = b.flights(o, dst, day, latency=1000)
        fault = {"call_index": 1, "kind": "error", "error": "upstream_unavailable"} if kind == "read_error" \
            else {"call_index": 1, "kind": "timeout", "timeout_ms": 3000}
        b.cfg("flight_search")["faults"] = [fault]
        b.say(0, f"Flights from {o[1]} to {dst[1]} {spoken_date(day)} please.")
        b.require("flight_search", {"origin": city(o), "destination": city(dst), "date": day.isoformat()})
        b.exp["max_write_calls"] = {"book_flight": 0}
        b.exp["final_snapshot"] = {"intent": FLIGHT_INTENTS, "slots": {"destination": city(dst), "date": day.isoformat()}}
        b.exp["final_mentions"] = [{"any_of": ids}]
        return b.done(f"flight_search fails once ({fault['kind']}); read-only, so retry and answer.")
    if kind == "manual_error":
        dev = rng.choice(DEVICES)
        b.enable("manual_lookup", "create_ticket")
        b.manual(dev)
        b.cfg("manual_lookup")["faults"] = [{"call_index": 1, "kind": "error", "error": "manual_service_503"}]
        b.say(0, f"My {dev[0]}, model {dev[1]}, shows error {dev[2]}. How do I fix it?")
        b.require("manual_lookup", {"device_model": {"any_of": [dev[1], dev[0]]}, "query": {"regex": _re(dev[2])}})
        b.forbid("create_ticket")
        b.exp["final_snapshot"] = {"intent": DEVICE_INTENTS, "slots": {"device_model": {"any_of": [dev[1], dev[0]]}}}
        b.exp["final_text"] = {"includes_any": dev[4]}
        return b.done("manual_lookup errors once; read-only, so retry and ground the answer in the manual.")
    fid = f"{rng.choice(AIRLINES)}-{rng.randint(100, 9899)}"
    b.enable("flight_search", "book_flight")
    ref = b.booking(fid, name)
    b.say(0, f"Book flight {fid} for {name}.")
    slots = {"flight_id": fid, "passenger_name": name}
    if kind == "write_not_executed":
        b.cfg("book_flight")["faults"] = [{"call_index": 1, "kind": "error",
                                           "error": "service_unavailable: booking not executed, safe to retry"}]
        b.require("book_flight", slots)
        b.exp["max_write_calls"] = {"book_flight": 1}
        b.exp["final_snapshot"] = {"intent": BOOK_INTENTS, "slots": slots}
        b.exp["final_mentions"] = [ref]
        return b.done("book_flight fails with an explicit 'not executed' error; one retry is safe.")
    b.cfg("book_flight")["faults"] = [{"call_index": 1, "kind": "timeout", "timeout_ms": 4000}]
    b.require("book_flight", slots, status="timeout", max_count=1)
    b.exp["max_write_calls"] = {"book_flight": 1}
    b.exp["must_report_failure"] = True
    b.exp["final_snapshot"] = {"intent": BOOK_INTENTS, "slots": slots}
    return b.done("book_flight times out (outcome unknown, not idempotent): no blind retry, report honestly.")


def fam_dup_trap(b: B, v: int) -> Dict[str, Any]:
    rng = b.rng
    kind = ["slow_repeat", "again_after_success", "confirm_during", "two_legit"][v % 4]
    b.enable("flight_search", "book_flight")
    n1, n2 = rng.sample(NAMES, 2)
    fid = f"{rng.choice(AIRLINES)}-{rng.randint(100, 9899)}"
    ref = b.booking(fid, n1)
    slots = {"flight_id": fid, "passenger_name": n1}
    if kind == "two_legit":
        ref2 = b.booking(fid, n2)
        b.say(0, f"Book {fid} for {n1}, and also {fid} for {n2}.")
        b.require("book_flight", slots, max_count=1)
        b.require("book_flight", {"flight_id": fid, "passenger_name": n2}, max_count=1)
        b.exp["max_write_calls"] = {"book_flight": 2}
        b.exp["final_snapshot"] = {"intent": BOOK_INTENTS, "slots": {"flight_id": fid}}
        b.exp["final_mentions"] = [ref, ref2]
        return b.done("Two different passengers on one flight: two writes are correct, a third is not.")
    if kind == "slow_repeat":
        b.cfg("book_flight")["faults"] = [{"call_index": 1, "kind": "slow", "latency_ms": 6000}]
        b.say(0, f"Book flight {fid} for {n1}.")
        e2 = b.say(rng.randint(2000, 3000), f"Hello? Did that go through? Book {fid} for {n1}.")
        b.say(rng.randint(4200, 5200), "Just book it now please.")
        b.forbid("book_flight", after_event=e2)
    elif kind == "again_after_success":
        b.say(0, f"Please book {fid} for {n1}.")
        e2 = b.say(rng.randint(2500, 3500), "Great, book it.")
        b.forbid("book_flight", after_event=e2)
    else:
        b.cfg("book_flight")["latency_ms"] = 3000
        b.say(0, f"Book {fid} for {n1}.")
        e2 = b.interrupt(rng.randint(600, 1200))
        b.say(b.d["events"][-1]["t_ms"] + rng.randint(0, 30), f"Hold on, that's for {n1}, right? Yes, go ahead.")
        b.forbid("book_flight", after_event=e2)
    b.require("book_flight", slots, max_count=1)
    b.exp["max_write_calls"] = {"book_flight": 1}
    b.exp["final_snapshot"] = {"intent": BOOK_INTENTS, "slots": slots}
    b.exp["final_mentions"] = [ref]
    return b.done(f"Duplicate-booking trap ({kind}): exactly one book_flight.")


def _re(code: str) -> str:
    import re
    return re.escape(code).replace("\\ ", "\\s*")


UNSEEN: List[Callable[[B], Dict[str, Any]]] = []


def _unseen(fn):
    UNSEEN.append(fn)
    return fn


@_unseen
def u_reserve_table(b: B) -> Dict[str, Any]:
    n = b.rng.randint(2, 8)
    day = REF_DATE + timedelta(days=b.rng.randint(1, 20))
    venue = b.rng.choice(["Toit", "Bombay Canteen", "Indian Accent", "Karavalli"])
    b.extra({"name": "reserve_table", "description": "Reserve a restaurant table. Creates a reservation.",
             "side_effect": "write",
             "parameters": {"type": "object", "additionalProperties": False, "required": ["venueSlug", "party_sz", "when_iso"],
                            "properties": {"venueSlug": {"type": "string", "description": "Restaurant name or slug"},
                                           "party_sz": {"type": "integer", "description": "Number of guests"},
                                           "when_iso": {"type": "string", "description": "Date and time, ISO 8601"}}},
             "mock": {"latency_ms": 700, "response": {"reservation_id": "RSV-4471", "status": "confirmed"}}})
    b.say(0, f"Reserve a table at {venue} for {n} people on {spoken_date(day)} at 8 pm.")
    args = {"venueSlug": {"regex": venue.split()[0]}, "party_sz": n, "when_iso": {"regex": day.isoformat()}}
    b.require("reserve_table", args, max_count=1)
    b.exp["max_write_calls"] = {"reserve_table": 1}
    b.exp["final_snapshot"] = {"intent": {"any_of": ["reserve_table"]}, "slots": {"party_sz": n}}
    b.exp["final_mentions"] = ["RSV-4471"]
    return b.done("Unseen write tool with camelCase/abbreviated params (venueSlug, party_sz, when_iso).")


@_unseen
def u_track_parcel(b: B) -> Dict[str, Any]:
    awb = "".join(str(b.rng.randint(0, 9)) for _ in range(10))
    b.extra({"name": "trk_pkg", "description": "Get courier tracking status for an air waybill. Read only.",
             "read_only": True,
             "parameters": {"type": "object", "required": ["awb_no"],
                            "properties": {"awb_no": {"type": "string", "pattern": "^[0-9]{10}$"}}},
             "mock": {"latency_ms": 400, "response": {"awb_no": awb, "status_code": "OFD", "eta": "today 6 pm"}}})
    b.say(0, f"Where is my parcel? The waybill is {' '.join(awb[:5])} {' '.join(awb[5:])}.")
    b.require("trk_pkg", {"awb_no": awb})
    b.exp["final_snapshot"] = {"intent": {"any_of": ["trk_pkg"]}, "slots": {"awb_no": awb}}
    b.exp["final_text"] = {"includes_any": ["6 pm", "today", "out for delivery"]}
    return b.done("Unseen read tool, opaque name (trk_pkg), digits spoken with spaces, pattern-validated.")


@_unseen
def u_thermostat(b: B) -> Dict[str, Any]:
    zone = b.rng.choice(["living", "bedroom", "kitchen"])
    tgt = b.rng.randint(19, 26)
    b.extra({"name": "hvac_setpoint", "description": "Set target temperature for a zone.",
             "annotations": {"readOnlyHint": False, "idempotentHint": True},
             "parameters": {"type": "object", "required": ["zone_label", "target_c"],
                            "properties": {"zone_label": {"type": "string", "enum": ["living", "bedroom", "kitchen"]},
                                           "target_c": {"type": "number"}}},
             "mock": {"latency_ms": 300, "response": {"zone_label": zone, "target_c": tgt, "applied": True}}})
    b.say(0, f"Set the {zone} room to {tgt} degrees.")
    b.require("hvac_setpoint", {"zone_label": zone, "target_c": tgt}, max_count=1)
    b.exp["final_snapshot"] = {"intent": {"any_of": ["hvac_setpoint"]}, "slots": {"zone_label": zone, "target_c": tgt}}
    b.exp["final_text"] = {"includes_any": [str(tgt)]}
    return b.done("Unseen MCP-style manifest (annotations.readOnlyHint), enum parameter.")


@_unseen
def u_renew(b: B) -> Dict[str, Any]:
    lic = f"KA{b.rng.randint(10, 99)}{b.rng.randint(1000000, 9999999)}"
    yrs = b.rng.choice([1, 5])
    b.extra({"name": "renewLicence", "description": "Renew a driving licence. Charges the fee.",  # no side_effect key: only "mutating"
             "mutating": True,
             "parameters": {"type": "object", "required": ["LicenceNumber", "yrs"],
                            "properties": {"LicenceNumber": {"type": "string"},
                                           "yrs": {"type": "integer", "enum": [1, 5]}}},
             "mock": {"latency_ms": 900, "response": {"receipt": "RCPT-8812", "valid_until": "2031-09-29"}}})
    b.say(0, f"Renew my driving licence {lic} for {'one year' if yrs == 1 else 'five years'}.")
    b.require("renewLicence", {"LicenceNumber": lic, "yrs": yrs}, max_count=1)
    b.exp["max_write_calls"] = {"renewLicence": 1}
    b.exp["final_snapshot"] = {"intent": {"any_of": ["renewLicence"]}, "slots": {"LicenceNumber": lic}}
    b.exp["final_mentions"] = ["RCPT-8812"]
    return b.done("Unseen tool with list-style parameter schema and PascalCase names.")


@_unseen
def u_fx(b: B) -> Dict[str, Any]:
    amt = b.rng.choice([120, 250, 999, 40])
    b.extra({"name": "fx_quote", "description": "Quote a currency conversion.", "kind": "query",
             "parameters": {"type": "object", "required": ["amt", "frm", "to_ccy"],
                            "properties": {"amt": {"type": "number"}, "frm": {"type": "string", "description": "ISO 4217 source"},
                                           "to_ccy": {"type": "string", "description": "ISO 4217 target"}}},
             "mock": {"latency_ms": 300, "response": {"quote": round(amt * 83.4, 2), "rate": 83.4}}})
    b.say(0, f"How much is {amt} US dollars in rupees?")
    b.require("fx_quote", {"amt": amt, "frm": "USD", "to_ccy": "INR"})
    b.exp["final_snapshot"] = {"intent": {"any_of": ["fx_quote"]}, "slots": {"frm": "USD", "to_ccy": "INR"}}
    b.exp["final_text"] = {"includes_any": [str(round(amt * 83.4, 2)), f"{amt * 83.4:,.2f}", str(int(amt * 83.4))]}
    return b.done("Unseen read tool (kind=query) where values must be normalised to ISO codes.")


@_unseen
def u_chain(b: B) -> Dict[str, Any]:
    order = f"OD{b.rng.randint(100000, 999999)}"
    sku = f"SKU-{b.rng.randint(1000, 9999)}"
    b.extra({"name": "order_info", "description": "Fetch an order.", "side_effect": "read",
             "parameters": {"type": "object", "required": ["orderNo"], "properties": {"orderNo": {"type": "string"}}},
             "mock": {"latency_ms": 500, "response": {"orderNo": order, "item_sku": sku, "delivered": True}}})
    b.extra({"name": "start_return", "description": "Open a return for an item.", "side_effect": "write",
             "parameters": {"type": "object", "required": ["item_sku", "reason_code"],
                            "properties": {"item_sku": {"type": "string"},
                                           "reason_code": {"type": "string", "enum": ["DAMAGED", "WRONG_ITEM", "NOT_NEEDED"]}}},
             "mock": {"latency_ms": 700, "response": {"rma": "RMA-5520", "status": "open"}}})
    b.say(0, f"The item in order {order} arrived damaged, I want to return it.")
    b.require("order_info", {"orderNo": order})
    b.require("start_return", {"item_sku": sku, "reason_code": "DAMAGED"}, max_count=1)
    b.exp["ordered_calls"] = True
    b.exp["max_write_calls"] = {"start_return": 1}
    b.exp["final_snapshot"] = {"intent": {"any_of": ["start_return"]}, "slots": {"item_sku": sku}}
    b.exp["final_mentions"] = ["RMA-5520"]
    return b.done("Chained unseen tools: order_info output (item_sku) feeds start_return.")


@_unseen
def u_missing(b: B) -> Dict[str, Any]:
    ref = f"PKG{b.rng.randint(10000, 99999)}"
    win = b.rng.choice(["morning", "afternoon", "evening"])
    b.extra({"name": "schedule_pickup", "description": "Schedule a courier pickup.", "side_effect": "write",
             "parameters": {"type": "object", "required": ["pkg_ref", "pickup_window"],
                            "properties": {"pkg_ref": {"type": "string"},
                                           "pickup_window": {"type": "string", "enum": ["morning", "afternoon", "evening"]}}},
             "mock": {"latency_ms": 600, "response": {"pickup_id": "PU-3390", "window": win}}})
    e1 = b.say(0, f"Schedule a pickup for package {ref}.")
    e2 = b.say(4000, f"The {win}, please.")
    b.exp["clarify"] = {"after_event": e1, "before_event": e2}
    b.forbid("schedule_pickup", before_event=e2)
    b.require("schedule_pickup", {"pkg_ref": ref, "pickup_window": win}, max_count=1)
    b.exp["final_snapshot"] = {"intent": {"any_of": ["schedule_pickup"]}, "slots": {"pkg_ref": ref, "pickup_window": win}}
    b.exp["final_mentions"] = ["PU-3390"]
    return b.done("Unseen write tool with a missing required enum parameter: clarify before calling.")


def fam_unseen(b: B, v: int) -> Dict[str, Any]:
    b.enable("flight_search")
    d = UNSEEN[v % len(UNSEEN)](b)
    b.forbid("flight_search")
    return d


def fam_hesitation(b: B, v: int) -> Dict[str, Any]:
    rng = b.rng
    b.enable("flight_search", "book_flight")
    name = rng.choice(NAMES)
    fid = f"{rng.choice(AIRLINES)}-{rng.randint(100, 9899)}"
    ref = b.booking(fid, name)
    t1 = rng.randint(3000, 6000)
    t2 = t1 + rng.randint(2000, 4000)
    b.say(0, "I'd like to book, um,", eot=False)
    b.say(t1, f"uh, flight {fid}, for, hmm,", eot=False)
    last = b.say(t2, f"for {name}.")
    b.forbid("book_flight", before_event=last)
    b.require("book_flight", {"flight_id": fid, "passenger_name": name}, max_count=1)
    b.exp["max_write_calls"] = {"book_flight": 1}
    b.exp["final_snapshot"] = {"intent": BOOK_INTENTS, "slots": {"flight_id": fid, "passenger_name": name}}
    b.exp["final_mentions"] = [ref]
    return b.done(f"Long hesitations ({t1} ms and {t2 - t1} ms gaps) before end of turn; no premature write.")


def fam_contradict(b: B, v: int) -> Dict[str, Any]:
    rng = b.rng
    o, (d1, d2), day = _trip(rng, 2)
    b.enable("flight_search", "book_flight")
    ids = b.flights(o, d1, day, latency=2000)
    b.flights(o, d2, day, latency=2000)
    b.say(0, f"Flights from {o[1]} to {d1[1]} on {spoken_date(day)}.")
    t2 = rng.randint(500, 1200)
    b.say(t2, f"No, {d2[1]}.")
    e3 = b.say(t2 + rng.randint(400, 1200), f"No no, sorry, {d1[1]} was right.")
    b.require("flight_search", {"origin": city(o), "destination": city(d1), "date": day.isoformat()})
    b.forbid("flight_search", {"destination": city(d2)}, after_event=e3)
    b.exp["max_write_calls"] = {"book_flight": 0}
    b.exp["final_snapshot"] = {"intent": FLIGHT_INTENTS, "slots": {
        "origin": city(o), "destination": city(d1), "date": day.isoformat()}}
    b.exp["final_mentions"] = [{"any_of": ids}]
    return b.done("Contradictory corrections A -> B -> A; final state must be A, B work cancelled.")


def fam_clarify(b: B, v: int) -> Dict[str, Any]:
    rng = b.rng
    o, (dst,), day = _trip(rng, 1)
    b.enable("flight_search", "book_flight")
    ids = b.flights(o, dst, day)
    e1 = b.say(0, f"I need a flight to {dst[1]}.")
    e2 = b.say(rng.randint(3500, 5000), f"From {o[1]}, on {spoken_date(day)}.")
    b.exp["clarify"] = {"after_event": e1, "before_event": e2}
    b.forbid("flight_search", before_event=e2)
    b.require("flight_search", {"origin": city(o), "destination": city(dst), "date": day.isoformat()})
    b.exp["max_write_calls"] = {"book_flight": 0}
    b.exp["final_snapshot"] = {"intent": {"any_of": FLIGHT_INTENTS["any_of"] + ["book_flight"]}, "slots": {
        "origin": city(o), "destination": city(dst), "date": day.isoformat()}}
    b.exp["final_mentions"] = [{"any_of": ids}]
    return b.done("Origin and date missing: ask, do not guess, then search.")


def fam_cancel_all(b: B, v: int) -> Dict[str, Any]:
    rng = b.rng
    o, (dst,), day = _trip(rng, 1)
    b.enable("flight_search", "book_flight")
    b.flights(o, dst, day, latency=3000)
    b.say(0, f"Search flights from {o[1]} to {dst[1]} on {spoken_date(day)}.")
    e2 = b.say(rng.randint(700, 1500), "Actually never mind, cancel everything.")
    b.require("flight_search", {"destination": city(dst)}, status="any")
    b.exp["must_cancel"] = [{"tool": "flight_search", "anchor_event": e2}]
    b.forbid("flight_search", after_event=e2)
    b.forbid("book_flight")
    b.exp["max_write_calls"] = {"book_flight": 0}
    b.exp["final_snapshot"] = {"intent": NULL_INTENT, "slots": {}}
    return b.done("User abandons the task mid-search: cancel, confirm, no further calls.")


def _dev_labels(dev, conf: float, code: bool = True) -> List[Dict[str, Any]]:
    lab = {"label": dev[0].replace(" ", "_"), "model": dev[1], "confidence": conf}
    if code:
        lab["text"] = dev[2]
    return [lab]


def fam_frame_clear(b: B, v: int) -> Dict[str, Any]:
    rng = b.rng
    dev = rng.choice(DEVICES)
    b.enable("manual_lookup", "create_ticket")
    b.manual(dev)
    if v % 2:  # question first, frame lands 30-80 ms later
        b.say(0, "How do I fix the error on this?", audio=False)
        b.frame(rng.randint(30, 80), _dev_labels(dev, 0.93))
    else:
        b.frame(0, _dev_labels(dev, 0.93))
        b.say(rng.randint(150, 600), "What does this error mean and how do I fix it?", audio=False)
    b.require("manual_lookup", {"device_model": {"any_of": [dev[1], dev[0]]}, "query": {"regex": _re(dev[2])}})
    b.forbid("create_ticket")
    b.exp["max_write_calls"] = {"create_ticket": 0}
    b.exp["final_snapshot"] = {"intent": DEVICE_INTENTS, "slots": {"device_model": {"any_of": [dev[1], dev[0]]}}}
    b.exp["final_text"] = {"includes_any": dev[4]}
    return b.done(f"Clear frame of a {dev[0]} showing {dev[2]}; ground the answer in the manual.")


def fam_frame_ambig(b: B, v: int) -> Dict[str, Any]:
    rng = b.rng
    d1, d2 = rng.sample(DEVICES, 2)
    b.enable("manual_lookup", "create_ticket")
    b.manual(d1)
    if v % 2:
        e0 = b.frame(0, [{"label": d1[0].replace(" ", "_"), "model": d1[1], "confidence": 0.48},
                         {"label": d2[0].replace(" ", "_"), "model": d2[1], "confidence": 0.46}])
    else:
        e0 = b.frame(0, _dev_labels(d1, 0.31, code=False))
    e1 = b.say(300, "Can you help me fix this?", audio=False)
    e2 = b.say(rng.randint(4000, 5500), f"It's the {d1[0]}, model {d1[1]}, showing {d1[2]}.", audio=False)
    b.exp["clarify"] = {"after_event": e0, "before_event": e2}
    b.forbid("manual_lookup", before_event=e2)
    b.forbid("manual_lookup", {"device_model": {"any_of": [d2[1], d2[0]]}})
    b.forbid("create_ticket")
    b.require("manual_lookup", {"device_model": {"any_of": [d1[1], d1[0]]}, "query": {"regex": _re(d1[2])}})
    b.exp["final_snapshot"] = {"intent": DEVICE_INTENTS, "slots": {"device_model": {"any_of": [d1[1], d1[0]]}}}
    b.exp["final_text"] = {"includes_any": d1[4]}
    return b.done("Ambiguous or low-confidence frame: clarify before any lookup.")


def fam_frame_switch(b: B, v: int) -> Dict[str, Any]:
    rng = b.rng
    d1, d2 = rng.sample(DEVICES, 2)
    b.enable("manual_lookup", "create_ticket")
    b.manual(d1, latency=2500)
    b.manual(d2, latency=900)
    b.frame(0, _dev_labels(d1, 0.92))
    b.say(200, "How do I clear this error?", audio=False)
    ti = rng.randint(900, 1600)
    ei = b.interrupt(ti)
    b.frame(ti + rng.randint(0, 30), _dev_labels(d2, 0.94))
    last = b.say(ti + rng.randint(40, 90), f"Sorry, wrong one, I meant this {d2[0]}.", audio=False)
    b.exp["must_cancel"] = [{"tool": "manual_lookup", "args": {"device_model": {"any_of": [d1[1], d1[0]]}}, "anchor_event": ei}]
    b.forbid("manual_lookup", {"device_model": {"any_of": [d1[1], d1[0]]}}, after_event=last)
    b.forbid("create_ticket")
    b.require("manual_lookup", {"device_model": {"any_of": [d2[1], d2[0]]}, "query": {"regex": _re(d2[2])}})
    b.exp["final_snapshot"] = {"intent": DEVICE_INTENTS, "slots": {"device_model": {"any_of": [d2[1], d2[0]]}}}
    b.exp["final_text"] = {"includes_any": d2[4], "excludes": [d1[3]]}
    return b.done("User switches device mid-lookup (new frame + barge-in); stale lookup must be cancelled.")


def fam_frame_ticket(b: B, v: int) -> Dict[str, Any]:
    rng = b.rng
    dev = rng.choice(DEVICES)
    b.enable("manual_lookup", "create_ticket")
    b.manual(dev)
    b.cfg("create_ticket")["latency_ms"] = 2500
    b.frame(0, _dev_labels(dev, 0.9))
    b.say(200, f"I already tried the manual steps for {dev[2]}. Please open a support ticket for this.", audio=False)
    e3 = b.say(rng.randint(1500, 2200), "Did you file it? Please open the ticket.", audio=False)
    b.forbid("create_ticket", after_event=e3)
    b.require("create_ticket", {"device_model": {"any_of": [dev[1], dev[0]]}}, max_count=1)
    b.exp["max_write_calls"] = {"create_ticket": 1}
    b.exp["final_snapshot"] = {"intent": TICKET_INTENTS, "slots": {"device_model": {"any_of": [dev[1], dev[0]]}}}
    b.exp["final_mentions"] = [{"regex": "TCK-\\d{5}"}]
    return b.done("Frame-grounded ticket with the user repeating the request while it is being created.")


FAMILIES: Dict[str, Callable[[B, int], Dict[str, Any]]] = {
    "near_simul": fam_near_simul, "post_result": fam_post_result, "fault": fam_fault, "dup_trap": fam_dup_trap,
    "unseen": fam_unseen, "hesitation": fam_hesitation, "contradict": fam_contradict, "clarify": fam_clarify,
    "cancel_all": fam_cancel_all, "frame_clear": fam_frame_clear, "frame_ambig": fam_frame_ambig,
    "frame_switch": fam_frame_switch, "frame_ticket": fam_frame_ticket,
}

# counts per modality for n=60 (30/18/12)
PLAN = {
    "text": [("near_simul", 4), ("post_result", 4), ("fault", 5), ("dup_trap", 4), ("unseen", 5),
             ("hesitation", 3), ("contradict", 3), ("clarify", 1), ("cancel_all", 1)],
    "audio": [("near_simul", 3), ("post_result", 2), ("fault", 2), ("dup_trap", 3), ("unseen", 2),
              ("hesitation", 2), ("contradict", 2), ("clarify", 1), ("cancel_all", 1)],
    "visual": [("frame_clear", 4), ("frame_ambig", 3), ("frame_switch", 3), ("frame_ticket", 2)],
}


def plan_for(n: int) -> List[Tuple[str, str]]:
    """(modality, family) list; n=60 uses PLAN exactly, other n scale it."""
    full = [(m, f) for m in ("text", "audio", "visual") for f, k in PLAN[m] for _ in range(k)]
    if n == len(full):
        return full
    target = {"text": round(n * 0.5), "audio": round(n * 0.3)}
    target["visual"] = n - target["text"] - target["audio"]
    out = []
    for m in ("text", "audio", "visual"):
        pool = [x for x in full if x[0] == m]
        out += [pool[int(i * len(pool) / max(1, target[m]))] for i in range(target[m])]
    return out


def generate(n: int = 60, seed: int = 5) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    out, counter = [], {}
    for i, (mod, fam) in enumerate(plan_for(n), start=1):
        v = counter.get((mod, fam), 0)
        counter[(mod, fam)] = v + 1
        # alternate variants across modalities so audio does not replay text variants
        v += {"text": 0, "audio": 1, "visual": 0}[mod]
        sid = f"g{i:02d}_{mod}_{fam}"
        b = B(sid, mod, fam, f"{fam.replace('_', ' ')} #{v}", random.Random(rng.random()))
        out.append(FAMILIES[fam](b, v))
    return out


def write(scenarios: List[Dict[str, Any]], out_dir: Path, assets: bool = True) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for sc in scenarios:
        p = out_dir / f"{sc['id']}.json"
        p.write_text(json.dumps(sc, indent=1) + "\n", encoding="utf-8")
        paths.append(p)
    if assets:
        try:
            from sim import assets as sim_assets
            sim_assets.generate(out_dir)
        except ImportError:  # harness not present: media refs stay dangling
            pass
    return paths


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "scenarios60")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--no-assets", action="store_true")
    a = ap.parse_args(argv)
    scs = generate(a.n, a.seed)
    write(scs, a.out, assets=not a.no_assets)
    mix = {m: sum(1 for s in scs if s["modality"] == m) for m in ("text", "audio", "visual")}
    print(f"wrote {len(scs)} scenarios to {a.out} {mix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
