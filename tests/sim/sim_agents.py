# tests/sim/sim_agents.py
"""Scripted agents for harness tests.

ScriptAgent reacts to triggers with fixed actions:
  "<event_id>"                   a scripted event was delivered
  "result:<call_id>:<status>"    a tool_result arrived
  {"type": "sleep", "ms": n}     inside an action list waits n virtual ms
ORACLE holds an ideal script per canonical scenario; every one must pass.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List


class ScriptAgent:
    def __init__(self, script: Dict[str, List[Dict[str, Any]]]):
        self.script = script
        self.seen: List[Dict[str, Any]] = []

    async def run(self, inbox: asyncio.Queue, outbox: asyncio.Queue) -> None:
        while True:
            ev = await inbox.get()
            self.seen.append(ev)
            if ev["type"] == "session_end":
                return
            if ev["type"] == "tool_result":
                key = f"result:{ev['call_id']}:{ev['status']}"
            else:
                key = ev.get("event_id", "")
            for a in self.script.get(key, []):
                if a.get("type") == "sleep":
                    await asyncio.sleep(a["ms"] / 1000.0)
                    continue
                outbox.put_nowait(dict(a))


def factory(script):
    return lambda: ScriptAgent(script)


def say(text, kind="ack"):
    return {"type": "speak", "text": text, "kind": kind}


def call(cid, tool, **args):
    return {"type": "tool_call", "call_id": cid, "tool": tool, "args": args}


def cancel(cid):
    return {"type": "cancel", "call_id": cid}


def clarify(text):
    return {"type": "clarify", "text": text}


def final(text, intent, **slots):
    return {"type": "final", "text": text, "snapshot": {"intent": intent, "slots": slots}}


ORACLE: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "s01_audio_self_correction": {
        "e1": [call("c1", "flight_search", origin="DEL", destination="BOM", date="2026-10-12")],  # speculative
        "e2": [cancel("c1"), say("Got it, Chennai. Searching Delhi to Chennai on October 12.", "progress"),
               call("c2", "flight_search", origin="DEL", destination="MAA", date="2026-10-12")],
        "result:c2:ok": [final("Earliest is 6E-2041 at 06:40.", "search_flights",
                               origin="DEL", destination="MAA", date="2026-10-12")],
    },
    "s02_text_destination_change": {
        "e2": [say("Searching Bangalore to Delhi.", "progress"),
               call("c1", "flight_search", origin="BLR", destination="DEL", date="2026-10-05")],
        "e3": [cancel("c1")],
        "e4": [say("Switching to Hyderabad.", "ack"),
               call("c2", "flight_search", origin="BLR", destination="HYD", date="2026-10-05")],
        "result:c2:ok": [final("6E-6122 at 08:20 or QP-1345 at 17:45.", "search_flights",
                               origin="BLR", destination="HYD", date="2026-10-05")],
    },
    "s03_text_chained_search_book": {
        "e2": [say("Finding the cheapest Pune to Goa flight.", "progress"),
               call("c1", "flight_search", origin="PNQ", destination="GOI", date="2026-10-20")],
        "result:c1:ok": [say("SG-208 is cheapest; booking it for Priya Sharma.", "progress"),
                         call("c2", "book_flight", flight_id="SG-208", passenger_name="Priya Sharma")],
        "result:c2:ok": [final("Booked SG-208, reference PS8K2Q.", "book_flight", flight_id="SG-208",
                               passenger_name="Priya Sharma", booking_ref="PS8K2Q")],
    },
    "s04_text_retry_after_error": {
        "e1": [say("Checking Kolkata to Delhi.", "progress"),
               call("c1", "flight_search", origin="CCU", destination="DEL", date="2026-11-02")],
        "result:c1:error": [say("The search service hiccuped, trying again.", "progress"),
                            call("c2", "flight_search", origin="CCU", destination="DEL", date="2026-11-02")],
        "result:c2:ok": [final("AI-763 at 06:05 or 6E-2257 at 14:50.", "search_flights",
                               origin="CCU", destination="DEL", date="2026-11-02")],
    },
    "s05_audio_clarification": {
        "e1": [clarify("Where are you flying from, and on what date?")],
        "e2": [say("Searching Mumbai to Delhi on October 15.", "progress"),
               call("c1", "flight_search", origin="BOM", destination="DEL", date="2026-10-15")],
        "result:c1:ok": [final("UK-955 at 08:00 or 6E-6031 at 12:25.", "search_flights",
                               origin="BOM", destination="DEL", date="2026-10-15")],
    },
    "s06_text_unseen_tool": {
        "e1": [say("Upgrading QX7P2M to business.", "progress"),
               call("c1", "upgrade_seat", booking_ref="QX7P2M", cabin="business")],
        "result:c1:ok": [final("Done: QX7P2M is now business, fare difference 7400 rupees.", "upgrade_seat",
                               booking_ref="QX7P2M", cabin="business")],
    },
    "s07_text_duplicate_booking_trap": {
        "e1": [say("Booking AI-202 for Rahul Verma.", "progress"),
               call("c1", "book_flight", flight_id="AI-202", passenger_name="Rahul Verma")],
        "e2": [say("It's still being processed; I won't book it twice.", "progress")],
        "e3": [say("Still waiting on the airline's confirmation.", "progress")],
        "result:c1:ok": [final("Confirmed: AI-202, reference RV4T9N.", "book_flight", flight_id="AI-202",
                               passenger_name="Rahul Verma", booking_ref="RV4T9N")],
    },
    "s08_audio_cancel_everything": {
        "e1": [say("Searching Chennai to Kolkata.", "progress"),
               call("c1", "flight_search", origin="MAA", destination="CCU", date="2026-10-22")],
        "e2": [cancel("c1")],
        "e3": [final("Okay, cancelled. Nothing was booked.", None)],
    },
    "s09_visual_ambiguous_frame": {
        "e2": [clarify("I can see a washer and a dryer. Which one has the error?")],
        "e4": [say("Looking up error 4C for the WW90T washer.", "progress"),
               call("c1", "manual_lookup", device_model="WW90T", query="error 4C", frame_id="s09_f2")],
        "result:c1:ok": [final("4C is a water supply error: open the tap fully and check the hose.",
                               "troubleshoot", device_model="WW90T", error_code="4C")],
    },
}


class FinalOnTurn:
    """Minimal agent for CLI tests: answers every end-of-turn with an empty final."""

    async def run(self, inbox, outbox):
        while True:
            ev = await inbox.get()
            if ev["type"] == "session_end":
                return
            if ev.get("end_of_turn"):
                outbox.put_nowait(final("Okay.", None))
