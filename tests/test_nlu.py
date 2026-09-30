# /mnt/project-files/theme5/tests/test_nlu.py
from datetime import date

import pytest

from theme5 import nlu

REF = date(2026, 10, 1)  # a Thursday


def test_clean_text_removes_fillers_and_stutters():
    assert nlu.clean_text("um book a a flight uh to to Pune") == "book a flight to Pune"


def test_route_date_passengers():
    p = nlu.parse("Find flights from New Delhi to Mumbai tomorrow for two people", ref=REF)
    assert p.intent == "search_flights"
    assert p.slots == {"origin": "New Delhi", "destination": "Mumbai", "date": "2026-10-02", "passengers": 2}


def test_self_repair_changes_only_the_repaired_slot():
    p = nlu.parse("book a flight from Delhi to Mumbai on 5th October, no wait, to Pune", ref=REF)
    assert p.correction and p.intent == "book_flight"
    assert p.slots["destination"] == "Pune" and p.slots["origin"] == "Delhi"
    assert p.slots["date"] == "2026-10-05" and "choice" not in p.slots


@pytest.mark.parametrize("text,last,expected", [
    ("actually make it three passengers", None, {"passengers": 3}),
    ("I mean Chennai", "destination", {"destination": "Chennai"}),
    ("sorry, from Jaipur", "destination", {"origin": "Jaipur"}),
    ("change the destination to Bangalore", None, {"destination": "Bangalore"}),
    ("no wait, Friday", "date", {"date": "2026-10-02"}),
])
def test_corrections(text, last, expected):
    p = nlu.parse(text, last_slot=last, ref=REF)
    assert p.correction
    assert p.slots == expected


def test_negated_value_ignored():
    assert nlu.parse("fly to Mumbai, not Delhi").slots == {"destination": "Mumbai"}


def test_cancel_vs_cancel_and_replan():
    assert nlu.parse("cancel that").cancel
    assert nlu.parse("never mind").cancel
    p = nlu.parse("never mind, search flights to Goa instead")
    assert not p.cancel and p.intent == "search_flights" and p.slots == {"destination": "Goa"}
    assert nlu.parse("please cancel my booking").intent == "cancel_booking"


def test_intents_and_misc_slots():
    p = nlu.parse("my washing machine shows error E4, what does this light mean")
    assert p.intent == "lookup_manual" and p.slots["device"] == "washing machine" and p.slots["error_code"] == "E4"
    p = nlu.parse("My TV is not working, raise a ticket, my name is Ravi Kumar")
    assert p.intent == "create_ticket" and p.slots["name"] == "Ravi Kumar" and "issue" in p.slots
    assert nlu.parse("navigate to the airport").slots["destination"] == "The Airport"
    assert nlu.parse("book the cheapest one").slots["choice"] == "cheapest"
    assert nlu.parse("business class please").slots["cabin"] == "business"
    assert nlu.parse("leave at 5 pm").slots["time"] == "17:00"
    assert nlu.parse("what's the weather in Chennai").slots["location"] == "Chennai"


def test_not_a_place_guard():
    assert "destination" not in nlu.parse("I want to book something").slots


def test_affirm_deny_thanks():
    assert nlu.parse("yes please").affirm
    assert nlu.parse("nope").deny
    assert nlu.THANKS_RE.search("thanks a lot")


def test_bare_value_for_clarifications():
    assert nlu.bare_value("Chennai", "location") == "Chennai"
    assert nlu.bare_value("it's Asha Verma", "name") == "Asha Verma"
    assert nlu.bare_value("three", "passengers") == 3
    assert nlu.bare_value("tomorrow", "date", REF) == "2026-10-02"
    assert nlu.bare_value("the door won't close", "issue") == "the door won't close"


def test_dates_without_reference_are_surface_forms():
    assert nlu.extract_date("next friday") == "next friday"
    assert nlu.extract_date("on October 12") == "october 12"
    assert nlu.extract_date("2026-12-01") == "2026-12-01"
    assert nlu.extract_date("no date here") is None
