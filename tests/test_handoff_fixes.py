# /mnt/project-files/theme5/tests/test_handoff_fixes.py
"""Unit tests for the fixes driven by the 60-scenario bench: unseen-tool
binding, A-B-A reverts, display codes, timed-out writes, compound requests."""
from __future__ import annotations

import asyncio
from datetime import date

from theme5 import nlg, nlu
from theme5.agent import Agent
from theme5.clock import VirtualClock
from theme5.planner import fill_args, goal_tool, schema_values, slot_for_param
from theme5.protocol import ParamSpec, parse_event, parse_tool_def, parse_tool_result
from theme5.tools import ToolRegistry

REF = date(2026, 9, 29)


def tool(name, props, required, **flags):
    return parse_tool_def({"name": name, "parameters": {"type": "object", "properties": props, "required": required},
                           **flags}).spec


def test_odd_param_names_map_to_slots():
    assert slot_for_param(ParamSpec("venueSlug", description="Restaurant name or slug")) == "location"
    assert slot_for_param(ParamSpec("party_sz")) == "passengers"
    assert slot_for_param(ParamSpec("when_iso")) == "date"
    assert slot_for_param(ParamSpec("guest_name")) == "name"
    assert slot_for_param(ParamSpec("x", description="Number of guests")) == "passengers"
    assert slot_for_param(ParamSpec("customerId")) is None


def test_date_and_time_combine_into_one_iso_param():
    t = tool("reserve_table", {"venueSlug": {"type": "string"}, "party_sz": {"type": "integer"},
                               "when_iso": {"type": "string", "description": "Date and time, ISO 8601"}},
             ["venueSlug", "party_sz", "when_iso"], side_effect="write")
    args, missing = fill_args(t, {"location": "Bombay Canteen", "passengers": 2, "date": "2026-10-12", "time": "20:00"})
    assert not missing and args == {"venueSlug": "Bombay Canteen", "party_sz": 2, "when_iso": "2026-10-12T20:00"}


def test_schema_values_pattern_enum_currency_ids():
    awb = tool("trk_pkg", {"awb_no": {"type": "string", "pattern": "^[0-9]{10}$"}}, ["awb_no"], read_only=True)
    assert schema_values(awb, "The waybill is 7 6 3 5 9 4 5 6 8 7.") == {"awb_no": "7635945687"}
    hv = tool("hvac_setpoint", {"zone_label": {"type": "string", "enum": ["living", "bedroom"]},
                                "target_c": {"type": "number"}}, ["zone_label", "target_c"])
    assert schema_values(hv, "Set the living room to 24 degrees.") == {"zone_label": "living", "target_c": 24}
    fx = tool("fx_quote", {"amt": {"type": "number"}, "frm": {"type": "string", "description": "ISO 4217 source"},
                           "to_ccy": {"type": "string", "description": "ISO 4217 target"}}, ["amt", "frm", "to_ccy"])
    assert schema_values(fx, "How much is 120 US dollars in rupees?") == {"frm": "USD", "to_ccy": "INR", "amt": 120}
    lic = tool("renewLicence", {"LicenceNumber": {"type": "string"}, "yrs": {"type": "integer", "enum": [1, 5]}},
               ["LicenceNumber", "yrs"])
    assert schema_values(lic, "Renew my driving licence KA743309996 for one year.") == \
        {"LicenceNumber": "KA743309996", "yrs": 1}
    assert schema_values(hv, "Book a flight on October 12th") == {}  # dates are not amounts


def test_goal_tool_uses_concepts_and_schema_evidence():
    reg = ToolRegistry()
    reg.load([tool("flight_search", {"origin": {"type": "string"}, "destination": {"type": "string"}},
                   ["origin", "destination"], read_only=True),
              tool("fx_quote", {"amt": {"type": "number"},
                                "frm": {"type": "string", "description": "ISO 4217 source"},
                                "to_ccy": {"type": "string", "description": "ISO 4217 target"}},
                   ["amt", "frm", "to_ccy"], kind="query", description="Quote a currency conversion.")])
    assert goal_tool(None, "How much is 120 US dollars in rupees?", reg).name == "fx_quote"
    assert goal_tool(None, "What a lovely day", reg) is None


def test_aba_revert_phrase_yields_the_bare_value():
    p = nlu.parse("No no, sorry, Kolkata was right.", "destination", REF)
    assert p.correction and p.slots == {"destination": "Kolkata"}


def test_display_codes_keep_their_case_and_models_are_read():
    assert nlu.parse("It's the microwave, model MS23K, showing C-d0.", "device", REF).slots == \
        {"device": "microwave", "error_code": "C-d0", "device_model": "MS23K"}
    assert nlu.parse("It's the dishwasher, showing LC.", "device", REF).slots["error_code"] == "LC"
    assert "error_code" not in nlu.parse("It says OK now", None, REF).slots
    assert nlu.detect_intent("Can you help me fix this?") == "lookup_manual"


def test_timeout_marks_outcome_unknown():
    tr = parse_tool_result(parse_event({"type": "tool_result", "call_id": "c1", "status": "timeout"}))
    assert not tr.ok and tr.outcome_unknown
    tr = parse_tool_result(parse_event({"type": "tool_result", "call_id": "c1", "error": "503 unavailable"}))
    assert not tr.outcome_unknown


def _run(agent, events):
    async def go():
        inbox, outbox = asyncio.Queue(), asyncio.Queue()
        for e in events:
            inbox.put_nowait(e)
        inbox.put_nowait(None)
        await agent.run(inbox, outbox)
        out = []
        while not outbox.empty():
            out.append(outbox.get_nowait())
        return out
    return asyncio.run(go())


BOOK = {"name": "book_flight", "parameters": {"type": "object", "required": ["flight_id", "passenger_name"],
                                              "properties": {"flight_id": {"type": "string"},
                                                             "passenger_name": {"type": "string"}}}}


def _agent():
    return Agent(clock=VirtualClock())


def test_timed_out_write_is_not_retried_and_is_reported_honestly():
    a = _agent()
    out = _run(a, [{"type": "tool_manifest", "t": 0, "tools": [BOOK]},
                   {"type": "text", "t": 0, "text": "Book flight AI-202 for Rahul Verma.", "end_of_turn": True}])
    calls = [x for x in out if x["type"] == "tool_call"]
    assert len(calls) == 1
    out = _run(a, [{"type": "tool_result", "t": 4000, "call_id": calls[0]["call_id"], "status": "timeout"}])
    assert not [x for x in out if x["type"] == "tool_call"]
    final = [x for x in out if x["type"] == "final_response"][-1]["text"]
    assert "timed out" in final and "can't confirm" in final
    assert a.engine.can_call(a.registry.get("book_flight"),
                             {"flight_id": "AI-202", "passenger_name": "Rahul Verma"}) is not None


def test_compound_request_runs_each_part_and_reports_all():
    assert nlu.split_compound("Book UK-1 for Aditya Rao, and also UK-1 for Arjun Rao.") == \
        ["Book UK-1 for Aditya Rao", "UK-1 for Arjun Rao"]
    assert len(nlu.split_compound("Find flights to Goa and then book the cheapest")) == 1
    a = _agent()
    out = _run(a, [{"type": "tool_manifest", "t": 0, "tools": [BOOK]},
                   {"type": "text", "t": 0, "text": "Book AI-202 for Aditya Rao, and also AI-202 for Arjun Rao.",
                    "end_of_turn": True}])
    c1 = [x for x in out if x["type"] == "tool_call"][0]
    out = _run(a, [{"type": "tool_result", "t": 900, "call_id": c1["call_id"], "result": {"booking_ref": "AAA111"}}])
    c2 = [x for x in out if x["type"] == "tool_call"][0]
    assert c2["arguments"]["passenger_name"] == "Arjun Rao" and not [x for x in out if x["type"] == "final_response"]
    out = _run(a, [{"type": "tool_result", "t": 1800, "call_id": c2["call_id"], "result": {"booking_ref": "BBB222"}}])
    final = [x for x in out if x["type"] == "final_response"][-1]["text"]
    assert "AAA111" in final and "BBB222" in final


def test_write_final_names_unseen_reference_and_new_facts():
    t = tool("renewLicence", {"LicenceNumber": {"type": "string"}}, ["LicenceNumber"], mutating=True)
    text = nlg.final(t, {"LicenceNumber": "KA1"}, {"receipt": "RCPT-8812", "valid_until": "2031-09-29"}, None)
    assert "RCPT-8812" in text and "2031-09-29" in text and "licence is renewed" in text
    assert "Nothing was booked" not in nlg.cancelled() and "confirmed" not in nlg.cancelled(True)


def test_repeated_digits_survive_stutter_removal():
    assert nlu.clean_text("The waybill is 3 4 2 7 5 3 3 5 0 8 to to Goa") == "The waybill is 3 4 2 7 5 3 3 5 0 8 to Goa"
    awb = tool("trk_pkg", {"awb_no": {"type": "string", "pattern": "^[0-9]{10}$"}}, ["awb_no"], read_only=True)
    assert schema_values(awb, nlu.clean_text("waybill 3 4 2 7 5 3 3 5 0 8")) == {"awb_no": "3427533508"}


def test_repeat_request_after_write_restates_it_as_a_final():
    a = _agent()
    out = _run(a, [{"type": "tool_manifest", "t": 0, "tools": [BOOK]},
                   {"type": "text", "t": 0, "text": "Please book AI-5828 for Ananya Das.", "end_of_turn": True}])
    c = [x for x in out if x["type"] == "tool_call"][0]
    _run(a, [{"type": "tool_result", "t": 900, "call_id": c["call_id"], "result": {"booking_ref": "ZVM8M5"}}])
    out = _run(a, [{"type": "text", "t": 3000, "text": "Great, book it.", "end_of_turn": True}])
    assert not [x for x in out if x["type"] == "tool_call"]
    final = [x for x in out if x["type"] == "final_response"][-1]
    assert "already done" in final["text"] and "ZVM8M5" in final["text"]
    assert final["state_snapshot"]["slots"]["flight_id"] == "AI-5828"
