# /mnt/project-files/theme5/tests/test_slots.py
"""Utterance sequences -> expected State Snapshots for theme5.slots."""
from __future__ import annotations

import asyncio
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from theme5 import slots as S  # noqa: E402
from theme5.protocol import ACT_FINAL, make_action, validate_action  # noqa: E402
from theme5.slots import HybridParser, SlotTracker  # noqa: E402
from theme5.state import SessionState  # noqa: E402


def run(*turns: str, tracker: SlotTracker | None = None) -> tuple[SlotTracker, list[S.SlotUpdate]]:
    t = tracker or SlotTracker()
    ups = [t.feed(u, chunk_id=f"c{i}", t=float(i * 1000)) for i, u in enumerate(turns)]
    return t, ups


def snap(t: SlotTracker) -> dict[str, Any]:
    return t.snapshot()["slots"]


# ---------------------------------------------------------------- the three brief examples

def test_01_no_wait_changes_only_destination():
    t, (u,) = run("Book Delhi to Mumbai on the 5th... no wait, Pune")
    assert t.snapshot() == {"intent": "book_flight",
                            "slots": {"origin": "Delhi", "destination": "Pune", "date": "5th"}}
    assert t.memory.past_values("destination") == ["Mumbai", "Pune"]


def test_02_repair_in_later_chunk_rewrites_only_destination():
    t = SlotTracker()
    u1 = t.feed("Book Delhi to Mumbai on the 5th...", chunk_id="c1", end_of_turn=False)
    assert u1.snapshot["slots"]["destination"] == "Mumbai" and u1.clarification is None
    u2 = t.feed("no wait, Pune", chunk_id="c2", end_of_turn=True)
    assert u2.changed == {"destination": "Pune"}
    assert u2.transitions == {"destination": ("Mumbai", "Pune")}
    prov = t.memory.provenance()
    assert prov["destination"]["source"] == "c2" and prov["origin"]["source"] == "c1"


def test_03_make_it_two_passengers_changes_only_passengers():
    t, ups = run("Book Delhi to Mumbai on the 5th... no wait, Pune", "Make it two passengers")
    assert ups[1].changed == {"passengers": 2}
    assert snap(t) == {"origin": "Delhi", "destination": "Pune", "date": "5th", "passengers": 2}


def test_04_earlier_date_with_no_prior_date_asks():
    t, ups = run("Book Delhi to Pune on the 5th", "Actually the earlier date")
    assert ups[1].changed == {}
    c = ups[1].clarification
    assert c is not None and c.slot == "date" and c.kind == "missing"
    assert snap(t)["date"] == "5th"


def test_05_earlier_date_unambiguous_reverts():
    # 5th then 10th: "earlier" means the 5th in both readings (said first, and sooner)
    t, ups = run("Book Delhi to Pune on the 5th", "Change the date to the 10th", "Actually the earlier date")
    assert ups[2].changed == {"date": "5th"} and ups[2].clarification is None
    assert ups[2].corrected == {"date"}


def test_06_earlier_date_ambiguous_asks_then_resolves():
    # 10th then 5th: "earlier date" = the 10th (said earlier) or the 5th (sooner)
    t, ups = run("Book Delhi to Pune on the 10th", "Sorry, the 5th", "Actually the earlier date")
    c = ups[2].clarification
    assert c is not None and c.kind == "value" and set(c.options) == {"10th", "5th"}
    assert ups[2].changed == {} and snap(t)["date"] == "5th"
    u = t.feed("the 10th")
    assert u.changed == {"date": "10th"} and t.pending is None
    assert t.memory.provenance()["date"]["via"] == "clarify"


def test_07_three_dates_earlier_asks_and_ordinal_answer_picks():
    t, ups = run("Flights from Delhi to Goa on the 3rd", "Actually the 7th", "Actually the 9th",
                 "Hmm, the earlier date")
    c = ups[3].clarification
    assert c is not None and c.options == ("3rd", "7th", "9th")
    t.feed("the first one")
    assert snap(t)["date"] == "3rd" and t.pending is None


def test_08_previous_destination_is_unambiguous():
    t, ups = run("Book Delhi to Mumbai", "Actually Pune", "Actually Goa", "No, the previous destination")
    assert ups[3].changed == {"destination": "Pune"}


# ---------------------------------------------------------------- hesitations and self-repair

def test_09_hesitation_stutter_and_i_mean():
    t, _ = run("um, to, to Goa, I mean Goa on Friday")
    assert snap(t) == {"destination": "Goa", "date": "friday"}


def test_10_truncated_word_and_fillers():
    t, _ = run("Search flights from Mum- Mumbai to uh Chennai")
    assert t.snapshot() == {"intent": "search_flights", "slots": {"origin": "Mumbai", "destination": "Chennai"}}


def test_11_date_self_repair_in_one_utterance():
    t, _ = run("Book Delhi to Goa on the 5th, sorry, the 6th")
    assert snap(t) == {"origin": "Delhi", "destination": "Goa", "date": "6th"}


def test_12_value_anchored_correction_targets_the_holder():
    t, ups = run("Book Mumbai to Delhi on Friday", "No, not Mumbai, Pune")
    assert ups[1].changed == {"origin": "Pune"}
    assert snap(t) == {"origin": "Pune", "destination": "Delhi", "date": "friday"}


def test_13_slot_noun_beats_preposition():
    t, ups = run("Book Delhi to Mumbai", "Change the origin to Chennai")
    assert ups[1].changed == {"origin": "Chennai"}


def test_14_scratch_that_discards_previous_clause():
    t, _ = run("Book a flight from Delhi to Goa. Scratch that")
    assert snap(t) == {}
    t2, _ = run("Book a flight to Goa, scratch that, to Pune")
    assert snap(t2) == {"destination": "Pune"}


def test_15_bare_city_repair_with_unknown_city_name():
    t, ups = run("Book Delhi to Mumbai", "No, Shillong")
    assert ups[1].changed == {"destination": "Shillong"}


def test_16_repaired_name():
    t, _ = run("Book Delhi to Goa, my name is Rahul, sorry, Rohan")
    assert snap(t)["name"] == "Rohan"


# ---------------------------------------------------------------- counts

def test_17_relative_passenger_changes():
    t, ups = run("Book Delhi to Goa for two people", "Add one more passenger", "Actually one less passenger")
    assert [u.changed for u in ups[1:]] == [{"passengers": 3}, {"passengers": 2}]


def test_18_make_it_bare_number_and_just_me():
    t, ups = run("Book Delhi to Goa, just me", "Make it three")
    assert ups[0].snapshot["slots"]["passengers"] == 1
    assert ups[1].changed == {"passengers": 3}


def test_18b_first_one_is_a_result_choice_when_nothing_is_pending():
    t, ups = run("Find flights from Delhi to Goa", "The first one")
    assert ups[1].changed == {"choice": 1} and ups[1].clarification is None


# ---------------------------------------------------------------- ambiguity

def test_19_bare_city_with_both_filled_asks_role_then_applies():
    t, ups = run("Book Delhi to Mumbai", "Pune")
    c = ups[1].clarification
    assert c is not None and c.kind == "role" and c.value == "Pune" and ups[1].changed == {}
    u = t.feed("as the destination")
    assert u.changed == {"destination": "Pune"} and t.pending is None


def test_20_bare_city_on_empty_state_guesses_destination_low_confidence():
    t, _ = run("Goa")
    assert snap(t) == {"destination": "Goa"}
    assert t.memory.slots["destination"].confidence == pytest.approx(S.CONF_DEFAULT)
    assert "destination" in t.memory.low_confidence(0.7)


# ---------------------------------------------------------------- goal changes

def test_21_search_to_ticket_keeps_name_drops_flight_slots():
    t, ups = run("Search flights from Delhi to Goa tomorrow, my name is Rahul Sharma",
                 "Actually, raise a support ticket about my refund instead")
    u = ups[1]
    assert t.snapshot() == {"intent": "create_ticket", "slots": {"name": "Rahul Sharma", "issue": "my refund"}}
    assert u.intent_changed and u.dropped == {"origin", "destination", "date"}
    assert set(t.memory.parked) == {"origin", "destination", "date"}


def test_22_goal_change_back_restores_parked_slots():
    t, _ = run("Search flights from Delhi to Goa, my name is Rahul",
               "Actually open a support ticket", "Sorry, book the flight after all")
    assert t.snapshot() == {"intent": "book_flight",
                            "slots": {"name": "Rahul", "origin": "Delhi", "destination": "Goa"}}


def test_23_search_to_book_keeps_everything():
    t, ups = run("Find flights from Delhi to Goa on the 5th for 2 passengers", "Great, book it")
    assert ups[1].intent_changed and ups[1].dropped == set() and ups[1].changed == {}
    assert snap(t) == {"origin": "Delhi", "destination": "Goa", "date": "5th", "passengers": 2}


def test_24_unseen_tool_schema_from_manifest():
    from theme5.protocol import ParamSpec, ToolSpec
    tool = ToolSpec("reserve_hotel", "", (ParamSpec("city", required=True), ParamSpec("guest_name"),
                                          ParamSpec("check_in_date")), True)
    t, _ = run("Book Delhi to Goa on the 5th, my name is Rahul")
    t.register_tools([tool])
    u = t.set_intent("reserve_hotel")
    assert u.dropped == {"origin", "destination"}
    assert snap(t) == {"date": "5th", "name": "Rahul"}


# ---------------------------------------------------------------- provenance, confidence, chunks

def test_25_asr_confidence_scales_slot_confidence():
    t = SlotTracker()
    t.feed("Book a flight from Delhi to Goa", chunk_id="a1", confidence=0.5)
    sv = t.memory.slots["destination"]
    assert sv.source == "a1" and sv.confidence == pytest.approx(S.CONF_CUE * 0.5)
    assert set(t.memory.low_confidence()) == {"origin", "destination"}


def test_26_partial_chunk_never_commits_a_half_word():
    t = SlotTracker()
    u1 = t.feed("Book a flight from Delhi to Shil", chunk_id="c1", end_of_turn=False)
    assert "destination" not in u1.snapshot["slots"]
    u2 = t.feed(" Shillong on the 5th", chunk_id="c2", end_of_turn=True)
    # the harness chunks on word boundaries; "Shil Shillong" collapses nothing, the cue binds the full name
    assert u2.snapshot["slots"]["origin"] == "Delhi"
    assert u2.snapshot["slots"]["date"] == "5th"


def test_27_reference_date_gives_iso_dates():
    t = SlotTracker(ref_date=date(2026, 9, 29))
    t.feed("Book Delhi to Goa on the 5th")
    assert snap(t)["date"] == "2026-10-05"
    t.feed("actually tomorrow")
    assert snap(t)["date"] == "2026-09-30"


def test_28_return_date_and_ticket_fields():
    t, _ = run("Flights from Delhi to Goa on the 5th returning on the 12th")
    assert snap(t)["return_date"] == "12th" and snap(t)["date"] == "5th"
    t2, _ = run("Open a support ticket, booking reference is ab12cd, email me at R.Sharma@mail.com, it's urgent")
    assert snap(t2) == {"booking_ref": "AB12CD", "email": "r.sharma@mail.com", "priority": "high"}


# ---------------------------------------------------------------- session scope and protocol

def test_29_sessions_share_nothing():
    a, _ = run("Book Delhi to Goa, my name is Rahul")
    b = SlotTracker()
    assert b.snapshot() == {"intent": None, "slots": {}}
    a.reset()
    assert a.snapshot() == {"intent": None, "slots": {}} and a.memory.history == {} and not a.memory.parked


def test_30_snapshot_is_protocol_valid_and_mirrors_into_session_state():
    t = SlotTracker()
    st = SessionState()
    for u in ("Book Delhi to Mumbai on the 5th, no wait, Pune", "Make it two passengers"):
        t.feed(u).apply_to(st)
    assert st.snapshot() == t.snapshot()
    act = make_action(ACT_FINAL, 1.0, snapshot=t.snapshot(), text="Booked.")
    assert validate_action(act) == []


# ---------------------------------------------------------------- LLM hook

class FakeLLM:
    def __init__(self, out: dict[str, Any] | None = None, delay: float = 0.0) -> None:
        self.out, self.delay, self.calls = out, delay, 0

    async def parse(self, utterance, snapshot, tools):  # noqa: ANN001
        self.calls += 1
        await asyncio.sleep(self.delay)
        return self.out


def test_31_llm_only_when_rules_find_nothing_and_capped():
    llm = FakeLLM({"intent": "book_flight", "slots": {"destination": "Leh", "bogus": "x", "passengers": 2}})
    t = SlotTracker(parser=HybridParser(llm))
    asyncio.run(t.feed_async("Book Delhi to Goa"))
    assert llm.calls == 0
    asyncio.run(t.feed_async("the mountain place with the monastery, for me and my mum"))
    assert llm.calls == 1
    assert snap(t)["destination"] == "Leh" and snap(t)["passengers"] == 2 and "bogus" not in snap(t)
    assert t.memory.slots["destination"].via == "llm"
    assert t.memory.slots["destination"].confidence <= S.LLM_MAX_CONF


def test_32_llm_timeout_falls_back_to_rules():
    llm = FakeLLM({"slots": {"destination": "Leh"}}, delay=1.0)
    t = SlotTracker(parser=HybridParser(llm, timeout_s=0.05))
    u = asyncio.run(t.feed_async("the mountain place"))
    assert u.changed == {} and snap(t) == {}


def test_33_fast_path_feed_never_calls_llm():
    llm = FakeLLM({"slots": {"destination": "Leh"}})
    t = SlotTracker(parser=HybridParser(llm))
    t.feed("the mountain place")
    assert llm.calls == 0
