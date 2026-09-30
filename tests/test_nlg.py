# /mnt/project-files/theme5/tests/test_nlg.py
from theme5 import nlg
from theme5.protocol import ToolSpec

SEARCH = ToolSpec("search_flights", "", (), False)
BOOK = ToolSpec("book_flight", "", (), True)
ODD = ToolSpec("frobnicate", "", (), True)


def test_progress_is_specific_and_not_a_completion_claim():
    s = nlg.progress(SEARCH, {"origin": "Delhi", "destination": "Pune", "date": "tomorrow"})
    assert s == "Searching flights from Delhi to Pune for tomorrow."
    b = nlg.progress(BOOK, {"passengers": 3}, revised=True)
    assert b.startswith("Okay, updating: booking flight") and "booked" not in b
    assert "again" in nlg.progress(BOOK, {}, retry=True)


def test_final_write_is_grounded_in_result():
    s = nlg.final(BOOK, {"passengers": 2}, {"booking_id": "BK1", "status": "confirmed"}, "book_flight")
    assert "booked" in s and "BK1" in s
    assert "succeeded" in nlg.final(ODD, {}, {}, None)


def test_final_read_summaries():
    assert nlg.final(SEARCH, {}, [], None) == "I didn't find anything for that."
    one = nlg.final(SEARCH, {}, {"answer": "Close the door."}, None)
    assert one == "Here's what I found: Close the door."
    many = nlg.final(SEARCH, {}, {"flights": [{"flight_id": "6E-1", "depart": "06:40", "price_inr": 4300},
                                              {"flight_id": "AI-2", "depart": "10:15"}]}, None)
    assert many == "I found 2 options: 6E-1 at 06:40 for 4300 and AI-2 at 10:15."
    steps = nlg.final(SEARCH, {}, {"section": "Error 4C", "steps": ["Open the tap.", "Check the hose"]}, None)
    assert "Error 4C" in steps and "Open the tap." in steps and "Check the hose." in steps


def test_failure_is_truthful():
    assert "nothing was changed" in nlg.failure(BOOK, "declined")
    assert "couldn't" in nlg.failure(SEARCH, "x")
