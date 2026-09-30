# /mnt/project-files/theme5/tests/test_agent.py
import asyncio

from theme5.agent import Agent
from theme5.protocol import from_wire, validate_action

SEARCH = {"name": "search_flights", "description": "Search flights", "read_only": True,
          "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"},
                                                          "date": {"type": "string"}}, "required": ["origin", "destination", "date"]}}
BOOK = {"name": "book_flight", "description": "Book a flight", "state_modifying": True,
        "parameters": {"type": "object", "properties": {"flight_id": {"type": "string"}, "passengers": {"type": "integer"}},
                       "required": ["flight_id", "passengers"]}}
MODIFY = {"name": "modify_booking", "description": "Change a booking", "state_modifying": True,
          "parameters": {"type": "object", "properties": {"booking_id": {"type": "string"}, "passengers": {"type": "integer"}},
                         "required": ["booking_id", "passengers"]}}
FLIGHTS = {"flights": [{"flight_id": "F1", "price": 5}, {"flight_id": "F2", "price": 3}]}


def manifest(*tools):
    return {"type": "tool_manifest", "t": 0, "tools": list(tools), "date": "2026-10-01"}


def text(t, s):
    return {"type": "text", "t": t, "text": s, "end_of_turn": True}


def result(t, cid, res=None, **kw):
    return {"type": "tool_result", "t": t, "call_id": cid, **({"result": res} if res is not None else {}), **kw}


def run(events, agent=None):
    agent = agent or Agent()

    async def go():
        inq, outq = asyncio.Queue(), asyncio.Queue()
        for e in events:
            inq.put_nowait(e)
        inq.put_nowait(None)
        await agent.run(inq, outq)
        out = []
        while not outq.empty():
            out.append(outq.get_nowait())
        return out

    wire = asyncio.run(go())
    return agent, wire, [from_wire(a) for a in wire]


def types(acts):
    return [a["type"] for a in acts]


def calls(acts):
    return [a for a in acts if a["type"] == "tool_call"]


def test_every_action_valid_and_wire_uses_kit_names():
    _, wire, acts = run([manifest(SEARCH), text(0, "find flights from Delhi to Mumbai tomorrow")])
    assert all(validate_action(a) == [] for a in acts)
    call = next(a for a in wire if a["type"] == "tool_call")
    assert call["name"] == "search_flights" and call["arguments"]["date"] == "2026-10-02"
    assert types(acts)[0] == "speak"  # substantive speech first, then the call


def test_correction_cancels_first_then_replans():
    _, _, acts = run([manifest(SEARCH), text(0, "find flights from Delhi to Mumbai tomorrow"),
                      {"type": "interrupt", "t": 300}, text(350, "no wait, to Pune"),
                      result(1000, "call-0001", FLIGHTS), result(1400, "call-0002", FLIGHTS)])
    after = [a for a in acts if a["t"] >= 350]
    assert after[0]["type"] == "cancel" and after[0]["call_id"] == "call-0001"
    assert calls(acts)[1]["args"]["destination"] == "Pune" and calls(acts)[1]["args"]["origin"] == "Delhi"
    finals = [a for a in acts if a["type"] == "final_response"]
    assert len(finals) == 1 and abs(finals[0]["t"] - 1400) < 50  # stale result at 1000 ignored
    assert finals[0]["state_snapshot"]["slots"]["destination"] == "Pune"


def test_chain_search_then_book_and_no_duplicate_write():
    _, _, acts = run([manifest(SEARCH, BOOK),
                      text(0, "book the cheapest flight from Delhi to Goa tomorrow for 2 passengers"),
                      result(500, "call-0001", FLIGHTS),
                      text(600, "book the cheapest flight from Delhi to Goa tomorrow for 2 passengers"),
                      result(900, "call-0002", {"booking_id": "B1"}),
                      text(1000, "book the cheapest flight from Delhi to Goa tomorrow for 2 passengers")])
    books = [c for c in calls(acts) if c["tool"] == "book_flight"]
    assert len(books) == 1 and books[0]["args"] == {"flight_id": "F2", "passengers": 2}
    finals = [a for a in acts if a["type"] == "final_response"]
    assert len(finals) == 2 and "B1" in finals[0]["text"]
    assert "already done" in finals[1]["text"] and "B1" in finals[1]["text"]  # repeat request: restated, not redone
    assert finals[0]["state_snapshot"]["intent"] == "book_flight"


def test_change_in_flight_write_cancels_and_rebooks_once():
    _, _, acts = run([manifest(SEARCH, BOOK), text(0, "book a flight from Delhi to Goa tomorrow for 2 passengers"),
                      result(500, "call-0001", FLIGHTS), text(600, "actually make it 3 passengers"),
                      result(900, "call-0002", {"booking_id": "OLD"}), result(1200, "call-0003", {"booking_id": "NEW"})])
    books = [c for c in calls(acts) if c["tool"] == "book_flight"]
    assert [b["args"]["passengers"] for b in books] == [2, 3]
    assert any(a["type"] == "cancel" and a["call_id"] == "call-0002" for a in acts)
    assert "NEW" in [a for a in acts if a["type"] == "final_response"][-1]["text"]


def test_change_after_commit_without_modify_tool_never_rebooks():
    _, _, acts = run([manifest(SEARCH, BOOK), text(0, "book a flight from Delhi to Goa tomorrow for 2 passengers"),
                      result(500, "call-0001", FLIGHTS), result(900, "call-0002", {"booking_id": "B1"}),
                      text(1000, "actually make it 3 passengers")])
    assert len([c for c in calls(acts) if c["tool"] == "book_flight"]) == 1
    assert acts[-1]["type"] == "speak" and "already" in acts[-1]["text"]


def test_change_after_commit_uses_modify_tool():
    _, _, acts = run([manifest(SEARCH, BOOK, MODIFY), text(0, "book a flight from Delhi to Goa tomorrow for 2 passengers"),
                      result(500, "call-0001", FLIGHTS), result(900, "call-0002", {"booking_id": "B1"}),
                      text(1000, "actually make it 3 passengers")])
    mod = [c for c in calls(acts) if c["tool"] == "modify_booking"]
    assert len(mod) == 1 and mod[0]["args"] == {"booking_id": "B1", "passengers": 3}


def test_cancel_everything():
    agent, _, acts = run([manifest(SEARCH), text(0, "find flights from Delhi to Mumbai tomorrow"), text(100, "never mind")])
    assert types(acts)[-2:] == ["cancel", "final_response"]
    assert acts[-1]["state_snapshot"] == {"intent": None, "slots": {}} and "No changes were made" in acts[-1]["text"]
    assert agent.state.intent is None and not agent.calls.inflight


def test_read_retry_then_failure_is_truthful():
    _, _, acts = run([manifest(SEARCH), text(0, "find flights from Delhi to Mumbai tomorrow"),
                      result(100, "call-0001", error="timeout"), result(200, "call-0002", error="boom"),
                      result(300, "call-0003", error="boom")])
    assert len(calls(acts)) == 3
    assert acts[-1]["type"] == "final_response" and "couldn't" in acts[-1]["text"]


def test_write_non_retryable_failure_is_not_retried():
    _, _, acts = run([manifest(SEARCH, BOOK), text(0, "book a flight from Delhi to Goa tomorrow for 1 passenger"),
                      result(500, "call-0001", FLIGHTS), result(900, "call-0002", error="card declined")])
    assert len([c for c in calls(acts) if c["tool"] == "book_flight"]) == 1
    assert "nothing was changed" in acts[-1]["text"]


def test_low_confidence_audio_confirms_then_proceeds():
    _, _, acts = run([manifest(SEARCH),
                      {"type": "audio", "t": 0, "path": "a.wav", "transcript": "find flights from Delhi to Goa tomorrow", "confidence": 0.3},
                      text(500, "yes")])
    assert acts[0]["type"] == "clarify" and "did you say" in acts[0]["text"].lower()
    assert calls(acts) and calls(acts)[0]["args"]["destination"] == "Goa"


def test_frame_then_question_grounds_lookup():
    manual = {"name": "manual_lookup", "description": "Manual lookup from a frame", "read_only": True,
              "parameters": {"type": "object", "properties": {"frame_id": {"type": "string"}, "query": {"type": "string"}},
                             "required": ["frame_id", "query"]}}
    _, _, acts = run([manifest(manual), {"type": "frame", "t": 0, "frame_id": "f9", "caption": "tv"},
                      text(50, "what does this blinking light mean")])
    assert calls(acts)[0]["args"]["frame_id"] == "f9"


def test_clarify_answer_binds_to_asked_slot_and_snapshot_uses_param_names():
    weather = {"name": "get_weather", "description": "Weather for a city", "read_only": True,
               "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
    _, _, acts = run([manifest(weather), text(0, "what's the weather like"), text(100, "Chennai"),
                      result(200, "call-0001", {"summary": "Rain"})])
    assert acts[0]["type"] == "clarify"
    assert calls(acts)[0]["args"] == {"city": "Chennai"}
    assert acts[-1]["state_snapshot"] == {"intent": "get_weather", "slots": {"city": "Chennai"}}


def test_cumulative_chunks_and_unknown_events_are_tolerated():
    agent, _, acts = run([manifest(SEARCH), {"type": "mystery", "t": 0},
                          {"type": "text", "t": 1, "text": "find flights"},
                          {"type": "text", "t": 2, "text": "find flights from Delhi to Goa tomorrow <EOT>"}])
    assert calls(acts)[0]["args"] == {"origin": "Delhi", "destination": "Goa", "date": "2026-10-02"}
    assert any(t.get("type") == "_ignored_event" for t in agent.trace)


def test_unintelligible_input_asks_instead_of_guessing():
    _, _, acts = run([manifest(SEARCH), text(0, "blorp")])
    assert types(acts) == ["clarify"]
