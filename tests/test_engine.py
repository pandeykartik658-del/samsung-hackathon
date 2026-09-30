# /mnt/project-files/theme5/tests/test_engine.py
"""Engine + coordinator guarantees requested for the core:
stale result dropped, late result after cancel, duplicate booking blocked,
snapshot present on every final response, virtual-clock timestamps."""
import asyncio

from theme5.agent import Agent
from theme5.clock import VirtualClock
from theme5.coordinator import LATE_AFTER_CANCEL, STALE_GENERATION
from theme5.engine import Engine
from theme5.events import FinalResponse, Speak
from theme5.protocol import ParamSpec, ToolResult, ToolSpec, from_wire, validate_action

SEARCH = ToolSpec("search_flights", "Search flights", (ParamSpec("destination", "string", True),), False)
BOOK = ToolSpec("book_flight", "Book a flight", (ParamSpec("flight_id", "string", True),), True)


class Recorder:
    """Minimal handler: records accepted results, emits a final for each."""

    def __init__(self):
        self.events, self.results = [], []
        self.engine = None

    async def on_event(self, ev):
        self.events.append(ev)

    def on_tool_result(self, call, result):
        self.results.append((call.call_id, result.result))
        self.engine.emit(FinalResponse(f"got {result.result}"))


def make(clock=None):
    h = Recorder()
    eng = Engine(h, clock=clock or VirtualClock())
    h.engine = eng
    eng._out = asyncio.Queue()
    return h, eng


def drain(q):
    out = []
    while not q.empty():
        out.append(from_wire(q.get_nowait()))
    return out


def run(coro):
    return asyncio.run(coro)


def test_stale_generation_result_is_dropped_and_logged():
    async def go():
        h, eng = make()
        c = eng.start_call(SEARCH, {"destination": "Mumbai"})
        eng.coord.next_generation()  # a re-plan that did not promote c
        eng.on_result(ToolResult(c.call_id, ok=True, result="MUMBAI"))
        return h, eng, c

    h, eng, c = run(go())
    assert h.results == []  # never acted on
    drop = eng.trace.of("dropped_result")
    assert drop and drop[-1]["reason"] == STALE_GENERATION and drop[-1]["call_id"] == c.call_id
    assert not any(a["type"] == "final_response" for a in eng.trace.actions())


def test_promoted_call_result_is_accepted():
    async def go():
        h, eng = make()
        c = eng.start_call(SEARCH, {"destination": "Pune"})
        eng.coord.next_generation()
        eng.coord.promote(c.call_id)
        eng.on_result(ToolResult(c.call_id, ok=True, result="PUNE"))
        return h

    assert run(go()).results == [("call-0001", "PUNE")]


def test_cancel_emits_within_grace_and_late_result_is_ignored():
    async def go():
        clock = VirtualClock()
        h, eng = make(clock)
        c = eng.start_call(SEARCH, {"destination": "Mumbai"})
        clock.set(450.0)
        assert eng.cancel_call(c.call_id, "superseded")
        assert not eng.cancel_call(c.call_id)  # idempotent: no second cancel action
        clock.set(1200.0)
        eng.on_result(ToolResult(c.call_id, ok=True, result="late"))
        return h, eng, c

    h, eng, c = run(go())
    acts = drain(eng._out)
    cancels = [a for a in acts if a["type"] == "cancel"]
    assert len(cancels) == 1 and cancels[0]["call_id"] == c.call_id and cancels[0]["t"] == 450.0
    assert c.status == "cancelled" and h.results == []
    assert eng.trace.of("dropped_result")[-1]["reason"] == LATE_AFTER_CANCEL


def test_duplicate_state_modifying_call_is_blocked_and_logged():
    async def go():
        h, eng = make()
        first = eng.start_call(BOOK, {"flight_id": "AI-202"})
        dup_pending = eng.start_call(BOOK, {"flight_id": " ai-202 "})  # same after normalisation
        eng.on_result(ToolResult(first.call_id, ok=True, result={"booking_ref": "R1"}))
        dup_committed = eng.start_call(BOOK, {"flight_id": "AI-202"})
        other = eng.start_call(BOOK, {"flight_id": "6E-1"})
        return eng, first, dup_pending, dup_committed, other

    eng, first, dup_pending, dup_committed, other = run(go())
    assert first is not None and other is not None
    assert dup_pending is None and dup_committed is None
    reasons = [n["reason"] for n in eng.trace.of("blocked_call")]
    assert reasons == ["duplicate_write_pending", "duplicate_write_committed"]
    assert len([a for a in drain(eng._out) if a["type"] == "tool_call" and a["tool"] == "book_flight"]) == 2


def test_failed_write_releases_key_but_cancel_does_too():
    async def go():
        h, eng = make()
        a = eng.start_call(BOOK, {"flight_id": "X"})
        eng.on_result(ToolResult(a.call_id, ok=False, error="declined"))
        b = eng.start_call(BOOK, {"flight_id": "X"})  # retry after definite failure: allowed
        eng.cancel_call(b.call_id)
        c = eng.start_call(BOOK, {"flight_id": "X"})  # ASSUMPTION: cancel-before-result == not executed
        return b, c

    b, c = run(go())
    assert b is not None and c is not None


def test_every_final_response_carries_a_valid_snapshot():
    async def go():
        h, eng = make()
        eng.coord.state.set_intent("search_flights")
        eng.coord.state.update({"destination": "Pune"})
        eng.emit(FinalResponse("done"))  # policy gave no snapshot: engine attaches current
        eng.emit(FinalResponse("done", {"intent": "x", "slots": {"a": 1}}))  # policy snapshot kept
        eng.emit(Speak("hi"))
        return eng

    eng = run(go())
    finals = [a for a in drain(eng._out) if a["type"] == "final_response"]
    assert len(finals) == 2
    assert finals[0]["state_snapshot"] == {"intent": "search_flights", "slots": {"destination": "Pune"}}
    assert finals[1]["state_snapshot"] == {"intent": "x", "slots": {"a": 1}}
    assert all(validate_action(a) == [] for a in finals)


def test_agent_finals_always_have_snapshots_across_flows():
    manifest = {"type": "tool_manifest", "t": 0, "date": "2026-10-01", "tools": [
        {"name": "search_flights", "read_only": True, "parameters": {"type": "object", "properties": {
            "origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}},
            "required": ["origin", "destination", "date"]}}]}
    events = [manifest,
              {"type": "text", "t": 0, "text": "find flights from Delhi to Goa tomorrow", "end_of_turn": True},
              {"type": "tool_result", "t": 100, "call_id": "call-0001", "result": [{"flight_id": "F1"}]},
              {"type": "text", "t": 200, "text": "and to Pune", "end_of_turn": True},
              {"type": "text", "t": 250, "text": "cancel that", "end_of_turn": True},
              {"type": "text", "t": 300, "text": "find flights from Delhi to Goa tomorrow", "end_of_turn": True},
              {"type": "tool_result", "t": 400, "call_id": "call-0003", "error": "boom"},
              {"type": "tool_result", "t": 450, "call_id": "call-0004", "error": "boom"},
              {"type": "tool_result", "t": 500, "call_id": "call-0005", "error": "boom"}]

    async def go():
        agent = Agent(clock=VirtualClock())
        inq, outq = asyncio.Queue(), asyncio.Queue()
        for e in events:
            inq.put_nowait(e)
        inq.put_nowait(None)
        await agent.run(inq, outq)
        return [from_wire(a) for a in drain(outq)] if False else drain(outq)

    acts = run(go())
    finals = [a for a in acts if a["type"] == "final_response"]
    assert len(finals) >= 3
    for f in finals:
        assert validate_action(f) == []
        assert set(f["state_snapshot"]) == {"intent", "slots"}


def test_actions_are_stamped_by_the_virtual_clock():
    async def go():
        clock = VirtualClock()
        agent = Agent(clock=clock)
        inq, outq = asyncio.Queue(), asyncio.Queue()
        task = asyncio.ensure_future(agent.run(inq, outq))
        for t, text in ((0, "hello"), (750, "blorp")):
            clock.set(t)
            await inq.put({"type": "text", "t": t, "text": text, "end_of_turn": True})
            await inq.join()
        await inq.put(None)
        await task
        return drain(outq)

    acts = run(go())
    assert [a["t"] for a in acts] == [0.0, 750.0]


def test_handler_exception_does_not_kill_the_session():
    class Boom(Recorder):
        async def on_event(self, ev):
            raise RuntimeError("bad")

    async def go():
        h = Boom()
        eng = Engine(h, clock=VirtualClock())
        h.engine = eng
        inq, outq = asyncio.Queue(), asyncio.Queue()
        for e in ({"type": "text", "t": 0, "text": "x", "end_of_turn": True}, None):
            inq.put_nowait(e)
        await eng.run(inq, outq)
        return eng

    eng = run(go())
    assert eng.trace.of("handler_error")


def test_floor_state_machine_through_a_turn():
    async def go():
        h, eng = make()
        f = eng.coord.floor
        seen = [f.state.value]
        from theme5.protocol import parse_event
        await eng.dispatch(parse_event({"type": "text", "t": 0, "text": "find", "end_of_turn": False}))
        seen.append(f.state.value)
        await eng.dispatch(parse_event({"type": "text", "t": 1, "text": "flights", "end_of_turn": True}))
        seen.append(f.state.value)
        eng.emit(Speak("Searching."))
        seen.append(f.state.value)
        c = eng.start_call(SEARCH, {"destination": "Goa"})
        seen.append(f.state.value)
        await eng.dispatch(parse_event({"type": "interrupt", "t": 5}))
        seen.append(f.state.value)
        barge = f.barge_in
        eng.on_result(ToolResult(c.call_id, ok=True, result=1))
        seen.append(f.state.value)
        return seen, barge

    seen, barge = run(go())
    assert seen == ["idle", "user_speaking", "idle", "agent_speaking", "waiting_on_tools",
                    "user_speaking", "idle"]
    assert barge
