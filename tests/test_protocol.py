# /mnt/project-files/theme5/tests/test_protocol.py
from theme5 import protocol as P


def test_event_aliases_and_nesting():
    ev = P.parse_event({"event": "Barge-In", "ts": 12, "data": {"text": "stop"}})
    assert ev.type == P.EV_INTERRUPT and ev.t == 12 and ev.payload["text"] == "stop"
    assert P.parse_event({"type": "tool_response", "t": 1}).type == P.EV_TOOL_RESULT
    assert P.parse_event({"type": "weird_thing"}).type == P.EV_UNKNOWN
    assert P.parse_event('{"type":"text","t":5,"text":"hi"}').payload["text"] == "hi"


def test_timestamps_units():
    assert P.to_ms(1.5, "s") == 1500.0
    assert P.to_ms(250) == 250.0
    assert P.to_ms(1.7e12) == 1.7e12
    assert P.to_ms("1970-01-01T00:00:01Z") == 1000.0


def test_end_of_turn_variants():
    assert P.is_end_of_turn(P.parse_event({"type": "text", "text": "hi", "end_of_turn": True}))
    assert P.is_end_of_turn(P.parse_event({"type": "text", "text": "hi <EOT>"}))
    assert P.is_end_of_turn(P.parse_event({"type": "eot"}))
    assert not P.is_end_of_turn(P.parse_event({"type": "text", "text": "hi"}))
    assert P.text_of(P.parse_event({"type": "text", "text": "hi <EOT>"})) == "hi"


def test_merge_chunk_incremental_and_cumulative():
    assert P.merge_chunk("", "book a") == "book a"
    assert P.merge_chunk("book a", "flight") == "book a flight"
    assert P.merge_chunk("book a", "book a flight") == "book a flight"
    assert P.merge_chunk("x", "y", cumulative=True) == "y"


def test_tool_result_parsing():
    ok = P.parse_tool_result(P.parse_event({"type": "tool_result", "call_id": "c1", "result": {"a": 1}}))
    assert ok.ok and ok.result == {"a": 1} and ok.call_id == "c1"
    bad = P.parse_tool_result(P.parse_event({"type": "tool_result", "call_id": "c2", "error": {"code": "timeout"}}))
    assert not bad.ok and bad.retryable
    hard = P.parse_tool_result(P.parse_event({"type": "tool_result", "call_id": "c3", "status": "failed", "error": "declined"}))
    assert not hard.ok and not hard.retryable


def test_manifest_read_write_classification():
    tools = P.parse_manifest({"tools": [
        {"name": "search", "read_only": True, "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}},
        {"name": "book", "parameters": {"type": "object", "properties": {}}},
    ]})
    by = {t.name: t for t in tools}
    assert by["search"].state_modifying is False and by["search"].required == ("q",)
    assert by["book"].state_modifying is True  # unknown => safe side


def test_builtin_manifest_parser_shapes():
    t = P.parse_tool({"name": "x", "params": [{"name": "a", "required": True}], "kind": "read"})
    assert t.required == ("a",) and not t.state_modifying


def test_idgen_is_per_session_and_deterministic():
    a, b = P.IdGen(), P.IdGen()
    assert a("call") == "call-0001" and a("call") == "call-0002" and b("call") == "call-0001"


def test_action_validation_and_wire_roundtrip():
    ids = P.IdGen()
    snap = P.snapshot_payload("book_flight", {"passengers": 2, "x": None})
    assert snap == {"intent": "book_flight", "slots": {"passengers": 2}}
    call = P.make_action(P.ACT_TOOL_CALL, 10, snap, ids=ids, call_id="call-0001", tool="book", args={"a": 1})
    assert P.validate_action(call) == []
    wire = P.to_wire(call, P.WireFormat(ts_key="ts", unit="s"))
    assert wire["name"] == "book" and wire["arguments"] == {"a": 1} and wire["ts"] == 0.01 and "t" not in wire
    back = P.from_wire(wire, P.WireFormat(ts_key="ts", unit="s"))
    assert back["tool"] == "book" and back["t"] == 10.0
    assert P.validate_action(P.make_action(P.ACT_FINAL, 0, None, ids=ids, text="hi"))  # final needs snapshot
    assert P.validate_action(P.make_action(P.ACT_SPEAK, 0, None, ids=ids, text="", kind="ack"))
    assert P.validate_action({"type": "nope"})


def test_wire_format_observes_ts_key():
    fmt = P.WireFormat()
    fmt.observe({"type": "text", "timestamp": 3})
    assert fmt.ts_key == "timestamp"
