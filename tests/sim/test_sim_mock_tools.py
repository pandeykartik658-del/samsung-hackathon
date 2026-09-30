# tests/sim/test_sim_mock_tools.py
from sim.mock_tools import MockToolServer, args_match, canonical_args, validate_args, value_matches

FS = {"origin": "DEL", "destination": "BOM", "date": "2026-10-12"}


def test_default_manifest_has_four_tools_with_modes():
    m = MockToolServer()
    modes = {t["name"]: (t["side_effect"], t["read_only"]) for t in m.manifest}
    assert modes == {"flight_search": ("read", True), "book_flight": ("write", False),
                     "create_ticket": ("write", False), "manual_lookup": ("read", True)}


def test_enabled_filter_and_unseen_tool():
    m = MockToolServer({"enabled": ["flight_search"], "extra": [
        {"name": "upgrade_seat", "side_effect": "write", "parameters": {"type": "object", "properties": {
            "cabin": {"type": "string", "enum": ["business"]}}, "required": ["cabin"]},
         "mock": {"latency_ms": 250, "response": {"status": "upgraded"}}}]})
    assert [t["name"] for t in m.manifest] == ["flight_search", "upgrade_seat"]
    assert "mock" not in m.by_name["upgrade_seat"]
    p = m.plan("upgrade_seat", {"cabin": "business"}, 1)
    assert (p.delay_ms, p.status, p.payload) == (250, "ok", {"status": "upgraded"})
    assert p.commit is not None
    assert m.plan("upgrade_seat", {"cabin": "first"}, 2).rejected == "invalid_arguments"


def test_generated_flight_search_is_deterministic():
    a = MockToolServer().plan("flight_search", FS, 1)
    b = MockToolServer().plan("flight_search", dict(FS), 7)
    assert a.payload == b.payload and len(a.payload["flights"]) == 3
    assert a.commit is None


def test_fixtures_first_match_wins():
    m = MockToolServer({"config": {"flight_search": {"fixtures": [
        {"match": {"destination": {"any_of": ["BOM", "Mumbai"]}}, "response": {"flights": ["x"]}},
        {"match": {}, "response": {"flights": ["y"]}}]}}})
    assert m.plan("flight_search", {**FS, "destination": "mumbai"}, 1).payload == {"flights": ["x"]}
    assert m.plan("flight_search", {**FS, "destination": "MAA"}, 1).payload == {"flights": ["y"]}


def test_fault_injection_by_call_index():
    m = MockToolServer({"config": {"flight_search": {"latency_ms": 1000, "timeout_ms": 4000, "faults": [
        {"call_index": 1, "kind": "error", "error": "503"},
        {"call_index": 2, "kind": "timeout"},
        {"call_index": 3, "kind": "slow", "factor": 3},
        {"call_index": 4, "kind": "slow", "latency_ms": 9000}]}}})
    p1, p2, p3, p4, p5 = (m.plan("flight_search", FS, i) for i in range(1, 6))
    assert (p1.status, p1.payload, p1.retryable) == ("error", {"error": "503"}, True)
    assert (p2.status, p2.delay_ms) == ("timeout", 4000)
    assert (p3.status, p3.delay_ms) == ("ok", 3000)
    assert p4.delay_ms == 9000
    assert (p5.status, p5.delay_ms) == ("ok", 1000)


def test_fault_by_arg_match():
    m = MockToolServer({"config": {"book_flight": {"faults": [{"match": {"flight_id": "X-1"}, "kind": "error"}]}}})
    assert m.plan("book_flight", {"flight_id": "X-1", "passenger_name": "A"}, 1).status == "error"
    assert m.plan("book_flight", {"flight_id": "X-2", "passenger_name": "A"}, 2).status == "ok"


def test_unknown_tool_and_bad_args_are_rejected():
    m = MockToolServer()
    assert m.plan("teleport", {}, 1).rejected == "unknown_tool"
    p = m.plan("flight_search", {"origin": "DEL"}, 1)
    assert p.status == "error" and p.rejected == "invalid_arguments" and "destination" in p.payload["error"]


def test_commit_records_bookings_and_duplicates_get_new_refs():
    m = MockToolServer()
    args = {"flight_id": "AI-202", "passenger_name": "Rahul Verma"}
    p1 = m.plan("book_flight", args, 1)
    m.commit(p1.commit, 100, "c1")
    p2 = m.plan("book_flight", args, 2)
    m.commit(p2.commit, 200, "c2")
    assert len(m.state.bookings) == 2 and len(m.state.writes) == 2
    assert p1.payload["booking_ref"] != p2.payload["booking_ref"]


def test_matchers():
    assert value_matches("Priya  Sharma", "priya sharma")
    assert value_matches({"any_of": ["GOI", "GOX"]}, "gox")
    assert value_matches({"regex": "4C"}, "error 4c on washer")
    assert not value_matches({"regex": "4C"}, None)
    assert args_match({"a": {"present": False}}, {"b": 1})
    assert not args_match({"a": {"present": True}}, {"b": 1})
    assert args_match(None, {"x": 1})
    assert canonical_args({"b": " X ", "a": 1}) == canonical_args({"a": 1, "b": "x"})


def test_validate_args_types_enum_pattern():
    tdef = {"parameters": {"type": "object", "additionalProperties": False, "properties": {
        "n": {"type": "integer"}, "c": {"type": "string", "enum": ["a"]}, "r": {"type": "string", "pattern": "^[A-Z]{2}$"}},
        "required": ["n"]}}
    assert validate_args(tdef, {"n": 1, "c": "a", "r": "AB"}) == []
    errs = validate_args(tdef, {"n": True, "c": "b", "r": "abc", "z": 1})
    assert len(errs) == 4


def test_validate_args_list_and_flat_parameter_styles():
    as_list = {"parameters": [{"name": "q", "type": "string", "required": True}, {"name": "n", "type": "integer"}]}
    flat = {"parameters": {"q": {"type": "string", "required": True}, "n": "integer"}}
    for tdef in (as_list, flat):
        assert validate_args(tdef, {"q": "x", "n": 2}) == []
        assert validate_args(tdef, {"n": "2"}) == ["missing required argument 'q'", "argument 'n' must be integer"]
