# /mnt/project-files/theme5/tests/test_planner.py
from theme5.planner import (
    ResultStore, choose, fill_args, goal_tool, next_step, result_items, slot_for_param,
)
from theme5.protocol import ParamSpec, ToolSpec
from theme5.state import SessionState
from theme5.tools import ToolRegistry

SEARCH = ToolSpec("search_flights", "Search flights", (
    ParamSpec("origin", "string", True), ParamSpec("destination", "string", True), ParamSpec("date", "string", True)), False)
BOOK = ToolSpec("book_flight", "Book a flight", (
    ParamSpec("flight_id", "string", True), ParamSpec("passengers", "integer", True)), True)
WEATHER = ToolSpec("get_weather", "Weather for a city", (ParamSpec("city", "string", True),), False)


def reg(*tools):
    return ToolRegistry(tools)


def test_slot_for_param_synonyms():
    assert slot_for_param(ParamSpec("from_city")) == "origin"
    assert slot_for_param(ParamSpec("arrival_airport")) == "destination"
    assert slot_for_param(ParamSpec("city")) == "location"
    assert slot_for_param(ParamSpec("num_passengers")) == "passengers"
    assert slot_for_param(ParamSpec("flight_id")) == "flight_id"
    assert slot_for_param(ParamSpec("pnr")) == "booking_ref"
    assert slot_for_param(ParamSpec("offer_id")) is None


def test_fill_args_coerces_and_reports_missing():
    args, missing = fill_args(BOOK, {"passengers": "3"})
    assert args == {"passengers": 3} and missing == ["flight_id"]
    t = ToolSpec("t", "", (ParamSpec("cabin_class", "string", True, ("ECONOMY", "BUSINESS")),), False)
    assert fill_args(t, {"cabin": "business"})[0] == {"cabin_class": "BUSINESS"}
    w, m = fill_args(WEATHER, {"destination": "Goa"})  # location falls back to destination
    assert w == {"city": "Goa"} and not m


def test_goal_tool_by_intent_and_unseen():
    r = reg(SEARCH, BOOK, WEATHER)
    assert goal_tool("book_flight", "", r).name == "book_flight"
    assert goal_tool("search_flights", "", r).name == "search_flights"
    assert goal_tool(None, "how's the weather", r).name == "get_weather"
    assert goal_tool(None, "sing me a song", r) is None


def test_chain_search_then_book_with_choice():
    r = reg(SEARCH, BOOK)
    st = SessionState(intent="book_flight", slots={"origin": "A", "destination": "B", "date": "d", "passengers": 2})
    rs = ResultStore()
    step = next_step(st, r, rs)
    assert step.kind == "call" and step.tool.name == "search_flights" and step.purpose == "prereq"
    rs.put("search_flights", step.args, {"flights": [{"flight_id": "X", "price": 9}, {"flight_id": "Y", "price": 3}]})
    step = next_step(st, r, rs)
    assert step.kind == "call" and step.tool.name == "book_flight" and step.args == {"flight_id": "X", "passengers": 2}
    st.slots["choice"] = "cheapest"
    assert next_step(st, r, rs).args["flight_id"] == "Y"


def test_stale_prereq_result_not_reused():
    r = reg(SEARCH, BOOK)
    st = SessionState(intent="book_flight", slots={"origin": "A", "destination": "B", "date": "d", "passengers": 1})
    rs = ResultStore()
    rs.put("search_flights", {"origin": "A", "destination": "B", "date": "d"}, [{"flight_id": "X"}])
    st.slots["destination"] = "C"
    step = next_step(st, r, rs)
    assert step.tool.name == "search_flights" and step.args["destination"] == "C"


def test_clarify_missing_and_done_and_idle():
    r = reg(SEARCH)
    st = SessionState(intent="search_flights", slots={"origin": "A"})
    step = next_step(st, r, ResultStore())
    assert step.kind == "clarify" and step.slot == "destination"
    st.slots.update({"destination": "B", "date": "d"})
    rs = ResultStore()
    rs.put("search_flights", {"origin": "A", "destination": "B", "date": "d"}, [])
    assert next_step(st, r, rs).kind == "done"
    assert next_step(SessionState(), r, rs).kind == "idle"
    assert next_step(SessionState(intent="navigate"), reg(), rs).kind == "unsupported"


def test_choose_and_items():
    items = [{"price": 5, "depart": "10:00"}, {"price": 2, "depart": "08:00"}]
    assert choose(items, None) is items[0]
    assert choose(items, "cheapest") is items[1]
    assert choose(items, "earliest") is items[1]
    assert choose(items, -1) is items[1]
    assert choose(items, 7) is None
    assert result_items({"flights": items}) == items
    assert result_items({"a": 1}) == [{"a": 1}]
