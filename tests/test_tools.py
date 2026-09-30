# /mnt/project-files/theme5/tests/test_tools.py
"""tools.py + protocol_manifest.py against three manifests the code has never
seen: a rail service (OpenAI function shape), an unusual-names airline API
(MCP shape) and a smart-home / helpdesk mix (flat and list param shapes).
Async tests run on sim.vloop's virtual clock, so backoff costs no real time."""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Callable

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sim.vloop import VirtualTimeLoop  # noqa: E402
from theme5.protocol import ToolResult, ToolSpec, ParamSpec  # noqa: E402
from theme5.protocol_manifest import parse_manifest_defs, parse_tool_def  # noqa: E402
from theme5 import tools as T  # noqa: E402

# ----------------------------------------------------------------------------
# Manifest 1: rail, OpenAI function-calling shape
# ----------------------------------------------------------------------------
RAIL = {"tools": [
    {"type": "function", "read_only": True, "function": {
        "name": "search_trains", "description": "Search trains",
        "parameters": {"type": "object", "properties": {
            "from_station": {"type": "string"},
            "to_station": {"type": "string"},
            "travel_date": {"type": "string", "format": "date"},
            "num_travellers": {"type": "integer", "minimum": 1, "maximum": 6, "default": 1},
        }, "required": ["from_station", "to_station", "travel_date"]}}},
    {"type": "function", "function": {  # no marker at all -> must be state-modifying
        "name": "reserve_seat", "description": "Reserve a train seat",
        "parameters": {"type": "object", "properties": {
            "train_ref": {"type": "string"},
            "holder_name": {"type": "string", "minLength": 2},
            "seats": {"type": "integer", "minimum": 1},
        }, "required": ["train_ref", "holder_name", "seats"]}}},
]}

# ----------------------------------------------------------------------------
# Manifest 2: airline with unusual parameter names, MCP shape
# ----------------------------------------------------------------------------
ODD = [
    {"name": "qZxFareProbe", "description": "Look up fares",
     "annotations": {"readOnlyHint": True},
     "inputSchema": {"type": "object", "properties": {
         "qx_src_iata": {"type": "string", "pattern": "^[A-Za-z ]+$"},
         "dstIATA": {"type": "string"},
         "whenISO": {"type": "string", "format": "date"},
         "numSeatsReq": {"type": "integer", "minimum": 1, "maximum": 9},
         "cbn": {"type": "string", "enum": ["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS"]},
     }, "required": ["qx_src_iata", "dstIATA", "whenISO"]}},
    {"name": "commitHold", "description": "Hold a fare",
     "annotations": {"readOnlyHint": False, "idempotentHint": True},
     "inputSchema": {"type": "object", "properties": {
         "fltRef": {"type": "string"}, "p_nm": {"type": "string", "description": "Passenger full name"},
     }, "required": ["fltRef", "p_nm"]}},
    {"name": "zz_issue_tkt", "description": "Issue the ticket",
     "inputSchema": {"type": "object", "properties": {
         "fltRef": {"type": "string"}, "clientToken": {"type": "string"},
     }, "required": ["fltRef", "clientToken"]}},
]

# ----------------------------------------------------------------------------
# Manifest 3: smart home + helpdesk, flat dict and list param shapes
# ----------------------------------------------------------------------------
HOME = {"manifest": {
    "thermo_set": {"kind": "write", "description": "Set the thermostat",
                   "params": {"zone": {"type": "string", "enum": ["living", "bedroom"], "required": True},
                              "celsius": {"type": "number", "required": True, "minimum": 10, "maximum": 30}}},
    "status_probe": {"side_effects": False, "description": "Read device status",
                     "params": {"device": "str"}},
    "ticket_open": {"description": "Open a support ticket",
                    "params": [{"name": "summary", "required": True},
                               {"name": "sev", "enum": ["low", "medium", "high"], "required": True}]},
    "confused": {"read_only": True, "mutating": True, "params": {}},
}}


def registry(*manifests: Any) -> T.ToolRegistry:
    r = T.ToolRegistry()
    for m in manifests:
        r.load_manifest(m, replace=False)
    return r


# ----------------------------------------------------------------------------
# Fake harness: scripted results delivered on the virtual clock
# ----------------------------------------------------------------------------
Reply = Callable[[dict], "tuple[bool, Any]"]


class FakeHarness:
    def __init__(self, script: dict[str, list[Any]], latency: float = 0.3) -> None:
        self.script = {k: list(v) for k, v in script.items()}
        self.latency = latency
        self.calls: list[T.Call] = []
        self.cancels: list[str] = []
        self.runner: T.ToolRunner | None = None
        self.log: list[tuple[float, str, str]] = []

    def emit_call(self, c: T.Call) -> None:
        loop = asyncio.get_running_loop()
        self.calls.append(c)
        self.log.append((loop.time(), "call", c.call_id))
        queue = self.script.get(c.tool, [])
        step = queue.pop(0) if queue else ("ok", {"echo": c.args})
        if step == "hang":
            return
        lat = self.latency
        if step == "late":  # arrives long after the call timeout
            step, lat = ("ok", {"late": True}), 100.0
        kind, payload = step
        res = ToolResult(c.call_id, kind == "ok", payload if kind == "ok" else None,
                         None if kind == "ok" else str(payload))
        loop.call_later(lat, self.runner.deliver, res)  # type: ignore[union-attr]

    def emit_cancel(self, c: T.Call) -> None:
        self.cancels.append(c.call_id)


def make(reg: T.ToolRegistry, script: dict[str, list[Any]], **kw: Any) -> tuple[T.ToolRunner, FakeHarness]:
    h = FakeHarness(script)
    r = T.ToolRunner(reg, h.emit_call, h.emit_cancel, **kw)
    h.runner = r
    return r, h


def run(coro_fn: Callable[[], Any]) -> Any:
    loop = VirtualTimeLoop()
    try:
        return loop.run_until_complete(coro_fn())
    finally:
        loop.close()


SLOTS = {"origin": "Pune", "destination": "Delhi", "date": "2026-10-02", "passengers": 2, "name": "Asha Rao"}

# ============================================================================
# Parsing and classification
# ============================================================================


def test_three_manifest_shapes_parse():
    r = registry(RAIL, ODD, HOME)
    assert len(r) == 9 and not r.errors
    assert r.get("search_trains").read_only
    assert r.get("search_trains").param("num_travellers").default == 1
    assert r.get("qZxFareProbe").param("cbn").enum == ("ECONOMY", "PREMIUM_ECONOMY", "BUSINESS")
    assert r.get("ticket_open").required == ("summary", "sev")
    assert r.get("thermo_set").param("celsius").type == "number"
    assert r.get("status_probe").param("device").type == "string"


def test_not_clearly_read_only_is_state_modifying():
    r = registry(RAIL, ODD, HOME)
    assert r.get("reserve_seat").state_modifying and r.get("reserve_seat").decided_by == "default"
    assert r.get("ticket_open").state_modifying
    assert r.get("confused").state_modifying  # conflicting markers
    assert r.get("thermo_set").state_modifying
    assert r.get("status_probe").read_only  # side_effects: false is a clear marker
    assert r.get("qZxFareProbe").read_only
    assert {t.name for t in r.read_only()} == {"search_trains", "qZxFareProbe", "status_probe"}


def test_idempotency_markers():
    r = registry(ODD, RAIL)
    assert r.get("commitHold").idempotent
    assert r.get("zz_issue_tkt").idempotency_param == "clientToken" and not r.get("zz_issue_tkt").idempotent
    assert not r.get("reserve_seat").idempotent and r.get("reserve_seat").idempotency_param is None


def test_malformed_manifest_never_raises():
    tools, errs = parse_manifest_defs({"tools": [{"description": "no name"}, 42, None,
                                                 {"name": "ok_tool", "parameters": "not json"},
                                                 {"name": "x", "parameters": {"properties": {"a": None}}}]})
    assert [t.name for t in tools] == ["ok_tool", "x"] and len(errs) == 3
    assert parse_manifest_defs("garbage")[0] == []
    assert parse_manifest_defs(None)[0] == []
    r = T.ToolRegistry()
    assert r.load_manifest({"tools": []}) == [] and len(r) == 0


def test_manifest_update_replaces_registry_and_toolspec_compat():
    r = registry(RAIL)
    r.load_manifest(ODD)
    assert "search_trains" not in r and "commitHold" in r
    r.load([ToolSpec("legacy", "", (ParamSpec("q", required=True),), False)])
    assert r.get("legacy").read_only and r.get("LEGACY") is r.get("legacy")
    assert parse_tool_def(ODD[0]).spec.state_modifying is False


# ============================================================================
# Binding, clarification, validation
# ============================================================================


def test_tokens_expand_unusual_names():
    assert T.tokens("numSeatsReq") == ["number", "seats", "requested"]
    assert T.tokens("qx_src_iata") == ["qx", "source", "airport"]
    assert T.tokens("dstIATA") == ["destination", "airport"]
    assert T.tokens("whenISO") == ["when", "iso"]


def test_bind_unusual_names_by_name_and_type():
    tool = registry(ODD).get("qZxFareProbe")
    b = T.bind_args(tool, {**SLOTS, "cabin": "premium economy"})
    assert b.complete, (b.missing, b.invalid)
    assert b.args == {"qx_src_iata": "Pune", "dstIATA": "Delhi", "whenISO": "2026-10-02",
                      "numSeatsReq": 2, "cbn": "PREMIUM_ECONOMY"}
    assert b.sources["numSeatsReq"] == "passengers"


def test_bind_by_description_and_type_filter():
    tool = registry(ODD).get("commitHold")
    b = T.bind_args(tool, {"passengers": 2, "name": "Asha Rao"}, context={"flight_id": "F9"})
    assert b.args == {"p_nm": "Asha Rao", "fltRef": "F9"}  # passengers (int) never lands in a name


def test_bind_leftover_value_evidence():
    tool = parse_tool_def({"name": "t", "read_only": True, "parameters": {"properties": {
        "p1": {"type": "string", "format": "date"}, "p2": {"type": "string", "enum": ["x", "y"]}},
        "required": ["p1", "p2"]}})
    b = T.bind_args(tool, {"date": "2026-10-02", "mode": "Y"})
    assert b.args == {"p1": "2026-10-02", "p2": "y"}


def test_missing_required_yields_one_question():
    tool = registry(RAIL).get("search_trains")
    b = T.bind_args(tool, {"origin": "Pune"})
    assert b.missing == ["to_station", "travel_date"] and b.args["from_station"] == "Pune"
    assert "num_travellers" not in b.missing  # optional with default
    assert T.clarification(tool, b.missing) == "Could you tell me the to station and travel date?"


def test_invalid_enum_and_range_ask_instead_of_guess():
    reg = registry(HOME, ODD)
    b = T.bind_args(reg.get("thermo_set"), {"zone": "garage", "celsius": 45})
    assert set(b.invalid) == {"zone", "celsius"} and not b.args
    b = T.bind_args(reg.get("ticket_open"), {"issue": "TV shows E4", "priority": "urgent"})
    assert b.args == {"summary": "TV shows E4"} and "sev" in b.invalid
    q = T.clarification(reg.get("ticket_open"), b.missing, b.invalid)
    assert q == "That option isn't available. Which severity would you like: low, medium or high?"
    b = T.bind_args(reg.get("qZxFareProbe"), {**SLOTS, "passengers": 12})
    assert "numSeatsReq" in b.invalid


def test_validate_args_catches_everything():
    reg = registry(RAIL, HOME)
    st = reg.get("search_trains")
    assert T.validate_args(st, {"from_station": "A", "to_station": "B", "travel_date": "2026-10-02"}) == []
    errs = T.validate_args(st, {"from_station": "A", "travel_date": "friday", "num_travellers": "2", "x": 1})
    assert any("to_station: required" in e for e in errs)
    assert any("travel_date: not a valid date" in e for e in errs)
    assert any("num_travellers: wrong type" in e for e in errs)
    assert any("x: unknown parameter" in e for e in errs)
    assert T.validate_args(reg.get("thermo_set"), {"zone": "living", "celsius": 31}) == ["celsius: above maximum 30"]
    assert T.validate_args(reg.get("reserve_seat"), {"train_ref": "T", "holder_name": "A", "seats": 1}) == \
        ["holder_name: shorter than 2"]
    assert T.validate_args(st, "nope") == ["args must be an object"]  # type: ignore[arg-type]


# ============================================================================
# Retry policy (pure)
# ============================================================================


def test_error_classification():
    def r(e: str, retryable: bool = False) -> ToolResult:
        return ToolResult("c", False, error=e, retryable=retryable)
    assert T.classify_error(r("503 Service Unavailable")) == "transient"
    assert T.classify_error(r("weird", retryable=True)) == "transient"
    assert T.classify_error(r("invalid date")) == "permanent"
    assert T.classify_error(r("backend busy, not executed")) == "not_executed"
    assert T.classify_error(r("kaboom")) == "unknown"
    assert T.classify_error(None, timed_out=True) == "transient"


def test_decide_retry_matrix():
    reg = registry(RAIL, ODD)
    p = T.RetryPolicy()
    ro, w, idem, keyed = (reg.get(n) for n in ("search_trains", "reserve_seat", "commitHold", "zz_issue_tkt"))
    assert T.decide_retry(ro, {}, "unknown", 1, p).retry
    assert not T.decide_retry(ro, {}, "permanent", 1, p).retry
    assert not T.decide_retry(ro, {}, "transient", 3, p).retry
    d = T.decide_retry(w, {}, "transient", 1, p)
    assert not d.retry and d.side_effect_unknown
    assert T.decide_retry(w, {}, "not_executed", 1, p).retry
    assert T.decide_retry(idem, {}, "transient", 1, p).retry
    assert T.decide_retry(keyed, {"clientToken": "k"}, "transient", 1, p).retry
    assert [p.delay(a) for a in (1, 2, 3, 5)] == [0.25, 0.5, 1.0, 2.0]


# ============================================================================
# Runner on the virtual clock
# ============================================================================


def test_read_only_retries_with_backoff_on_virtual_clock():
    reg = registry(RAIL)
    runner, h = make(reg, {"search_trains": [("error", "503 unavailable"), ("error", "timeout"),
                                             ("ok", {"trains": [{"id": "T1"}]})]})

    async def go():
        t0 = asyncio.get_running_loop().time()
        o = await runner.call("search_trains", slots=SLOTS)
        return o, asyncio.get_running_loop().time() - t0

    o, elapsed = run(go)
    assert o.ok and o.attempts == 3 and len(set(o.call_ids)) == 3
    assert o.args == {"from_station": "Pune", "to_station": "Delhi", "travel_date": "2026-10-02", "num_travellers": 2}
    assert elapsed == pytest.approx(3 * 0.3 + 0.25 + 0.5)  # latency x3 + backoff 0.25 + 0.5


def test_read_only_gives_up_honestly():
    reg = registry(RAIL)
    runner, h = make(reg, {"search_trains": [("error", "503")] * 5})
    o = run(lambda: runner.call("search_trains", slots=SLOTS))
    assert o.status == T.FAILED and o.attempts == 3 and len(h.calls) == 3
    assert o.note.startswith("I couldn't search trains after 3 tries")


def test_read_only_permanent_error_not_retried():
    reg = registry(RAIL)
    runner, h = make(reg, {"search_trains": [("error", "station not found")]})
    o = run(lambda: runner.call("search_trains", slots=SLOTS))
    assert o.status == T.FAILED and len(h.calls) == 1


def test_timeout_cancels_orphan_and_retries_read():
    reg = registry(RAIL)
    runner, h = make(reg, {"search_trains": ["hang", ("ok", {"trains": []})]},
                     policy=T.RetryPolicy(call_timeout_s=2.0))
    o = run(lambda: runner.call("search_trains", slots=SLOTS))
    assert o.ok and o.attempts == 2 and h.cancels == [o.call_ids[0]]


def test_non_idempotent_write_never_retried_and_reported_honestly():
    reg = registry(RAIL)
    runner, h = make(reg, {"reserve_seat": [("error", "504 gateway timeout")]})
    slots = {"name": "Asha Rao", "passengers": 2}
    o = run(lambda: runner.call("reserve_seat", {"train_ref": "T1"}, slots=slots))
    assert o.status == T.UNCERTAIN and len(h.calls) == 1
    assert "can't confirm whether it went through" in o.note
    # Same request again must not be re-issued automatically.
    o2 = run(lambda: runner.call("reserve_seat", {"train_ref": "T1"}, slots=slots))
    assert o2.status == T.IN_FLIGHT and len(h.calls) == 1


def test_write_timeout_is_uncertain_not_retried():
    reg = registry(RAIL)
    runner, h = make(reg, {"reserve_seat": ["late"]}, policy=T.RetryPolicy(call_timeout_s=2.0))
    o = run(lambda: runner.call("reserve_seat", {"train_ref": "T1", "holder_name": "Asha", "seats": 1}))
    assert o.status == T.UNCERTAIN and len(h.calls) == 1 and h.cancels == o.call_ids


def test_write_permanent_failure_releases_key():
    reg = registry(RAIL)
    runner, h = make(reg, {"reserve_seat": [("error", "sold out"), ("ok", {"pnr": "P1"})]})
    args = {"train_ref": "T1", "holder_name": "Asha", "seats": 1}
    o = run(lambda: runner.call("reserve_seat", args))
    assert o.status == T.FAILED and o.note.endswith("so nothing was changed.")
    assert run(lambda: runner.call("reserve_seat", args)).ok  # user may try again explicitly


def test_write_not_executed_is_retried():
    reg = registry(RAIL)
    runner, h = make(reg, {"reserve_seat": [("error", "busy: not executed"), ("ok", {"pnr": "P1"})]})
    o = run(lambda: runner.call("reserve_seat", {"train_ref": "T1", "holder_name": "Asha", "seats": 1}))
    assert o.ok and o.attempts == 2


def test_idempotent_write_retries():
    reg = registry(ODD)
    runner, h = make(reg, {"commitHold": [("error", "503"), ("ok", {"hold": "H1"})]})
    o = run(lambda: runner.call("commitHold", {"fltRef": "F1"}, slots={"name": "Asha Rao"}))
    assert o.ok and o.attempts == 2 and h.calls[0].args == h.calls[1].args


def test_dedup_key_autofilled_and_stable_across_retries():
    reg = registry(ODD)
    runner, h = make(reg, {"zz_issue_tkt": [("error", "connection reset"), ("ok", {"tkt": "K1"})]})
    o = run(lambda: runner.call("zz_issue_tkt", {"fltRef": "F1"}))
    assert o.ok and o.attempts == 2
    toks = {c.args["clientToken"] for c in h.calls}
    assert len(toks) == 1 and next(iter(toks)).startswith("idem-")


def test_duplicate_booking_trap_reuses_result():
    reg = registry(RAIL)
    runner, h = make(reg, {"reserve_seat": [("ok", {"pnr": "P1"})]})
    args = {"train_ref": "T1", "holder_name": "Asha", "seats": 1}

    async def go():
        a = await runner.call("reserve_seat", args)
        b = await runner.call("reserve_seat", {"train_ref": "t1 ", "holder_name": "asha", "seats": 1})
        return a, b

    a, b = run(go)
    assert a.ok and b.ok and b.deduped and b.result == {"pnr": "P1"} and len(h.calls) == 1


def test_concurrent_duplicate_write_blocked():
    reg = registry(RAIL)
    runner, h = make(reg, {"reserve_seat": [("ok", {"pnr": "P1"})]})
    args = {"train_ref": "T1", "holder_name": "Asha", "seats": 1}

    async def go():
        return await asyncio.gather(runner.call("reserve_seat", args), runner.call("reserve_seat", args))

    a, b = run(go)
    assert a.ok and b.status == T.IN_FLIGHT and len(h.calls) == 1


def test_needs_input_emits_nothing():
    reg = registry(ODD)
    runner, h = make(reg, {})
    o = run(lambda: runner.call("qZxFareProbe", slots={"origin": "Pune"}))
    assert o.status == T.NEEDS_INPUT and o.missing == ["dstIATA", "whenISO"] and not h.calls
    assert o.question == "Could you tell me the destination airport and date?"


def test_unknown_tool_and_invalid_explicit_args():
    reg = registry(HOME)
    runner, h = make(reg, {})
    assert run(lambda: runner.call("nope")).status == T.UNKNOWN_TOOL
    o = run(lambda: runner.call("thermo_set", {"zone": "living", "celsius": "hot"}))
    assert o.status == T.NEEDS_INPUT and "celsius" in o.invalid and not h.calls


def test_task_cancel_emits_cancel_and_late_result_is_stale():
    reg = registry(RAIL)
    runner, h = make(reg, {"reserve_seat": ["late"]})
    args = {"train_ref": "T1", "holder_name": "Asha", "seats": 1}

    async def go():
        task = asyncio.ensure_future(runner.call("reserve_seat", args))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        cid = h.calls[0].call_id
        assert h.cancels == [cid]
        assert runner.deliver(ToolResult(cid, True, {"pnr": "LATE"})) is False
        # key released: the corrected request can be issued
        h.script["reserve_seat"] = [("ok", {"pnr": "P2"})]
        return await runner.call("reserve_seat", {**args, "seats": 2})

    o = run(go)
    assert o.ok and o.result == {"pnr": "P2"} and len(h.calls) == 2
    assert runner.stale_results[0].result == {"pnr": "LATE"}


def test_runner_cancel_resolves_call_without_retry():
    reg = registry(RAIL)
    runner, h = make(reg, {"search_trains": ["hang"]})

    async def go():
        task = asyncio.ensure_future(runner.call("search_trains", slots=SLOTS))
        await asyncio.sleep(0.1)
        assert runner.cancel_all() == [h.calls[0].call_id]
        return await task

    o = run(go)
    assert o.status == T.FAILED and o.error == "cancelled" and len(h.calls) == 1


# ============================================================================
# Chained calls
# ============================================================================

TRAINS = {"trains": [{"id": "T1", "fare": 900, "departs": "08:00"},
                     {"id": "T2", "fare": 650, "departs": "11:30"},
                     {"id": "T3", "fare": 700, "departs": "06:15"}]}


def test_chain_feeds_output_into_next_tool():
    reg = registry(RAIL)
    runner, h = make(reg, {"search_trains": [("ok", TRAINS)], "reserve_seat": [("ok", {"pnr": "P7"})]})
    steps = [T.Step("search_trains", name="search"), T.Step("reserve_seat")]
    co = run(lambda: runner.chain(steps, {**SLOTS, "choice": "cheapest"}, intent="book_train"))
    assert co.ok and co.results["reserve_seat"] == {"pnr": "P7"}
    assert h.calls[1].args == {"train_ref": "T2", "holder_name": "Asha Rao", "seats": 2}
    assert [c.purpose for c in h.calls] == ["prereq", "goal"]


def test_chain_ordinal_choice_and_explicit_ref():
    reg = registry(RAIL)
    runner, h = make(reg, {"search_trains": [("ok", TRAINS), ("ok", TRAINS)],
                           "reserve_seat": [("ok", {"pnr": "A"}), ("ok", {"pnr": "B"})]})
    run(lambda: runner.chain([T.Step("search_trains"), T.Step("reserve_seat")], {**SLOTS, "choice": 3}))
    assert h.calls[1].args["train_ref"] == "T3"
    run(lambda: runner.chain([T.Step("search_trains", name="s"),
                              T.Step("reserve_seat", {"train_ref": "$steps.s.trains.0.id", "seats": 1})],
                             {**SLOTS, "passengers": 3}))
    assert h.calls[3].args == {"train_ref": "T1", "holder_name": "Asha Rao", "seats": 1}


def test_chain_unusual_names_three_steps():
    reg = registry(ODD)
    runner, h = make(reg, {"qZxFareProbe": [("ok", {"flights": [{"id": "F1"}, {"id": "F2"}]})],
                           "commitHold": [("ok", {"hold": {"id": "H9", "flight_id": "F2"}})],
                           "zz_issue_tkt": [("ok", {"ticket": "K1"})]})
    steps = [T.Step("qZxFareProbe"), T.Step("commitHold"), T.Step("zz_issue_tkt")]
    co = run(lambda: runner.chain(steps, {**SLOTS, "choice": 2}))
    assert co.ok, [o.status for o in co.outcomes]
    assert h.calls[1].args == {"fltRef": "F2", "p_nm": "Asha Rao"}
    assert h.calls[2].args["fltRef"] == "F2"


def test_chain_stops_and_asks_when_input_missing():
    reg = registry(RAIL)
    runner, h = make(reg, {"search_trains": [("ok", TRAINS)]})
    co = run(lambda: runner.chain([T.Step("search_trains"), T.Step("reserve_seat")],
                                  {k: v for k, v in SLOTS.items() if k != "name"}))
    assert not co.ok and co.last.status == T.NEEDS_INPUT and co.last.missing == ["holder_name"]
    assert len(h.calls) == 1


def test_extract_outputs_and_pick_item():
    out = T.extract_outputs({"status": "ok", "trains": TRAINS["trains"], "tags": ["a"]}, "earliest")
    assert out["train_id"] == "T3" and out["id"] == "T3" and out["status"] == "ok" and out["tags"] == ["a"]
    assert T.pick_item([1, 2, 3], -1) == 3 and T.pick_item([1, 2], 9) == 1 and T.pick_item([], 1) is None
    assert T.pick_item(TRAINS["trains"], "latest")["id"] == "T2"
    assert T.extract_outputs(None) == {} and T.extract_outputs("x") == {}
