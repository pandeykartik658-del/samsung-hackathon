# /mnt/project-files/theme5/tests/fixtures/mini_harness.py
"""Unit-test fixture only (moved from theme5/). End-to-end runs use sim/ and
bench/scorer.py, the one rubric.

Local stand-in for the (unreleased) evaluation kit: virtual-clock
discrete-event replay, deterministic mock tools with latency and fault
injection, complete event/action trace. ASSUMPTION: scenario file format is
ours; adapt when the official kit ships."""
from __future__ import annotations

import asyncio
import heapq
import itertools
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from theme5.agent import Agent
from theme5.clock import VirtualClock
from theme5.protocol import (
    ACT_CANCEL, ACT_TOOL_CALL, EV_MANIFEST, EV_TOOL_RESULT, SCENARIO_WALL_CAP_S, Event, from_wire, parse_event,
)



def _render(obj: Any, args: dict[str, Any]) -> Any:
    if isinstance(obj, str):
        m = re.fullmatch(r"\{(\w+)\}", obj)
        if m and m.group(1) in args:
            return args[m.group(1)]
        return re.sub(r"\{(\w+)\}", lambda mm: str(args.get(mm.group(1), mm.group(0))), obj)
    if isinstance(obj, dict):
        return {k: _render(v, args) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_render(v, args) for v in obj]
    return obj


@dataclass
class MockTool:
    name: str
    state_modifying: bool
    latency_ms: float = 500.0
    response: Any = None
    faults: dict[int, dict[str, Any]] = field(default_factory=dict)  # attempt no -> error
    attempts: int = 0

    def invoke(self, args: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        self.attempts += 1
        fault = self.faults.get(self.attempts)
        if fault:
            return float(fault.get("latency_ms", self.latency_ms)), {
                "ok": False, "error": fault.get("error", "fault"), "retryable": bool(fault.get("retryable", True))}
        return self.latency_ms, {"ok": True, "result": _render(self.response, args)}


@dataclass
class RunResult:
    scenario: dict[str, Any]
    events: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    commits: list[tuple[str, dict[str, Any]]]
    dropped_results: list[str]
    latency: list[tuple[str, float, float | None]]  # (event type, event t, wall ms to first spoken action)
    timed_out: bool = False


def load_scenario(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


async def run_scenario(scn: dict[str, Any], agent_factory: Callable[..., Agent] | None = None,
                       cap_s: float = SCENARIO_WALL_CAP_S) -> RunResult:
    clock = VirtualClock()
    agent = (agent_factory or Agent)(clock=clock)
    in_q: asyncio.Queue[Any] = asyncio.Queue()
    out_q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    task = asyncio.get_running_loop().create_task(agent.run(in_q, out_q))

    mocks = {t["name"]: MockTool(
        name=t["name"], state_modifying=bool(t.get("state_modifying", True)),
        latency_ms=float(t.get("mock", {}).get("latency_ms", 500)),
        response=t.get("mock", {}).get("response"),
        faults={int(f["attempt"]): f for f in t.get("mock", {}).get("faults", [])},
    ) for t in scn["tools"]}
    manifest_tools = [{k: v for k, v in t.items() if k != "mock"} for t in scn["tools"]]

    seq = itertools.count()
    heap: list[tuple[float, int, dict[str, Any]]] = []
    heapq.heappush(heap, (0.0, next(seq), {"type": EV_MANIFEST, "t": 0.0, "tools": manifest_tools,
                                           **({"date": scn["date"]} if "date" in scn else {})}))
    for e in scn["events"]:
        heapq.heappush(heap, (float(e["t"]), next(seq), dict(e)))

    events: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    commits: list[tuple[str, dict[str, Any]]] = []
    dropped: list[str] = []
    cancelled: set[str] = set()
    pending_writes: dict[str, tuple[str, dict[str, Any]]] = {}
    latency: list[tuple[str, float, float | None]] = []
    t_start = time.perf_counter()
    timed_out = False

    while heap:
        if time.perf_counter() - t_start > cap_s:
            timed_out = True
            break
        t, _, raw = heapq.heappop(heap)
        clock.now = t
        raw = {**raw, "t": t}
        ev: Event = parse_event(raw)
        if ev.type == EV_TOOL_RESULT:
            cid = str(ev.payload.get("call_id"))
            if cid in cancelled:
                dropped.append(cid)
                continue
            if cid in pending_writes and ev.payload.get("ok"):
                commits.append(pending_writes.pop(cid))
        events.append(raw)
        w0 = time.perf_counter()
        await in_q.put(raw)
        await in_q.join()
        await agent.drain()
        first: float | None = None
        while not out_q.empty():
            a = from_wire(out_q.get_nowait())
            actions.append(a)
            if first is None and a["type"] not in (ACT_TOOL_CALL, ACT_CANCEL):
                first = (time.perf_counter() - w0) * 1000.0
            if a["type"] == ACT_TOOL_CALL:
                mock = mocks.get(a["tool"])
                if mock is None:
                    res = {"ok": False, "error": "unknown_tool", "retryable": False}
                    lat = 10.0
                else:
                    lat, res = mock.invoke(a["args"])
                    if mock.state_modifying:
                        pending_writes[a["call_id"]] = (a["tool"], a["args"])
                heapq.heappush(heap, (t + lat, next(seq), {"type": EV_TOOL_RESULT, "call_id": a["call_id"], **res}))
            elif a["type"] == ACT_CANCEL:
                cancelled.add(a["call_id"])
                pending_writes.pop(a["call_id"], None)
        latency.append((ev.type, t, first))

    await in_q.put(None)
    await asyncio.wait_for(task, timeout=5)
    return RunResult(scn, events, actions, commits, dropped, latency, timed_out)


def run_file(path: str | Path) -> RunResult:
    return asyncio.run(run_scenario(load_scenario(path)))
