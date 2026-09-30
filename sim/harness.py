# sim/harness.py
"""Virtual-clock streaming harness (guide section 4).

For one scenario:
  1. Start the agent with an inbox (events) and outbox (actions) queue.
  2. Replay scripted events at their exact virtual timestamps.
  3. Serve tool_call actions from MockToolServer as async tasks with
     deterministic latency / faults; deliver tool_result events on completion.
  4. Honour cancel actions (the mock call stops; no side effect is committed).
  5. Log every event, action and harness step to a Trace (JSONL).
  6. Stop when the agent is quiescent after the last scripted event (final
     emitted after the last user turn, no calls in flight, `settle_ms` of
     silence) or at duration_ms, then send session_end.

ASSUMPTION: the harness acknowledges a cancelled call with a tool_result event
of status "cancelled" (emit_cancel_ack=True). The real kit may not.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import wire
from .adapter import Codec, accepts_kw
from .mock_tools import MockToolServer
from .scenario import Scenario
from .trace import Trace
from .vloop import VirtualTimeLoop

WALL_CAP_S = 120.0


@dataclass
class HarnessConfig:
    strip_oracle: bool = False
    emit_cancel_ack: bool = True
    settle_ms: float = 500.0
    poll_ms: float = 20.0
    shutdown_ms: float = 200.0
    charge_compute: bool = False
    wall_cap_s: float = WALL_CAP_S


@dataclass
class RunResult:
    scenario_id: str
    trace: Trace
    state: Dict[str, Any]
    end_reason: str
    wall_s: float
    virtual_ms: float
    agent_error: Optional[str] = None
    records: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class _Call:
    call_id: str
    tool: str
    args: Dict[str, Any]
    task: Optional[asyncio.Task] = None
    done: bool = False


class Harness:
    def __init__(self, scenario: Scenario, agent_factory: Callable[[], Any],
                 codec: Optional[Codec] = None, config: Optional[HarnessConfig] = None):
        self.sc = scenario
        self.agent_factory = agent_factory
        self.codec = codec or Codec()
        self.cfg = config or HarnessConfig()
        self.trace = Trace()
        self.tools = MockToolServer(scenario.tools)
        # Resolve media paths before the loop starts: filesystem calls on a network mount
        # would otherwise be charged to agent latency under --charge-compute.
        self._asset_paths = {e["id"]: str((scenario.base_dir / e["path"]).resolve())
                             for e in scenario.events if "path" in e}
        self._calls: Dict[str, _Call] = {}
        self._t0 = 0.0
        self._inbox: Optional[asyncio.Queue] = None
        self._last_action_ms = -1.0
        self._last_user_ms = -1.0
        self._last_final_ms = -1.0
        self._delivered = 0
        self._stop = False

    # -- clock -----------------------------------------------------------------
    def now_ms(self) -> float:
        return (asyncio.get_running_loop().time() - self._t0) * 1000.0

    # -- event delivery ----------------------------------------------------------
    def _send(self, ev: Dict[str, Any]) -> None:
        ev = dict(ev)
        ev["t"] = round(self.now_ms(), 3)
        if self.cfg.strip_oracle:
            ev = wire.strip_oracle(ev)
        self.trace.log(ev["t"], "in", ev["type"], ev)
        if wire.is_user_turn_boundary(ev):
            self._last_user_ms = ev["t"]
        self._inbox.put_nowait(self.codec.event_out(ev))

    def _deliver_scripted(self, raw: Dict[str, Any]) -> None:
        ev = {k: v for k, v in raw.items() if k not in ("id", "t_ms")}
        ev["event_id"] = raw["id"]
        if "path" in ev:
            ev["path"] = self._asset_paths[raw["id"]]
        self._send(ev)
        self._delivered += 1

    def _deliver_group(self, evs: List[Dict[str, Any]]) -> None:
        for raw in evs:
            self._deliver_scripted(raw)

    def _manifest_event(self) -> Dict[str, Any]:
        ev = {"type": "tool_manifest", "event_id": "manifest", "tools": self.tools.manifest,
              "session": self.sc.session}
        if self.sc.session.get("reference_date"):
            ev["date"] = self.sc.session["reference_date"]  # protocol.py reads `date`
        return ev

    # -- action handling -----------------------------------------------------------
    async def _consume(self, outbox: asyncio.Queue) -> None:
        while True:
            obj = await outbox.get()
            t = self.now_ms()
            try:
                action = wire.normalize_action(self.codec.action_in(obj))
            except Exception as exc:  # codec failure is a protocol error
                self.trace.log(t, "out", "invalid", {"raw": repr(obj), "errors": [f"encode failed: {exc}"]})
                continue
            errs = wire.validate_action(action)
            if errs:
                self.trace.log(t, "out", "invalid", {"raw": action if isinstance(action, dict) else repr(action),
                                                     "errors": errs})
                continue
            self.trace.log(t, "out", action["type"], action)
            self._last_action_ms = t
            if action["type"] == "final":
                self._last_final_ms = t
            elif action["type"] == "tool_call":
                self._start_call(action)
            elif action["type"] == "cancel":
                self._cancel_call(action["call_id"])

    def _start_call(self, a: Dict[str, Any]) -> None:
        cid, tool, args = a["call_id"], a["tool"], a.get("args", {}) or {}
        if cid in self._calls:
            self.trace.log(self.now_ms(), "sys", "duplicate_call_id", {"call_id": cid})
            self._send({"type": "tool_result", "call_id": cid, "tool": tool, "ok": False, "status": "error",
                        "error": "duplicate_call_id", "retryable": False})
            return
        idx = self.tools.next_call_index(tool)
        plan = self.tools.plan(tool, args, idx)
        call = _Call(cid, tool, args)
        self._calls[cid] = call
        self.trace.log(self.now_ms(), "sys", "tool_started", {
            "call_id": cid, "tool": tool, "args": args, "call_index": idx,
            "side_effect": "write" if self.tools.is_write(tool) else "read",
            "planned_status": plan.status, "planned_delay_ms": plan.delay_ms,
            "rejected": plan.rejected})
        call.task = asyncio.ensure_future(self._run_call(call, plan))

    async def _run_call(self, call: _Call, plan) -> None:
        await asyncio.sleep(plan.delay_ms / 1000.0)
        call.done = True
        t = self.now_ms()
        if plan.status == "ok" and plan.commit is not None:
            self.tools.commit(plan.commit, t, call.call_id)
            self.trace.log(t, "sys", "write_committed", {"call_id": call.call_id, "tool": call.tool,
                                                          "args": call.args, "result": plan.payload})
        self.trace.log(t, "sys", "tool_completed", {"call_id": call.call_id, "tool": call.tool,
                                                     "status": plan.status})
        ev = {"type": "tool_result", "call_id": call.call_id, "tool": call.tool,
              "ok": plan.status == "ok", "status": plan.status}
        if plan.status == "ok":
            ev["result"] = plan.payload
        else:
            ev["error"] = plan.payload.get("error", plan.status)
            ev["retryable"] = plan.retryable
        self._send(ev)

    def _cancel_call(self, cid: str) -> None:
        call = self._calls.get(cid)
        t = self.now_ms()
        if call is None or call.done:
            self.trace.log(t, "sys", "cancel_ignored", {"call_id": cid,
                                                         "reason": "unknown" if call is None else "already_completed"})
            return
        call.done = True
        call.task.cancel()
        self.trace.log(t, "sys", "tool_cancelled", {"call_id": cid, "tool": call.tool})
        if self.cfg.emit_cancel_ack:
            self._send({"type": "tool_result", "call_id": cid, "tool": call.tool, "ok": False,
                        "status": "cancelled", "error": "cancelled_by_agent", "retryable": False})

    # -- lifecycle -------------------------------------------------------------------
    def _in_flight(self) -> int:
        return sum(1 for c in self._calls.values() if not c.done)

    def _quiescent(self, now: float) -> bool:
        return (self._delivered == len(self.sc.events)
                and self._in_flight() == 0
                and self._last_final_ms >= self._last_user_ms
                and self._last_final_ms >= 0
                and now - max(self._last_action_ms, self._last_user_ms) >= self.cfg.settle_ms)

    async def _main(self) -> str:
        loop = asyncio.get_running_loop()
        self._t0 = loop.time()
        self._inbox = asyncio.Queue()
        outbox: asyncio.Queue = asyncio.Queue()
        self.trace.log(0.0, "sys", "scenario_start", {"scenario_id": self.sc.id, "modality": self.sc.modality,
                                                      "duration_ms": self.sc.duration_ms})
        f = self.agent_factory
        agent = f(clock=self.now_ms) if accepts_kw(f, "clock") else f()
        agent_task = asyncio.ensure_future(agent.run(self._inbox, outbox))
        consumer = asyncio.ensure_future(self._consume(outbox))

        has_manifest = any(e["type"] == "tool_manifest" for e in self.sc.events)
        if not has_manifest:
            self._send(self._manifest_event())
        # One timer per distinct timestamp, delivering its events in script order
        # (asyncio does not order timers that share a deadline).
        groups: Dict[float, List[Dict[str, Any]]] = {}
        for raw in self.sc.events:
            if raw["type"] == "tool_manifest" and "tools" not in raw:
                raw = {**self._manifest_event(), **raw}
            groups.setdefault(float(raw["t_ms"]), []).append(raw)
        for t_ms, evs in groups.items():
            loop.call_at(self._t0 + t_ms / 1000.0, self._deliver_group, evs)

        reason = "deadline"
        try:
            while self.now_ms() < self.sc.duration_ms:
                if self._stop:
                    reason = "wall_cap"
                    break
                if agent_task.done():
                    exc = None if agent_task.cancelled() else agent_task.exception()
                    if exc is not None:
                        self.trace.log(self.now_ms(), "sys", "agent_error", {"error": repr(exc)})
                        self.agent_error = repr(exc)
                        reason = "agent_error"
                        break
                if self._quiescent(self.now_ms()):
                    reason = "quiescent"
                    break
                await asyncio.sleep(self.cfg.poll_ms / 1000.0)
            if not agent_task.done():
                self._send({"type": "session_end", "event_id": "session_end"})
                await asyncio.sleep(self.cfg.shutdown_ms / 1000.0)
        finally:
            for c in self._calls.values():
                if c.task and not c.task.done():
                    c.task.cancel()
            for t in (agent_task, consumer):
                t.cancel()
            await asyncio.gather(agent_task, consumer, *[c.task for c in self._calls.values() if c.task],
                                 return_exceptions=True)
            # Drain actions the consumer had not processed yet is unnecessary: it is cancelled
            # only after shutdown_ms of virtual time, so everything emitted was consumed.
        self.trace.log(self.now_ms(), "sys", "scenario_end", {"reason": reason, "state": self.tools.state.to_dict()})
        return reason

    def _on_wall_cap(self) -> None:
        self._stop = True
        try:
            self.trace.log(self.now_ms(), "sys", "wall_cap_exceeded", {"cap_s": self.cfg.wall_cap_s})
        except RuntimeError:
            pass

    def run(self) -> RunResult:
        self.agent_error = None
        loop = VirtualTimeLoop(charge_compute=self.cfg.charge_compute, wall_cap_s=self.cfg.wall_cap_s,
                               on_wall_cap=self._on_wall_cap)
        try:
            reason = loop.run_until_complete(self._main())
            vms = loop.time() * 1000.0
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()
        return RunResult(self.sc.id, self.trace, self.tools.state.to_dict(), reason,
                         loop.wall_elapsed(), vms, self.agent_error, self.trace.records)


def run_scenario(scenario: Scenario, agent_factory: Callable[[], Any], codec: Optional[Codec] = None,
                 config: Optional[HarnessConfig] = None, trace_path: Optional[Path] = None) -> RunResult:
    res = Harness(scenario, agent_factory, codec, config).run()
    if trace_path is not None:
        res.trace.write_jsonl(trace_path)
    return res
