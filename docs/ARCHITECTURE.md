# /mnt/project-files/theme5/docs/ARCHITECTURE.md
# Architecture

Built from `docs/SPEC.md` (R01–R56, U01–U26). The rubric mapping and status are in `docs/PLAN.md`.

## Module layout (`theme5/`)

| Module | Role | Owner |
|---|---|---|
| `protocol.py` | Adapter. The only place that knows kit spellings: event/action names, timestamp key + unit, wire field names, manifest shapes and read-only markers (merged from `protocol_manifest.py`, now a re-export shim), snapshot shape, tunables. Every guess is labelled ASSUMPTION + U-number. Companion: `protocol_media.py`. | core |
| `clock.py` | `VirtualClock` (harness/tests), `EventClock` (kit: latest event t + local elapsed), `Stopwatch`. Decisions never read wall time. | core |
| `events.py` | Typed inputs and outputs for guide §3.1, `typed()` and `action_fields()`. | core |
| `engine.py` | Event loop: inbox → dispatch → handler; the single `emit()` (stamp, snapshot, validate, `to_wire`); `start_call` / `cancel_call`; per-call watcher tasks. | core |
| `coordinator.py` | Call registry, `CancelToken`, generation counter, idempotency ledger, floor state machine. Synchronous and I/O-free. | core |
| `agent.py` | Policy: the handler the engine calls. Parse, update slots, re-plan, choose what to say. Kit entrypoint `Agent`. | core |
| `fastpath.py` | Rule-based fillers, acknowledgements and turn classifier. Not yet called by `agent.py`. | fast-path thread |
| `slowpath.py` | Background ASR/vision/LLM work: inline if it finishes within 250 ms, otherwise acknowledged and tracked. | core |
| `slots.py` | Session slot tracker with parked slots on goal switches. Integration hook is in `docs/SLOTS_INTEGRATION.md`; not yet wired. | slots thread |
| `tools.py` | Manifest registry, argument binding and validation, retry policy. | tools thread |
| `multimodal.py` | WAV/PNG perception backends and frame grounding. Drop-in `Perception`; not yet wired. | multimodal thread |
| `trace.py` | `TraceLog`: actions plus `_`-prefixed notes (dropped results, blocked calls, call metadata). | core |
| `cli.py` | `python -m theme5 run scenarios/*.json` and `python -m theme5 serve` (JSONL). | core |
| `nlu.py`, `planner.py`, `nlg.py`, `state.py`, `plugins.py` | Fast-path language rules, manifest-driven planner, response templates, session state, and LLM/perception plug-in interfaces. | core |
| (tests only) `tests/fixtures/mini_harness.py`, `mini_scorer.py` | The old local replay and proxy rubric, kept as unit-test fixtures. End-to-end runs use `sim/` scored by `bench/scorer.py`; `python -m theme5 run` wraps `sim.run`. | core |

The core modules contain working logic, not stubs, because the agent was already implemented when the skeleton request arrived. `tests/test_imports.py` imports every module.

## Concurrency model

- **One event loop and two queues.** `Engine.run(inbox, outbox)` consumes events strictly in order. The handler runs synchronously for text, interrupts and results. Nothing awaits between an event and the cancellations it causes.
- **One task per in-flight tool call.** `start_call` registers a `Call` in the coordinator, emits `tool_call`, and starts `_watch(call)`. The watcher waits on the call's `CancelToken`. If `TOOL_TIMEOUT_MS` is set it can also synthesise a retryable timeout; that setting is off by default because the kit times calls itself. The kit executes the tools; the watcher only owns the call's lifecycle.
- **Cancellation tokens.** `Coordinator.cancel` sets the call status to `cancelled` (terminal) and trips its token. The engine then emits `cancel` on the same clock tick and stops the watcher. A result that arrives later is classified `late_after_cancel`, logged, and dropped.
- **Generation counters.** Each re-plan calls `next_generation()`. In-flight calls whose arguments the new plan still wants are promoted with `promote()`; all others are cancelled. `resolve()` passes a result to the policy only if the call is in flight and belongs to the current generation. Everything else (`stale_generation`, `late_after_cancel`, `duplicate_result`, `unknown_call`) is logged in the trace and never acted on.
- **Slow path.** Perception and LLM work run through `SlowPath.inline_or_background`. Results within 250 ms are handled inline, which keeps utterances in order. Slower work gets a filler line and continues as a tracked task. `drain()` waits for it, and `cancel_all()` stops it at session end.

## Events and actions (guide §3.1)

| Inputs (`events.py`) | Outputs (`events.py`) |
|---|---|
| `ToolManifest(t, tools, reference_date)` | `Speak(text, kind)` with kind = filler, ack, progress or info |
| `TextChunk(t, text, end_of_turn, cumulative)`, `EndOfTurn(t)` | `ToolCall(call_id, tool, args, generation, idempotency_key)` |
| `AudioClip(t, ref, transcript, confidence, end_of_turn)` | `Cancel(call_id, reason)` |
| `VideoFrame(t, ref, caption, labels)` | `Clarify(text, slot)` |
| `Interruption(t, text)` | `FinalResponse(text, snapshot)` |
| `ToolResultEvent(t, ToolResult)`, `SessionEnd(t)`, `UnknownEvent(t, raw_type)` | `StateSnapshot(intent, slots)` |

Every emitted action carries `action_id`, `t` (session clock), and `state_snapshot`. `emit()` guarantees that a final response has a valid snapshot. Wire names come from `protocol.WIRE_FIELDS`; for example, `tool` and `args` are sent as `name` and `arguments` (U12). `generation` and `idempotency_key` stay in the trace unless `EMIT_CALL_META` is set.

## Floor state machine (`coordinator.Floor`)

```
            user_partial / interrupt                agent_speak
   ┌──────────────────────────────┐        ┌───────────────────────┐
 IDLE ──────────────────────► USER_SPEAKING ──user_eot──► IDLE | WAITING_ON_TOOLS
   ▲                              ▲  ▲                         │
   │ final / call_done(0 left)    │  └──── interrupt ──────────┤  (barge-in: floor.barge_in)
   │                              │                            ▼
   └──────── WAITING_ON_TOOLS ◄── call_started ─── AGENT_SPEAKING
```

The engine sends signals on events (text/audio partial vs end-of-turn, interrupt) and on emits (speak/clarify, tool_call, final, result). `user_eot` and `call_done` resolve to `WAITING_ON_TOOLS` when calls are in flight, otherwise `IDLE`. Speech has no end signal in the guide, so `AGENT_SPEAKING` lasts until the next signal (ASSUMPTION U08).

## Idempotency keys (state-modifying tools)

`key = sha256("<tool>|<canonical args>")[:16]`, where canonical args is JSON with sorted keys and trimmed, lower-cased strings.

1. **Issue:** blocked with `duplicate_write_committed` if the key has committed, or `duplicate_write_pending` if it is reserved. Otherwise the key is reserved and the call goes out. Blocked attempts are logged as `_blocked_call`.
2. **Ok result:** the key is committed and the result stored.
3. **Definite failure:** the key is released, so a retry is allowed. Writes are retried only on a retryable error (U14).
4. **Cancel before any result:** the key is released (ASSUMPTION U10/U15: not executed).
5. **Ok result on a stale call:** the key is committed anyway, so the same arguments are never sent again.
6. **Changes after a commit:** these go through a modify tool from the manifest if there is one. Otherwise the agent says the action is already done and does not issue a second write.

## An interruption end to end

1. The kit sends `interrupt` at t=1900 ms. The engine moves the floor to `USER_SPEAKING` (barge-in) and clears the partial-turn buffer.
2. `text_chunk "Actually, make that Hyderabad instead." end_of_turn=true` arrives on the same tick.
3. `nlu.parse` detects the repair ("actually") and returns `{destination: "Hyderabad"}` with `correction=True`.
4. `SessionState.update` changes only `destination`, bumps `version` and logs the correction. Slots derived from the old destination are cleared.
5. `_replan` calls `next_generation()` (say generation 5). `_invalidate` recomputes each in-flight call's arguments against the new slots.
6. The Delhi `flight_search` no longer matches, so `engine.cancel_call` marks it cancelled and emits `cancel` stamped t=1900. This happens before any speech, within the grace period.
7. `planner.next_step` produces `flight_search(BLR→Hyderabad)`. The engine emits the progress line "Okay, updating: searching flights…" and then `tool_call` (call-0002, generation 5).
8. At t=3000 the Delhi result arrives. `resolve()` returns `late_after_cancel`; the result is logged and never reaches the planner or the speech.
9. At t=4900 the Hyderabad result arrives. It is live and current, so `on_tool_result` stores it and re-plans, and the planner reports the goal as done.
10. The engine emits `final_response` with the flight list and `state_snapshot {intent: flight_search, slots: {origin, destination: Hyderabad, date}}`.

## Interruption handling and speculation

- **Diff, don't flush.** When an interrupt or a new chunk changes the intent or slots, `agent._still_wanted` rebuilds each in-flight call's arguments from the new slots. `engine.reconcile` bumps the generation, promotes calls whose arguments are unchanged, and cancels only the ones that are now invalid (logged as `_reconcile`). The planner then issues whatever is missing, and every emitted action carries the updated snapshot.
- **Speculative reads.** On a partial turn (`SPECULATE_READ_ONLY`, ASSUMPTION), `_speculate` runs the planner on a scratch copy of the state. It silently starts a read-only call once the required slots are filled and `slowpath.speculation_ok` accepts the text, meaning it doesn't end in a hesitation, a repair marker, a connective or a comma. At end of turn, an in-flight speculative call whose arguments still match is adopted and narrated; one that no longer matches is cancelled (`speculation_revised`) and reissued. Speculation never touches state-modifying tools.
- **Adversarial timing** (`tests/test_interruptions.py`, traces in `runs/adversarial/`):
  - A correction after the result has already returned gets no cancel; the agent issues a fresh call and a second final.
  - When two interrupts land 40 ms apart, each cancel is stamped on its own event's tick.
  - An interrupt during a retry cancels the retry attempt; the original attempt had already failed.
  - "Cancel" after a call has failed emits no cancel and no `cancel_ignored`, just a final with an empty snapshot.

## Verification

- `python3 -m pytest -q tests/`: 383 passed, 3 skipped. This includes `tests/test_interruptions.py` (the adversarial timing cases, run through `sim/`) and `tests/test_engine.py`, which covers stale results dropped, late results after cancel, duplicate bookings blocked, a snapshot on every final, virtual-clock stamps, and the floor state machine.
- `python3 -m sim.run scenarios --agent theme5.agent:Agent`: 9/9 scenarios pass.
- `python3 -m bench.run_suite bench/scenarios60 --agent theme5.agent:Agent`: 60/60, suite 100.0 (component means TC 40, IR 34.2, LAT 15, SP 10).
