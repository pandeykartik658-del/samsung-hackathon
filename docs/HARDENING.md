# Profiling and hardening (2026-09-29)

Reproduce: `python -m bench.profile_suite bench/scenarios60 --repeat 3 --out bench/runs/profile_after`
(timing pass with compute charged to the virtual clock and every loop callback timed; a separate
memory pass with tracemalloc). Raw output: `bench/runs/profile_before/`, `bench/runs/profile_after/`.
Tests: `tests/test_hardening.py` (64 tests).

## Before / after (60 scenarios x 3 runs, this container, local disk)

| metric | before | after |
|---|---|---|
| first spoken action p50 | 0.67 ms | 0.69 ms |
| first spoken action p95 | 1.28 ms | 1.27 ms |
| first spoken action max | 6.80 ms (cold first turn) | 3.95 ms |
| cold first text turn in a fresh process (6 trials) | 6.9 to 8.3 ms | 1.0 to 1.3 ms |
| agent callbacks over 5 ms | 1 (cold start) | 1 (9 ms, first scenario, see below) |
| memory growth over 180 sessions | ~21 KB, plateaus | ~25 KB, plateaus |
| malformed events that kill the agent | any non-object (1st one tried: `1`) | none in 60,000 fuzzed events |
| session with a tool that never returns | no final_response, ever | safe final at 110 s with valid snapshot |
| session_end with a hung slow task | waits forever | bounded by the watchdog point |
| bench 60 / sim 9 | 100.0 / 9 pass | 100.0 / 9 pass |

On the shared network mount the cold first turn was 390 to 440 ms (lazy regex compilation plus slow
file stats); warm-up removes the agent's share of it.

## Findings

1. Slowest path to first spoken action: the first user turn of a process (lazy `re` compilation and
   first-call paths in nlu/planner). Warm turns take 0.7 ms. Fix: `theme5/warmup.py` runs one
   throwaway mini-session (manifest, partial, correction, interrupt, results, retry, cancel, audio,
   frame, unknown event) once per process from `Agent.setup()` and from Agent construction.
2. Blocking calls: none in the agent after warm-up. Every loop callback over 5 ms was either the cold
   start or the sim harness resolving asset paths on the network mount (`Harness._deliver_group`,
   up to 580 ms there). That harness cost is charged to agent latency under `--charge-compute`;
   the real kit will not have it.
3. Memory: no leak. Growth is Python's own `re` cache (bounded at 512 patterns); peak 153 KB.
4. Ways past 120 s, all fixed:
   - a tool that never returns: the agent waited forever with no final (`TOOL_TIMEOUT_MS` is None);
   - `Engine.run` drained slow-path tasks at session end with no bound;
   - any non-object event (number, list, bad JSON, bytes) raised out of `Engine.run`, so the agent
     died and never answered;
   - NaN/inf timestamps flowed into action times and produced non-JSON payloads.
   Still open (policy, not fixed here): a turn answered only with an ack, e.g. "Great, book it." after
   the booking is done (g15, g38), gets no final_response, so the harness runs to its deadline. The
   watchdog closes it at 110 s if the kit waits for a final.

## Watchdog (`theme5/watchdog.py`)

Started by `Engine.run()`. At `protocol.WATCHDOG_AT_S` (110 s; ASSUMPTION U06: the cap counts from our
run() entry) it checks whether the session is hanging: a user turn with no final_response since, a
call in flight, or slow work pending. If so it cancels every in-flight call and slow task and emits
one final_response with the current snapshot, saying honestly what did not finish (a pending write
"may not have gone through, please check"). An answered, idle session is left alone. Clock: the
running loop (virtual in the sim, monotonic in the kit). `engine.watchdog.remaining_s()` is the budget
hook multimodal.py's `remaining_s` expects.

## Protocol fuzzing

`parse_event` never raises: non-objects become `unknown` events. `to_ms` is total and finite.
`Engine.run` skips anything that still fails to parse. `validate_action` rejects NaN/inf. A final
whose snapshot cannot be built or serialised is retried with a JSON-clean snapshot, and the snapshot
falls back to the coordinator's state, then to an empty valid one. The fuzz tests cover malformed
events, missing fields, unknown types, out-of-order and non-finite timestamps and a 100 KB utterance,
and check the agent stays responsive and every emitted action is valid strict JSON.
