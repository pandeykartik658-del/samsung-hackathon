# /mnt/project-files/theme5/docs/PLAN.md
# Architecture plan, mapped to the rubric

Requirements and eval-kit unknowns (U01–U26) live in `docs/SPEC.md`. This file says how the code answers them.

## Pipeline (one `Agent` per session, no module-level state)

```
events ─► protocol.parse_event ─► Agent.handle ──fast path (sync, no awaits)──────────────► actions
           (aliases, units,        │  nlu.parse → SessionState.update (localised)            (validated,
            nesting, EOT, cum.)    │  _invalidate: cancel superseded calls FIRST               to_wire'd)
                                   │  planner.next_step → speak+tool_call | clarify | final
                                   └─ slow path (tasks): ASR / vision / optional LLM
tool_result ─► stale? drop ─► ledger/results ─► replan (chain, retry, finalize)
```

| Module | Role |
|---|---|
| `protocol.py` | Only file that knows kit spellings: event/action names, timestamp key+unit, wire field names, manifest flags, snapshot shape, all tunables. Every guess is labelled ASSUMPTION with its SPEC U-number. Delegates manifest parsing to `protocol_manifest.py` when present. |
| `nlu.py` | Rule fast path (µs): disfluency cleanup, self-repair split, bare-value repairs, cancel, intents, slots (route, dates, pax, cabin, names, flight ids, booking refs, device, error code). |
| `state.py` | Session slots + intent + version counter; localised corrections. |
| `planner.py` | Manifest-driven: picks the goal tool by intent hints or utterance match (unseen tools), binds args via synonyms/enums/types, chains read-only prerequisites to obtain ids, asks for missing slots. |
| `tools.py` | Registry, in-flight `CallTracker` (per-session ids), `IdempotencyLedger`. Extended by the tools thread. |
| `nlg.py` | Specific progress lines, grounded finals (ids, times, prices, refs, manual steps), truthful failures. |
| `plugins.py` | `LLMPlugin` (bounded, falls back to rules) and `Perception` (payload transcript/labels default; ambiguity by top-2 margin). |
| `agent.py` | Coordination layer: ordering, cancel-first invalidation, dedupe, retries, modify-or-refuse after commit, snapshots in tool-param names. |
| `harness.py`, `scorer.py` | Minimal local replay + proxy rubric for `scenarios/0*.json`. The fuller kit stand-in is `sim/` (sim thread). |

## Rubric mapping

| Category | What earns it | Where |
|---|---|---|
| Task Completion 40 | Correct goal tool from manifest; args bound to param names/enums; search→book chaining with choice (first/cheapest/earliest); clarification for missing slots; final grounded in result fields. | planner, nlu, nlg |
| Interruption Recovery 35 | On any slot/intent change, recompute each in-flight call's args; mismatch ⇒ `cancel` emitted before any speech in the same handler; results for cancelled ids dropped; derived ids cleared when their source slots change; no stale reruns. | agent `_invalidate`, CallTracker |
| Latency 15 | Every user turn gets a substantive action synchronously on the triggering event (progress line naming the concrete task, clarification, or answer). Fillers only when perception is slow (>250 ms). | agent, nlg |
| Safety & Protocol 10 | Ledger keyed on tool+canonical args; unknown tools default to state-modifying; writes retried only on retryable errors; after a commit, changes go through a modify tool or are refused, never a second booking; every action validated before it leaves. | tools, agent `_emit` |
| Quality ×0.8–1.2 | Short varied lines, no completion claims before ok results, final states only what results confirm. | nlg |

## Status (2026-09-29)

- `pytest tests/`: 365 passed, 3 skipped (all threads' tests). Engine/coordinator design: `docs/ARCHITECTURE.md`.
- `python -m theme5 run scenarios/0*.json`: 5/5 at 100 (proxy scorer).
- `python -m sim.run scenarios --agent theme5.agent:Agent`: 9/9 sim scenarios pass; bench weighted total 98.4 (s08 loses on the quality multiplier).

## Next steps

1. Wire in sibling modules: `slots.py` (see `docs/SLOTS_INTEGRATION.md`), `fastpath.py`, `MultimodalPerception` from `multimodal.py` (see `docs/MULTIMODAL.md`); fold `protocol_media.py` into `protocol.py`.
2. Speculative read-only calls on partial transcripts (SPEC §5), behind a flag in `protocol.py`.
3. When the kit ships: run its 9 public scenarios, change only `protocol.py`, record per-scenario scores in the README.
4. Submission: README, Dockerfile, requirements, tag `PRISM_GENAI_HACKATHON_Y2026` (SPEC R50–R54).
