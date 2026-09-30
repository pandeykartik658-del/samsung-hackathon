# docs/SPEC.md — Theme 05 "Interruptible Real-Time Agents"

Sources and citation keys:

- **G** = `Theme 5_Guide.pdf` (v1.0.0, 3 pages). Cited as `G§<section> p<page>`.
- **H** = `Samsung PRISM_Y2026_GenAI_Hackathon_3rd_Edition.V2.pdf` (15 pages). Cited as `H p<page>`.
- Anything not stated in G or H is marked **ASSUMPTION** and must be isolated behind `protocol.py`.

Rubric keys: **TC** Task Completion 40, **IR** Interruption Recovery 35, **LAT** Response Latency 15, **SP** Safety & Protocol 10, **QM** quality multiplier 0.80–1.20 (G§5 p2–3). **MM×1.5** means hidden multimodal scenarios are weighted 1.5× (G§5 p3). **JURY** is the round-level judging rubric (H p11). **DQ** is a disqualification risk (H p12–13).

---

## 0. Facts that frame everything

| Fact | Source |
|---|---|
| Agent talks over two async queues: timestamped events in, actions out. | G§3 p1 |
| Each scenario is scored 0–100 **strictly from trace logs**. What is not in the trace does not exist. | G§5 p2 |
| Public suite: 9 canonical scenarios. Hidden set: ~60. Both are 50% text, 30% audio, 20% visual. | G§4 p2 |
| Hidden multimodal scenarios score 1.5×. Audio + visual is 50% of scenarios, so roughly 60% of hidden weight (ASSUMPTION: 1.5× applies per scenario to the total). | G§4 p2, G§5 p3 |
| Mock tools: flight search, booking, ticket creation, frame-grounded manual lookups, with deterministic latency and fault injection. | G§4 p2 |
| Harness uses a virtual clock with deterministic event replay. | G§4 p2 |
| Eval kit "will be released post the registrations". Registration closed 16 Sep. | G§4 p2, H p9 |
| Two different rubrics exist: the automated per-scenario rubric (G§5) and the Round 1 jury rubric: prototype & functionality 30, technical depth 25, innovation 20, relevance 15, presentation 10. How they combine is not stated. | G§5 p2, H p11 |

**Inconsistencies in the sources**

- H p8 (Theme 05 scope) says "Full-duplex: begin retrieving before the utterance ends" and "Corpus provided". Both lines are identical to Theme 04 (H p7) and look copy-pasted. We read the first as "start speculative work on partial transcripts" and assume no corpus exists for Theme 05. **ASSUMPTION.**
- H p9 and p13 put the final submission deadline at 25 Sep 2026; today is 29 Sep 2026. Dates are marked "tentative" (H p9). Confirm the live deadline before planning.

---

## 1. Requirements

### 1a. Agent behaviour and interface

| ID | Requirement | Source | Rubric |
|---|---|---|---|
| R01 | Communicate only via two async queues: timestamped input events, output actions. | G§3 p1 | all |
| R02 | Accept transcribed text chunks with end-of-turn markers. | G§3.1 p1 | TC, LAT |
| R03 | Accept raw audio clips (WAV). | G§3.1 p1 | TC, MM×1.5 |
| R04 | Accept video frames (PNG). | G§3.1 p1 | TC, MM×1.5 |
| R05 | Accept interruption signals. | G§3.1 p1 | IR |
| R06 | Accept asynchronous tool results (arrive out of order, possibly after cancellation). | G§3.1 p1 | TC, IR |
| R07 | Accept scenario tool manifests (per scenario, so tools are dynamic). | G§3.1 p1 | TC, SP |
| R08 | Emit spoken fillers. | G§3.1 p1 | LAT, QM |
| R09 | Emit non-blocking tool calls, each with an explicit `call_id`. | G§3.1 p1 | TC, SP |
| R10 | Emit cancellations. | G§3.1 p1 | IR |
| R11 | Emit clarification requests. | G§3.1 p1 | TC, QM |
| R12 | Emit final responses carrying a structured State Snapshot (intent and slot values). | G§3.1 p1 | TC, SP |
| R13 | Fast path responds within "a few hundred milliseconds" (acknowledgment, clarification, progress narration). | G§1 p1 | LAT |
| R14 | Slow path runs async tools, multimodal processing and complex reasoning without blocking the fast path. | G§1 p1 | TC, LAT |
| R15 | Coordination layer: non-blocking execution, call cancellation, state snapshot updates, idempotency for state-modifying actions. | G§1 p1 | IR, SP |
| R16 | Floor management: meaningful responses quickly, **no false completion claims**, **no excessive fillers**. | G§3.2(1) p2 | LAT, QM |
| R17 | Cancel superseded in-flight tool calls promptly, "within few ms grace period". | G§3.2(2) p2 | IR |
| R18 | After an interruption, update the state snapshot and re-plan cleanly; no stale re-runs. | G§3.2(2) p2, G§5 p2 | IR |
| R19 | Keep session-scoped slot state across turns; apply **localized** slot corrections (change only the corrected slot). | G§3.2(3) p2 | TC, IR |
| R20 | Parse dynamic tool definitions and classify each as read-only or state-modifying. | G§3.2(4) p2 | TC, SP |
| R21 | Never issue duplicate state-changing calls. | G§3.2(4) p2, G§5 p3 | SP, TC |
| R22 | Handle tools never seen before (defined only in the manifest). | G§4 p2 | TC |
| R23 | Handle chained calls (output of one tool feeds another). | G§4 p2 | TC |
| R24 | Handle retries under injected faults. | G§4 p2 | TC |
| R25 | Process audio and frames "behind conversational acknowledgments" (speak first, process in background). | G§3.2(5) p2 | LAT, MM×1.5 |
| R26 | Clarify ambiguous perceptions instead of guessing. | G§3.2(5) p2 | TC, QM |
| R27 | Emit well-formed JSON with valid snapshots and identifiers. | G§3.2(6) p2, G§5 p3 | SP |
| R28 | Task Completion is judged on correct tool execution, valid argument extraction, snapshot accuracy, and grounding of the final response in tool results. | G§5 p2 | TC |
| R29 | Latency is measured as time to first **substantive** spoken action after user input **or** interruption. | G§5 p2 | LAT |
| R30 | Transcript must be natural, truthful and relevant (multiplier 0.80–1.20). | G§5 p3 | QM |
| R31 | Use-case behaviours: drop stale route calculation when destination changes; change booking parameters mid-booking without double-booking; ground device queries in camera frames and manuals; resolve hesitations and self-repairs. | G§2 p1 | IR, TC |
| R32 | Keep the user engaged, drive the conversation toward the end goal, handle goal switches without losing relevant session context. | H p8 | TC, QM |
| R33 | Multimodal inputs and multimodal outputs. | H p8 | TC, JURY |

### 1b. Execution constraints

| ID | Requirement | Source | Rubric |
|---|---|---|---|
| R40 | Python 3.10–3.12. | G§6 p3 | all (won't run otherwise) |
| R41 | 120 s wall-clock cap per scenario. | G§6 p3 | TC |
| R42 | 300 s setup/warm-up hook available (load models here, not on first event). | G§6 p3 | LAT |
| R43 | Session-scoped memory only; no cross-session caching. | G§6 p3, H p8 | SP |
| R44 | Out of scope: wake-word detection, voice synthesis tuning, UI. | G§6 p3, H p8 | — |
| R45 | Deterministic behaviour under replay (virtual clock). | G§4 p2 | all |

### 1c. Submission

| ID | Requirement | Source | Rubric |
|---|---|---|---|
| R50 | Public or shared GitHub repo with README, reproducible setup, Dockerfile, requirements. | H p11, p13 | JURY, DQ |
| R51 | Release tag `PRISM_GENAI_HACKATHON_Y2026` on the final commit; the tagged commit is what is judged. | H p13 | DQ |
| R52 | Everything referenced (PPT, video, docs) present in the tagged commit. | H p13 | DQ |
| R53 | Demo video, max 5 minutes (YouTube or Drive link). | H p11–12 | JURY |
| R54 | Deck (PPT/PDF) named `CollegeName_TeamName`: theme ID, title, team, problem in own words, solution + architecture diagram, stack, innovation, results, limitations. | H p12–13 | JURY 10 |
| R55 | One submission per team via Google Form; team of max 4 from one college. | H p12 | DQ |
| R56 | Final demo (top 15): live walkthrough and Q&A on design decisions, trade-offs, and worklet potential. | H p11 | JURY |

---

## 2–3. Unknowns about the hidden evaluation kit, safest assumption, and tolerance

Every row is an **ASSUMPTION** until the kit is released. All tolerance code lives in `protocol.py` (normalise inbound → internal types; serialise internal → outbound). Core logic never touches raw field names.

| ID | Unknown | Safest assumption | How to stay tolerant |
|---|---|---|---|
| U01 | **Transport**: `asyncio.Queue` objects passed in, JSONL over stdin/stdout, websocket, or a callback? (G§3 says only "two asynchronous queues".) | In-process `asyncio.Queue` pair. | Core depends on an `EventSource`/`ActionSink` interface. Ship adapters for `asyncio.Queue`, JSONL stdio, and a plain async callable. Swapping transport is one file. |
| U02 | **Entrypoint**: module path, class/function name, signature, how the 300 s setup hook is invoked. | `Agent` class with `async setup()` and `async run(inbox, outbox)`. | Thin `entrypoint.py` shim; aliases (`main`, `run_agent`, `create_agent`) are one-liners. `setup()` is idempotent and safe to skip (lazy init fallback). |
| U03 | **Event type names** (e.g. `transcript` / `text_chunk` / `asr_partial`, `interrupt` / `barge_in`, `tool_result` / `tool_response`). | Snake_case nouns close to G§3.1 wording. | Alias table maps many spellings to an internal enum, case- and separator-insensitive. Unknown types are logged to trace and ignored, never raised. |
| U04 | **Discriminator key** (`type`, `event`, `kind`, `event_type`) and payload nesting (flat vs `data`/`payload`). | `type`, flat. | Probe an ordered key list; if a `data`/`payload` dict exists, merge it up before lookup. |
| U05 | **Timestamp field and units**: `ts`/`timestamp`/`t`; ms int, s float, ISO-8601, or virtual ticks. | Float seconds or int ms on a virtual clock. | Normalise to float seconds: ISO string → parse; number ≥ 1e11 → epoch ms; otherwise infer from the first two events' spacing and a config override. Output timestamps (if required) are emitted in the **same unit and field name as input**. |
| U06 | **Virtual clock**: does the harness patch the event loop, or stamp actions itself on receipt? | Harness stamps actions when they hit the queue. | Never read `time.time()` for decisions; use a `Clock` fed by the latest event timestamp. Emit the first substantive action synchronously on the triggering event, with no `await` before it, so latency is minimal under either model. |
| U07 | **Text chunk semantics**: incremental vs cumulative, partial vs final, end-of-turn form (`is_final`, `eot`, `end_of_turn`, separate event, `<EOT>` token). | Incremental chunks with a boolean end-of-turn flag. | Detect cumulative chunks by prefix match and diff. Treat any known EOT flag, EOT event, or EOT token as end of turn. Start speculative intent parsing on partials (H p8). |
| U08 | **Interruption signal**: explicit event, or implied by new user speech while we speak or tools run? Does it carry text? | Explicit event, may or may not carry text. | Treat both explicit interruption events and new user speech during in-flight work as barge-in. If the interruption has no text, stop speaking, hold state, and wait for the next chunk before re-planning. |
| U09 | **Grace period value** ("few ms", G§3.2(2)). | ≤ 5 ms virtual time, possibly 0. | On interruption, emit cancellations for all superseded calls **first**, in the same handler, before any speech, with no `await` between. Unit test asserts cancel is the first action after the interruption. |
| U10 | **Cancellation semantics**: does a cancelled call still return a result? Is a cancel ack sent? Can a completed call be cancelled? | Late results may still arrive. | Track call state (`pending/cancelled/done`). Drop results for cancelled ids from reasoning (log only). Never cancel a `done` id. Never re-issue a cancelled call with identical args unless the user re-asks. |
| U11 | **Tool manifest format**: OpenAI function schema, JSON Schema, MCP (`inputSchema`, `annotations.readOnlyHint`), or custom; how read-only vs state-modifying is marked (`side_effects`, `mutating`, `read_only`, `idempotent`, `kind`). When it arrives (start only, or mid-scenario updates). | JSON-Schema-style params plus an explicit read-only/state-modifying flag. | Parser accepts all listed shapes. **Any tool without a clear read-only marker is treated as state-modifying** (the safe side for R21). Name heuristics (`search/get/list/lookup` vs `book/create/cancel/update`) only break ties. Manifest updates replace the registry mid-session. |
| U12 | **Tool call action fields**: `tool`/`name`, `args`/`arguments`, args as object or JSON string. Only `call_id` is confirmed (G§3.1). | `{"type":"tool_call","call_id":..,"name":..,"arguments":{..}}`. | One serialiser per action type in `protocol.py`; field names come from a config table. Contract tests pin the shape so a kit change is one edit. No extra fields beyond the schema. |
| U13 | **call_id format and ownership**: agent-generated or harness-assigned? Uniqueness scope? | Agent-generated string, unique within the session. | Deterministic generator (`call-0001`, …) for replay. If the harness echoes or assigns ids, map ours ↔ theirs. Never reuse an id, even after cancel. |
| U14 | **Tool result and fault format**: `result`/`output`, `error`/`status`, fault types (timeout, 5xx, transient vs permanent), whether retries are expected. | Result carries `call_id` plus either a result or an error with a code/message. | Classify errors transient vs permanent by code and keywords. Retry read-only calls with bounded backoff (max 2, new `call_id`). Retry state-modifying calls **only** on an explicit "not executed" error, with the same arguments and idempotency key if the schema has one; otherwise tell the user and clarify. Narrate the retry. |
| U15 | **Definition of "duplicate state-changing call"**: same tool + same args? same tool twice in a session? a retry after failure? | Same tool with equivalent args while a prior one is pending or succeeded. | Ledger keyed on `(tool, canonicalised args)` for state-modifying tools; block a second emit. When a correction changes args mid-booking: cancel the pending call first, then issue the new one. If the first already succeeded, prefer a modify/cancel tool from the manifest; if none, clarify rather than book twice. |
| U16 | **State snapshot schema**: key names (`intent`, `slots`), where it lives (every final response only, or also on separate state-update actions), slot value normalisation (dates, cities, IATA codes). | `{"intent": str, "slots": {name: value}}` on every final response; slot names = tool parameter names. | Snapshot builder in one place; value normalisers per type from the manifest schema (`date`, `enum`, `integer`). If a standalone state action exists, emit it after every interruption and correction too. |
| U17 | **Intent vocabulary**: free text, fixed label set, or the tool name? | Tool name of the primary goal (e.g. `book_flight`). | Configurable intent-label mapping; default to manifest tool name. |
| U18 | **Output action type names** (`speak`/`say`/`filler`, `tool_call`, `cancel`, `clarify`, `final_response`). Is a filler a distinct type from substantive speech? | Distinct types as listed in G§3.1. | Single enum → string table. Tag each speech action with a `kind` internally so fillers and substantive speech can be split or merged by config. |
| U19 | **Audio input**: sample rate, channels, inline base64 vs file path, whether a transcript is also provided, whether offline ASR is expected, whether network is available. | WAV file path or base64; no transcript; no network. | Accept path, base64 and raw bytes. Offline ASR loaded in the 300 s setup hook, pluggable, with a rule-based fallback that acknowledges and asks the user to repeat. If a transcript field is present, prefer it. |
| U20 | **Frame input**: path vs base64, resolution, what "grounding" means (read a label, identify a device model, LED/state, error code). | PNG path or base64; grounding feeds `manual_lookup`-style tools. | Accept path, base64 and bytes. Fast OCR/features first, optional VLM plug-in behind it. Below a confidence threshold, ask a targeted clarification ("Is that the model number on the back label?"). |
| U21 | **What counts as "substantive"** for latency (G§5): does "Okay" or "Hmm" count? | Generic fillers do not count. | First action restates understood intent or the concrete next step ("Searching flights to Delhi for Friday"), emitted from the fast path within ~200 ms. Pure fillers only when there is nothing specific to say. |
| U22 | **Quality multiplier judge**: human, LLM judge, or heuristics. | LLM judge reading the transcript. | Short, varied, non-repetitive phrasing; never claim completion before a success result; final response quotes tool results (grounding). |
| U23 | **Scenario end**: explicit end event, queue close, or timeout? Must the agent exit? | Explicit end event or inbox close. | Handle all three; flush pending actions; cancel pending calls only if the kit counts them (config); return well within 120 s. |
| U24 | **Invalid output handling**: does one malformed action zero the scenario or just cost SP points? | Could be scenario-fatal. | Validate every outgoing action against the pinned schema before emit; on failure, log and drop instead of emitting a broken payload. Catch-all around handlers so one bad event never kills the session. |
| U25 | **Runtime environment**: CPU/GPU, network, installed packages, Docker used or not. | CPU only, no network, Docker image built from our Dockerfile. | Stdlib-only core. Heavy models are optional extras loaded in setup, with rule-based fallbacks. |
| U26 | **Process model**: one process per scenario, or many sessions in one process? | One process may host several sessions. | No module-level mutable state; everything hangs off the `Agent`/session instance (also satisfies R43). |

---

## 4. Top 10 ways to lose points

1. **Duplicate state-changing call** (re-booking after a correction, retrying a booking that actually succeeded, re-emitting on a replayed event). Costs SP and TC, and is the exact failure G§2 names ("double-booking"). *Guard: ledger + default-to-state-modifying (U11, U15).*
2. **Late or missing cancellation.** Speaking or planning before cancelling, or awaiting anything first, can miss the "few ms" grace period (G§3.2(2)). Hits IR, the second-largest bucket. *Guard: cancel-first handler (U09).*
3. **Stale re-runs and stale grounding.** Re-issuing the old query after an interruption, or letting a late result from a cancelled call reach the final response (G§5 p2 "absence of stale re-runs"). *Guard: call-state tracking (U10).*
4. **False completion claims.** "Your flight is booked" before the result arrives, or after an error (G§3.2(1)). Costs TC grounding and the QM truthfulness factor, which scales the whole scenario.
5. **Slow first substantive action.** Waiting for end-of-turn, a tool result, or an LLM before speaking; loading models on first event instead of in the 300 s hook (G§5 p2, G§6 p3). Hits LAT on every turn and every interruption.
6. **Filler spam.** Too many "one moment" lines is explicitly penalised (G§3.2(1)) and drags QM naturalness down. Fillers also likely don't count as substantive (U21).
7. **Snapshot drift.** Snapshot not updated after a correction, or a correction resets unrelated slots instead of a localized edit (G§3.2(3)). Hits TC snapshot accuracy and IR "updated state snapshots".
8. **Protocol errors.** Malformed JSON, missing or reused `call_id`, invalid snapshot, wrong field names (G§3.2(6), G§5 p3). Possibly scenario-fatal (U24). *Guard: validated serialisers + contract tests.*
9. **Brittleness.** Crashing on an unknown event type, unseen tool, odd manifest shape, or injected fault; not retrying transient failures; exceeding the 120 s cap (G§4 p2, G§6 p3). A crash is a 0 for the scenario.
10. **Weak multimodal handling.** Blocking on ASR or frame processing before acknowledging, or guessing on an ambiguous image instead of clarifying (G§3.2(5)). Multimodal is half the scenarios at 1.5× weight in the hidden set (G§5 p3), so this is the most expensive category to be mediocre at.

---

## 5. What a winning submission looks like (one page)

**In the trace, for every scenario**

- Within ~200 ms of every user turn or interruption, a specific spoken action that shows understanding ("Checking Friday flights to Delhi"). Never an empty "Okay" as the only response. **ASSUMPTION:** target under the guide's "few hundred ms" (G§1).
- Tool calls fire on partial transcripts when intent and required slots are already clear (speculative execution, G§6 p3 focus areas), read-only only. State-modifying calls wait for confirmed slots.
- On interruption: `cancel` for every superseded call is the first action, then a short acknowledgment of the change, then the new plan. Late results from cancelled calls never appear in speech or snapshots.
- Zero duplicate state-changing calls across all ~60 hidden scenarios. Mid-booking changes go through cancel-then-reissue or a modify tool, never a second booking.
- Slot corrections touch one slot; the snapshot after each turn is exactly right and uses manifest parameter names and types.
- Unseen tools just work, because the agent is driven by the manifest, not by hard-coded tool names.
- Faults are retried where safe and narrated honestly ("The booking service timed out, trying once more"). The final response states only what tool results confirm.
- Audio and frames are acknowledged instantly, processed in the background, and ambiguous perceptions produce one targeted clarification question.
- Every payload validates. No crash on any input.

**Architecture that produces that trace**

- `protocol.py`: the only place that knows kit field names; alias tables, normalisers, validated serialisers (section 2–3).
- Fast path: rule-based intent/slot extraction and response templates, no LLM (project rule 6), synchronous on each event.
- Slow path: tool calls, ASR, vision, optional LLM planner, all as cancellable `asyncio` tasks.
- Coordinator: call registry (`pending/cancelled/done`), state-modifying ledger, session slot store, snapshot builder, plan versioning so stale results are recognised by version, not by timing.
- Optional LLM and VLM plug-ins with rule-based fallbacks; heavy models warmed in the 300 s setup hook.

**Proof, not claims**

- pytest for every module, plus a local replay harness that mimics G§4 (virtual clock, mock flight/booking/ticket/manual tools, fault and latency injection) and scores traces with our own implementation of the G§5 rubric. Our own versions of the 9 public scenarios, then adversarial ones: interruption during the grace window, correction after booking succeeded, cumulative vs incremental chunks, unknown event types, manifest updated mid-session.
- Once the kit is released: run the 9 public scenarios, adjust only `protocol.py`, and keep a table of scores per scenario in the README.

**Submission package (H p11–13)**

- Repo with README (one-command setup and run), Dockerfile, requirements, tag `PRISM_GENAI_HACKATHON_Y2026` on the final commit with everything referenced inside it.
- 5-minute video showing a live interruption, a mid-booking correction without double-booking, and a frame-grounded lookup, with the trace side by side.
- Deck on the `CollegeName_TeamName` template: architecture diagram (fast/slow/coordination), measured latency and per-category scores, innovation (speculative execution, manifest-driven tools, cancel-first recovery), honest limitations. This also covers the jury rubric: working prototype 30, technical depth 25, innovation 20, relevance 15, presentation 10 (H p11).
