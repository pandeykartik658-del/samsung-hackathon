# /mnt/project-files/theme5/theme5/agent.py
"""Agent policy on top of the engine.

Fast path (inline, no awaits on I/O): parse -> update state -> new plan
generation -> cancel superseded calls (promote the rest) -> plan ->
speak + call | clarify | final. Slow path: perception and optional LLM via
engine.slow. Mechanics (ids, clock, cancellation, idempotency, snapshots on
the wire) live in engine.py / coordinator.py.
"""
from __future__ import annotations

import asyncio
from datetime import date
from typing import Any, Callable, Iterable

from . import nlg, nlu, warmup
from .coordinator import Call
from .engine import Engine
from .events import Clarify, FinalResponse, Speak
from .planner import (ResultStore, Step, fill_args, goal_tool, next_step, schema_values, slot_for_param,
                      tokens)
from .multimodal import HybridPerception, perception_mode
from .plugins import LLMPlugin, PayloadPerception, Perception, llm_parse
from .protocol import (
    ACK_ON_INTERRUPT, ASR_MIN_CONFIDENCE, EV_AUDIO, EV_EOT, EV_FRAME, EV_INTERRUPT, EV_MANIFEST, EV_TEXT,
    MAX_RETRIES, SNAPSHOT_INTENT, SNAPSHOT_SLOTS, SPECULATE_READ_ONLY, SPEAK_ACK, SPEAK_FILLER, SPEAK_INFO, SPEAK_PROGRESS,
    Clock, Event, ToolResult, ToolSpec, is_cumulative, is_end_of_turn, media_ref, merge_chunk,
    parse_manifest, snapshot_payload, text_of,
)
from .slowpath import INLINE_BUDGET_S, speculation_ok
from .state import SessionState
from .tools import ToolRegistry, canonical_args

MODIFY_WORDS = {"modify", "update", "change", "amend", "reschedule", "edit"}


class Agent:
    """Kit entrypoint (ASSUMPTION U02): `await Agent().setup()` then
    `await agent.run(inbox, outbox)`."""

    def __init__(self, tools: Iterable[ToolSpec] = (), llm: LLMPlugin | None = None,
                 perception: Perception | None = None, clock: Clock | None = None,
                 ref_date: date | None = None, warm: bool = True) -> None:
        if warm:
            warmup.warm_up()  # once per process: first real turn skips cold code paths
        self.engine = Engine(self, clock=clock, snapshot_fn=self.snapshot)
        self.coord = self.engine.coord
        self.calls = self.coord  # .inflight view, kept for callers/tests
        self.state = self.coord.state
        self.trace = self.engine.trace
        self.clock = self.engine.clock
        self.registry = ToolRegistry(tools)
        self.results = ResultStore()
        self.llm = llm
        if perception is None:  # hints first, real ASR/OCR for raw media (THEME5_PERCEPTION=hints: hints only)
            perception = (PayloadPerception() if perception_mode() == "hints"
                          else HybridPerception(remaining_s=self.engine.watchdog.remaining_s))
        self.perception: Perception = perception
        self.ref_date = ref_date
        self._buf = ""
        self._last_slot: str | None = None
        self._last_utt = ""
        self._derived: set[str] = set()
        self._finalized: set[str] = set()
        self._committed: dict[str, tuple[Call, Any]] = {}  # tool -> (call, result)
        self._pending_confirm: str | None = None
        self._awaiting: str | None = None  # slot named by our last clarification
        self._unannounced: set[str] = set()  # speculative call ids not yet narrated
        self._queue: list[tuple[str, str | None]] = []  # remaining parts of a compound request
        self._batch: list[str] = []  # finals of finished parts, joined into the last final
        self._forced_intent: str | None = None
        self._saw_ambiguous = False  # an ambiguous frame is pending the user's answer
        self._done_writes: dict[str, Any] = {}  # writes finished by earlier parts of a compound request

    # ------------------------------------------------------------------ lifecycle
    async def setup(self) -> None:
        """Warm-up hook (G§6: 300 s budget). Idempotent."""
        warmup.warm_up()
        for plugin in (self.llm, self.perception):
            fn = getattr(plugin, "setup", None)
            if fn is not None:
                await fn()

    async def run(self, in_q: asyncio.Queue[Any], out_q: asyncio.Queue[dict[str, Any]]) -> None:
        await self.engine.run(in_q, out_q)

    async def handle(self, ev: Event) -> None:
        await self.engine.dispatch(ev)

    async def drain(self) -> None:
        await self.engine.drain()

    # ------------------------------------------------------------------ output helpers
    def snapshot(self) -> dict[str, Any]:
        """U16/U17: intent = goal tool name, slot names = goal tool param names
        (internal names for slots the goal tool does not take)."""
        goal = goal_tool(self.state.intent, self._last_utt, self.registry) if self.state.intent else None
        intent = goal.name if (goal and SNAPSHOT_INTENT == "tool") else self.state.intent
        slots = dict(self.state.slots)
        if goal and SNAPSHOT_SLOTS == "tool_params":
            for p in goal.params:
                s = slot_for_param(p)
                if s and s != p.name and s in slots and p.name not in slots:
                    slots[p.name] = slots.pop(s)
        return snapshot_payload(intent, slots)

    def _say(self, text: str, kind: str) -> None:
        self.engine.emit(Speak(text, kind))

    def _clarify(self, text: str, slot: str | None = None) -> None:
        self.engine.emit(Clarify(text, slot))

    def _final(self, text: str) -> None:
        if self._queue:  # compound request: run the next part, report everything at the end
            self._batch.append(text)
            nxt, intent = self._queue.pop(0)
            self.state.reset_task()
            self._derived.clear()
            self._done_writes.update(self._committed)
            self._committed.clear()  # a new item, not a change to the last one (ledger still blocks repeats)
            self._forced_intent = intent
            self.on_utterance(nxt)
            return
        if self._batch:
            text, self._batch = " ".join(self._batch + [text]), []
        self.engine.emit(FinalResponse(text))

    # ------------------------------------------------------------------ events
    async def on_event(self, ev: Event) -> None:
        if ev.type == EV_MANIFEST:
            self._on_manifest(ev)
        elif ev.type in (EV_TEXT, EV_EOT):
            self._buf = merge_chunk(self._buf, text_of(ev), is_cumulative(ev))
            if is_end_of_turn(ev):
                utt, self._buf = self._buf.strip(), ""
                if utt:
                    self.on_utterance(utt)
            else:
                self._speculate()
        elif ev.type == EV_INTERRUPT:
            self._on_interrupt(ev)
        elif ev.type == EV_AUDIO:
            await self._on_audio(ev)
        elif ev.type == EV_FRAME:
            await self.engine.slow.inline_or_background(self._on_frame(ev), name="frame")

    def _on_manifest(self, ev: Event) -> None:
        self.registry.load(parse_manifest(ev))
        ref = ev.payload.get("date") or ev.payload.get("today")  # ASSUMPTION: optional reference date
        if ref and self.ref_date is None:
            try:
                self.ref_date = date.fromisoformat(str(ref)[:10])
            except ValueError:
                pass

    def _on_interrupt(self, ev: Event) -> None:
        self._buf = ""  # whatever we were accumulating is superseded by the barge-in
        txt = text_of(ev)
        if txt:
            self.on_utterance(txt)
        elif ACK_ON_INTERRUPT:
            self._say("Yes?", SPEAK_ACK)

    async def _on_audio(self, ev: Event) -> None:
        """Fast transcripts are handled inline so utterances keep event order;
        slow ASR is acknowledged and finishes in the background (R25)."""
        eot = is_end_of_turn(ev) or "end_of_turn" not in ev.payload
        fut = asyncio.ensure_future(self.perception.transcribe(ev))
        done, _ = await asyncio.wait({fut}, timeout=INLINE_BUDGET_S)
        if not done:
            self._say("One moment.", SPEAK_FILLER)
            self.engine.slow.spawn(self._finish_audio(fut, eot), name="asr")
            return
        await self._finish_audio(fut, eot)

    async def _finish_audio(self, fut: "asyncio.Future[tuple[str | None, float]]", eot: bool = True) -> None:
        try:
            text, conf = await fut
        except Exception:  # noqa: BLE001
            text, conf = None, 0.0
        if not text:
            self._clarify("Sorry, I didn't catch that. Could you say it again?")
            return
        if not eot:  # clip is a partial turn: hold it until the turn ends (U07)
            self._buf = merge_chunk(self._buf, text)
            self._speculate()
            return
        text, self._buf = merge_chunk(self._buf, text), ""
        if conf < ASR_MIN_CONFIDENCE:
            self._pending_confirm = text
            self._clarify(f'Just to check, did you say "{text}"?')
            return
        self.on_utterance(text)

    async def _on_frame(self, ev: Event) -> None:
        ref = media_ref(ev) or f"frame@{ev.t:g}"
        try:
            note, conf = await self.perception.describe(ev)
        except Exception:  # noqa: BLE001
            note, conf = None, 0.0
        self.state.add_frame(ref, note)
        ambiguous = note is not None and conf < ASR_MIN_CONFIDENCE
        self._saw_ambiguous |= ambiguous
        facts_fn = getattr(self.perception, "facts", None)
        facts = facts_fn(ev) if not ambiguous and facts_fn is not None else {}
        if facts:
            self.state.update(facts)
            if self._awaiting in ("device", "device_model", "model", "frame", "error_code") and not self._saw_ambiguous:
                self._awaiting = None  # a clear frame answers "which device?" itself, unless the
                # question came from visual ambiguity: then the user's words decide (ASSUMPTION)
        if self.state.intent is None or self._awaiting is not None:
            return  # nothing to ground yet, or the user owes us an answer first
        if ambiguous:
            self._awaiting = "device"
            self._clarify(f"I can see {note}. Which one do you mean?", "device")
            return
        self._replan(revised=False)

    # ------------------------------------------------------------------ utterances
    def on_utterance(self, utt: str) -> None:
        if self._pending_confirm is not None:
            pending, self._pending_confirm = self._pending_confirm, None
            head = nlu.parse(utt, self._last_slot, self.ref_date)
            if head.affirm and not head.slots:
                utt = pending
            elif head.deny and not head.slots:
                self._clarify("Okay, could you say it again?")
                return
        forced, self._forced_intent = self._forced_intent, None
        if forced is None:
            parts = nlu.split_compound(utt, self.ref_date)
            if len(parts) > 1:
                utt = parts[0]
                lead = nlu.parse(utt, self._last_slot, self.ref_date).intent or self.state.intent
                self._queue, self._batch = [(x, lead) for x in parts[1:]], []
        p = nlu.parse(utt, self._last_slot, self.ref_date)
        if forced and p.intent is None:
            p.intent = forced
        awaiting, self._awaiting = self._awaiting, None
        self._saw_ambiguous = self._saw_ambiguous and awaiting is None
        if awaiting and not p.slots and not p.cancel and (p.intent is None or p.intent == self.state.intent):
            v = nlu.bare_value(p.clean, awaiting, self.ref_date)
            if v is not None:
                p.slots = {awaiting: v}

        if p.cancel:
            self._queue, self._batch = [], []
            self.coord.next_generation()
            for cid in list(self.coord.inflight):
                self.engine.cancel_call(cid, "user_cancelled")
            self.state.reset_task()
            self._derived.clear()
            self._final(nlg.cancelled(bool(self._committed or self._done_writes)))
            return

        intent = p.intent
        if intent is None and self.state.intent is None:
            g = goal_tool(None, p.clean, self.registry)
            intent = g.name if g else None
        if intent is None and self.state.intent is None and not p.slots:
            if self.llm is not None:
                self._say("Let me think about that.", SPEAK_FILLER)
                self.engine.slow.spawn(self._llm_fallback(utt), name="llm")
            else:
                self._clarify("Sorry, what would you like me to do?")
            return

        if intent and self.state.intent and nlu.DOMAIN_OF_INTENT.get(intent, intent) != \
                nlu.DOMAIN_OF_INTENT.get(self.state.intent, self.state.intent) and not p.correction:
            self.state.slots = {k: v for k, v in self.state.slots.items() if k == "frame"}
            self._derived.clear()
            self.state.version += 1

        changed = self.state.update(p.slots, correction=p.correction)
        if changed - self._derived:
            for k in list(self._derived):
                self.state.slots.pop(k, None)
            self._derived.clear()
        intent_changed = self.state.set_intent(intent) if intent else False
        self._last_utt = p.clean
        changed |= self._schema_fill(p.clean)
        if p.slots:
            self._last_slot = list(p.slots)[-1]
        revised = bool(changed or intent_changed) and (bool(self.coord.inflight) or p.correction)
        if not self._replan(revised=revised):
            done = self._committed.get(self.state.intent or "") if p.intent and not changed else None
            if self.coord.inflight and not changed:
                self._say("Still working on it.", SPEAK_INFO)
            elif done is not None and (tool := self.registry.get(done[0].tool)) is not None:
                # "Great, book it." after the booking went through: restate it, never redo it
                self._final(nlg.repeat_done(tool, done[0].args, done[1]))
            elif nlu.THANKS_RE.search(p.clean):
                self._say("You're welcome.", SPEAK_ACK)
            else:
                self._say("Okay.", SPEAK_ACK)

    def _schema_fill(self, text: str) -> set[str]:
        """Unseen tools: read params no known slot covers straight from the
        utterance using the goal tool's schema (pattern, enum, currency, ids)."""
        goal = goal_tool(self.state.intent, text, self.registry) if self.state.intent else None
        if goal is None:
            return set()
        _, missing = fill_args(goal, self.state.slots)
        found = {k: v for k, v in schema_values(goal, text).items() if k in missing}
        return self.state.update(found) if found else set()

    async def _llm_fallback(self, utt: str) -> None:
        out = await llm_parse(self.llm, utt, self.state.snapshot(), self.registry.all())
        if not out or (not out.get("intent") and not out.get("slots")):
            self._clarify("Sorry, what would you like me to do?")
            return
        self.state.update(dict(out.get("slots") or {}))
        if out.get("intent"):
            self.state.set_intent(str(out["intent"]))
        if not self._replan(revised=False):
            self._clarify("Sorry, what would you like me to do?")

    # ------------------------------------------------------------------ planning
    def _relevant(self, c: Call, goal: ToolSpec | None) -> bool:
        if self.state.intent is None or goal is None:
            return False
        if c.tool == goal.name:
            return True
        tool = self.registry.get(c.tool)
        if tool is None or tool.state_modifying:
            return False
        return bool((tokens(goal.name) | tokens(goal.description)) & (tokens(tool.name) | tokens(tool.description)))

    def _still_wanted(self, state: SessionState, utterance: str) -> "Callable[[Call], bool]":
        goal = goal_tool(state.intent, utterance, self.registry)

        def wanted(c: Call) -> bool:
            tool = self.registry.get(c.tool)
            if tool is None or state.intent is None or goal is None:
                return False
            args, missing = fill_args(tool, state.slots)
            return not missing and canonical_args(args) == canonical_args(c.args) and self._relevant(c, goal)
        return wanted

    def _invalidate(self) -> int:
        """Diff the new plan against in-flight calls: keep (promote) calls whose
        args are unchanged, cancel the rest before any speech."""
        _, invalid = self.engine.reconcile(self._still_wanted(self.state, self._last_utt))
        return len(invalid)

    # ------------------------------------------------------------------ speculation
    def _speculate(self) -> None:
        """Partial turn: if a read-only call is already fully specified, start
        it silently; revise earlier speculative calls the partial no longer
        supports. Never touches session state or state-modifying tools."""
        if not SPECULATE_READ_ONLY or not speculation_ok(self._buf) or self._pending_confirm is not None:
            return
        p = nlu.parse(self._buf, self._last_slot, self.ref_date)
        if p.cancel:
            return
        intent = p.intent or self.state.intent
        if intent is None:
            g = goal_tool(None, p.clean, self.registry)
            intent = g.name if g else None
        if intent is None:
            return
        slots = dict(self.state.slots)
        if set(p.slots) - self._derived:
            for k in self._derived:
                slots.pop(k, None)
        slots.update(p.slots)
        scratch = SessionState(intent=intent, slots=slots)
        wanted = self._still_wanted(scratch, p.clean)
        for c in list(self.coord.inflight.values()):
            if c.purpose == "speculative" and not wanted(c):
                self.engine.cancel_call(c.call_id, "speculation_revised")
                self._unannounced.discard(c.call_id)
        step = next_step(scratch, self.registry, self.results, p.clean)
        if step.kind != "call" or step.tool is None or step.tool.state_modifying:
            return
        if self.engine.can_call(step.tool, step.args) is not None:
            return
        c = self.engine.start_call(step.tool, step.args, "speculative", intent)
        if c is not None:
            self._unannounced.add(c.call_id)

    def _enrich_query(self) -> None:
        """Ground a free-text query in perceived facts (e.g. a display code)."""
        q, code = self.state.slots.get("query"), self.state.slots.get("error_code")
        if isinstance(q, str) and code and str(code).lower() not in q.lower():
            self.state.update({"query": f"{q.rstrip()} (error {code})"})

    def _replan(self, revised: bool) -> bool:
        """Returns True if a user-facing action was emitted."""
        self._enrich_query()
        self._invalidate()
        step = next_step(self.state, self.registry, self.results, self._last_utt)
        if step.derived:
            self.state.update(step.derived)
            self._derived |= set(step.derived)
        return self._execute(step, revised)

    def _execute(self, step: Step, revised: bool, retry_of: Call | None = None) -> bool:
        if step.kind == "idle":
            return False
        if step.kind in ("clarify", "choose_failed"):
            self._awaiting = step.slot
            self._clarify(step.text, step.slot)
            return True
        if step.kind == "unsupported":
            self._say(step.text, SPEAK_INFO)
            return True
        if step.kind == "done":
            return self._finalize(step.tool, step.args)
        assert step.kind == "call" and step.tool is not None
        tool, args = step.tool, step.args
        if tool.state_modifying:
            done, res = self.coord.committed_result(tool.name, args)
            if done:
                self.results.put(tool.name, args, res)
                return self._finalize(tool, args)
            prior = self._committed.get(tool.name)
            if prior is not None and step.purpose == "goal" and self.engine.can_call(tool, args) is None:
                return self._modify_or_refuse(tool, prior)
        blocked = self.engine.can_call(tool, args)
        if blocked == "identical_call_in_flight":
            c = self.coord.find_inflight(f"{tool.name}|{canonical_args(args)}")
            if c is not None and c.call_id in self._unannounced:
                # a speculative call the final turn confirmed: adopt and narrate it now
                self._unannounced.discard(c.call_id)
                c.purpose = step.purpose
                self._say(nlg.progress(tool, args), SPEAK_PROGRESS)
                return True
        if blocked is not None:
            if blocked != "identical_call_in_flight":
                self.coord.issue(tool, args)  # records the blocked duplicate in the trace
            return False
        self._say(nlg.progress(tool, args, retry=retry_of is not None, revised=revised), SPEAK_PROGRESS)
        attempt = retry_of.attempt + 1 if retry_of else 1
        self.engine.start_call(tool, args, step.purpose, self.state.intent, attempt)
        return True

    def _modify_or_refuse(self, tool: ToolSpec, prior: tuple[Call, Any]) -> bool:
        mod = next((t for t in self.registry.all() if t.state_modifying and tokens(t.name) & MODIFY_WORDS), None)
        if mod is not None:
            slots = dict(self.state.slots)
            if isinstance(prior[1], dict):
                for k, v in prior[1].items():
                    slots.setdefault(k, v)
            args, missing = fill_args(mod, slots)
            if not missing:
                if self.engine.can_call(mod, args) is not None:
                    return False
                self._say(nlg.progress(mod, args, revised=True), SPEAK_PROGRESS)
                self.engine.start_call(mod, args, "goal", self.state.intent)
                return True
        self._say(nlg.already_done(tool), SPEAK_INFO)
        return True

    def _finalize(self, tool: ToolSpec | None, args: dict[str, Any]) -> bool:
        if tool is None:
            return False
        key = f"{self.state.version}|{tool.name}|{canonical_args(args)}"
        if key in self._finalized:
            return False
        self._finalized.add(key)
        ent = self.results.by_tool.get(tool.name)
        self.state.completed.add(self.state.intent or tool.name)
        self._final(nlg.final(tool, args, ent[1] if ent else None, self.state.intent))
        return True

    # ------------------------------------------------------------------ results
    def on_tool_result(self, c: Call, tr: ToolResult) -> None:
        tool = self.registry.get(c.tool)
        if tool is None:
            return
        if not tr.ok:
            unknown = c.state_modifying and tr.outcome_unknown
            can_retry = c.attempt <= MAX_RETRIES and (not c.state_modifying or (tr.retryable and not unknown))
            if can_retry:
                self._execute(Step("call", tool=tool, args=c.args, purpose=c.purpose), revised=False, retry_of=c)
            else:
                self._final(nlg.failure(tool, tr.error, uncertain=unknown))
            return
        if c.state_modifying:
            self._committed[c.tool] = (c, tr.result)
            if isinstance(tr.result, dict):
                ids = {k: v for k, v in tr.result.items()
                       if isinstance(v, (str, int)) and (k.endswith("_id") or k.endswith("_ref")
                                                         or k in ("confirmation", "pnr", "reference"))}
                self.state.update(ids)
        self.results.put(c.tool, c.args, tr.result)
        self._replan(revised=False)
