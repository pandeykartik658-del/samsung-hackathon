# /mnt/project-files/theme5/docs/SLOTS_INTEGRATION.md
# slots.py integration (for the agent.py owner)

`theme5/slots.py` is not wired in yet; `agent.on_utterance` still uses `nlu.parse` + `SessionState.update`,
and it wipes every slot except `frame` on a cross-domain goal change. Proposed hook, no other file changes:

```python
# Agent.__init__
self.slots = SlotTracker(ref_date=ref_date)          # or SlotTracker(parser=HybridParser(llm, ref_date, tools))
# on tool manifest
self.slots.register_tools(self.registry.all())       # unseen tools get a relevance set from their params
# on each text chunk (fast path, sync, ~0.5 ms)
upd = self.slots.feed(text, chunk_id=ev.id, t=ev.t, end_of_turn=is_end_of_turn(ev), confidence=asr_conf)
upd.apply_to(self.state)                              # keeps SessionState.version as the staleness signal
fp.on_slots_changed(upd.transitions)                  # {slot: (old, new)} for correction acks
if upd.clarification: emit ACT_CLARIFY(text=upd.clarification.question, slot=upd.clarification.slot)
if upd.changed or upd.intent_changed: re-plan; cancel in-flight calls whose args used a changed slot
# derived / perception values
self.slots.set_slot("frame", ref, source=ev.id)
```

`SlotUpdate` fields: `changed` (slot -> new value, None = removed), `corrected` (had a value before this turn),
`dropped` (parked by a goal change), `intent`, `intent_changed`, `clarification`, `snapshot`, `final`.
Parked slots come back if the goal returns (ticket -> booking). `reset()` at session end; nothing is module-level.

Slot names match `planner.SLOT_SYNONYMS` keys plus `return_date` and `booking_ref`, which the planner
may want as synonyms (`return_date`, `pnr`, `booking_id`). ASSUMPTION: value formats follow `nlu`
(ISO dates when a reference date is set, else "5th", "friday", "march 5").
