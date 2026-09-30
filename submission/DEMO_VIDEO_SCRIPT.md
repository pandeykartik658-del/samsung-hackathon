# /mnt/project-files/theme5/jury/DEMO_VIDEO_SCRIPT.md
# Demo video script (target 4:30, hard cap 5:00)

Theme 05, Interruptible Real-Time Agents. Built around the trace viewer. Every number and line of dialogue below comes from a real run on 2026-09-30 (`bash run_demo.sh`, `python -m bench.run_suite bench/scenarios60`). Say out loud, once, that scores come from our own simulator and scorer because the official kit is not released.

## Before recording

```bash
cd theme5-submission            # or the cloned repo
bash run_demo.sh                # writes runs/demo/ (scenarios, bench, trace_viewer.html)
```

Open in a browser, one tab each (already built, trace embedded):
- `submission/viewer/s02_text_destination_change.html`
- `submission/viewer/s07_text_duplicate_booking_trap.html`
- `submission/viewer/s09_visual_ambiguous_frame.html`

Screen at 1600x1000 or larger, browser zoom 110%, light theme. Keep a terminal open in the repo, font 16 pt.

## Shot list

| # | Time | On screen | Voice-over |
|---|---|---|---|
| 1 | 0:00-0:25 | Deck slide 1, then slide 2 | "Assistants take turns: listen, think, speak. People don't. They cut in, correct themselves and change their mind while a search or a booking is still running. We built an agent that keeps talking, cancels the work nobody wants any more, re-plans, and never books twice." |
| 2 | 0:25-1:00 | Deck slide 4 (architecture) | "Events come in on one queue, actions go out on another. Only protocol.py knows the evaluation kit's field names, so the real kit is a one-file change. The fast path is rules only, no LLM, so the first answer goes out on the same event tick. The coordinator owns every tool call: cancel tokens, generation counters and an idempotency ledger. Slow work, like speech recognition and OCR, runs as cancellable background tasks, and a watchdog closes any turn still open at 110 seconds." |
| 3 | 1:00-2:00 | s02 viewer tab. Press **Play** at 1x. Pause at about 2.0 s. Point at the red dashed line, then the grey Delhi bar ending at 1.9 s. Resume to the end. | "The user asks for flights from Bangalore to Delhi. At 0.4 seconds, at the end of the sentence, the agent says what it is doing and starts the search. At 1.9 seconds the user interrupts: 'Actually, make that Hyderabad instead.' On that same tick the Delhi search is cancelled, before the agent says anything, and a Hyderabad search starts. The snapshot on the right now says destination Hyderabad. At 4.9 seconds the result lands and the agent answers with two flights. Anything the Delhi call returns later is marked stale and ignored." |
| 4 | 2:00-2:50 | s07 viewer tab. Play. Pause at 4.5 s, then at 6.0 s. | "Now the trap. 'Book flight AI-202 for Rahul Verma.' The booking takes six seconds. At 2.5 seconds the user asks again, 'Did that go through? Please book AI-202', and at 4.5 seconds, 'Just book it now please.' Each time the agent says it is still working and sends nothing. The ledger has the booking reserved under a key built from the tool and its arguments. One booking, reference RV4T9N, not three." |
| 5 | 2:50-3:35 | s09 viewer tab. Play. Pause at 0.3 s, then 5.0 s, then 5.7 s. | "Camera frame, and: 'How do I fix the error on this?' The frame is ambiguous, so the agent doesn't guess. It asks one question: 'Which device is this about?' The user says 'The washer. It's showing 4C.' With a second frame, it looks up the manual for model WW90T and answers from the manual: error 4C, water supply, check that the tap is fully open." |
| 6 | 3:35-4:10 | Terminal: `python -m bench.run_suite bench/scenarios60 --agent theme5.agent:Agent` (scroll to the MEAN / Suite lines) | "We can't see the hidden set, so we built our own: 60 adversarial scenarios in 13 families, 30 text, 18 audio, 12 visual, scored by a replica of the guide's rubric. Suite score 100, and 100 on a fresh seed it was never tuned on. First spoken action in 0.69 milliseconds median. Zero crashes across 60,000 malformed events. To be clear, these are our own simulator and scorer. The official kit isn't out yet." |
| 7 | 4:10-4:30 | Deck slide 8 (limitations), then slide 9 | "What's not done: the kit's field names are still guesses, isolated in one file. By default the agent trusts the kit's transcripts; Whisper and OCR are built and tested in CI but not switched on. Next: conform to the real kit, then an on-device intent model and streaming speech. Everything runs offline in one Docker image. Thank you." |

Total spoken words: about 560, which fits 4:30 at a relaxed pace.

## Checks before uploading

- Length under 5:00 (deck p11/p13 limit).
- Upload to YouTube (unlisted) or Drive with link sharing on, then put the link on deck slide 5 and slide 11, and in the Google Form.
- Put the final deck and video link in the tagged commit (`PRISM_GENAI_HACKATHON_Y2026`); deck p13 says everything referenced must be in the tagged commit.

## Sources for every claim

| Claim | Source |
|---|---|
| s02 timings 0.4 / 1.9 / 4.9 s, cancel on the same tick | `runs/demo/scenarios/traces/s02_text_destination_change.jsonl` |
| s07 repeats at 2.5 and 4.5 s, one booking RV4T9N at 6.0 s | `runs/demo/scenarios/traces/s07_text_duplicate_booking_trap.jsonl` |
| s09 clarify at 0.3 s, manual WW90T error 4C at 5.7 s | `runs/demo/scenarios/traces/s09_visual_ambiguous_frame.jsonl` |
| Bench 100.0, 13 families, 30/18/12 | `bench/runs` REPORT.md from `bench.run_suite bench/scenarios60` |
| Held-out seed 100.0 | `python -m bench.gen_scenarios --seed 20261015` then `bench.run_suite` |
| 0.69 ms median, p95 1.27 ms; 60,000 fuzzed events | `docs/HARDENING.md` |
| Whisper built and tested, not on by default | `theme5/agent.py` defaults to `PayloadPerception`; CI run https://github.com/pandeykartik658-del/samsung-hackathon/actions/runs/36743838976 |
| Without kit transcripts and models: suite 78.2, audio 54.7 | `bench.run_suite bench/scenarios60 --strip-oracle` (no Whisper/OCR installed) |
