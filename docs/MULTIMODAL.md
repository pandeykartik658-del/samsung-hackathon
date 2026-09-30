# docs/MULTIMODAL.md — audio and frame perception

Owner: "Multimodal audio and frames" thread. Files: `theme5/multimodal.py`, `theme5/protocol_media.py`, `tests/test_multimodal.py`.

## Behaviour (guide 3.2(5))

1. On an audio or frame event the engine calls `mm.on_audio(ev)` / `mm.on_frame(ev, query, known)`. These are synchronous: they return a `PerceptionJob` whose `ack` ("Let me look at that.") can be emitted before any `await`.
2. Perception runs as a cancellable asyncio task. Blocking model code runs on one private worker thread per backend, so the event loop never blocks. `cancel(job_id)` / `cancel_all()` on interruption; a new frame supersedes an in-flight older frame.
3. The outcome is either `ok` (transcript text, or grounded `FrameFacts` + `slots`) or `clarify` with one targeted question. The processor never guesses: too dark, blank, two devices in view, nothing readable, or low confidence all produce a question.
4. Grounding is rule-based: backends only return raw text/labels + confidence; `ground_frame` keeps only what literally appears (device, part, model number, error code, indicator, brand). A chatty hosted VLM cannot inject facts that the rules do not find in its output.
5. `manual_lookup` flow: `plan_manual_lookup(tools, facts, query, known)` finds the manual tool in any manifest and maps facts to parameters by name (`mdlSku`, `faultCode`, `snapshotRef`, `applianceKind` with enum, ...). Missing required params produce one question via `clarify_for_missing`. `grounded_answer(facts, result)` builds the final sentence from perception + tool result only.

## Time budget (120 s cap)

- Per job: audio 15 s, frame 12 s (ASSUMPTION, `protocol_media.py`), shared across the backend chain; a timeout keeps the best partial result and marks `degraded`.
- `remaining_s` callable (engine watchdog): jobs shrink to leave 8 s reserve and skip heavy backends when the budget is gone.
- Clips are truncated to 30 s before ASR. Silent clips (< -50 dBFS) skip ASR and ask the user to repeat.
- Timeouts use the loop clock, so they run on virtual time under `sim/vloop.py` (tested).
- Measured here (CPU container): ack 0.1 ms; RapidOCR warm-up 1.2 s; 640x480 frame OCR + grounding 0.4-0.6 s.

## Backends and dependency choices

| Role | Default | Why | Installed size |
|---|---|---|---|
| ASR hint | `HintTranscriber` | Uses a transcript if the kit ships one (ASSUMPTION U19). Zero cost. | 0 |
| ASR offline | `FasterWhisperTranscriber` (`base.en`, int8, CPU) | Best accuracy per CPU-second offline; CTranslate2 needs no PyTorch. | ~155 MB packages (ctranslate2 59, numpy 45, av 32, tokenizers 12, rest ~7) + model: tiny.en 75 MB, base.en 145 MB, small.en 480 MB |
| Frame hint | `HintFrameAnalyzer` | Uses labels/caption if the kit ships them (ASSUMPTION U20). | 0 |
| Frame offline | `RapidOcrAnalyzer` | Display codes and model labels are text; PP-OCR ONNX weights are inside the wheel, so no download at runtime. | ~280 MB (opencv 188, onnxruntime 67, rapidocr 16, shapely 7) + numpy. Installing `opencv-python-headless` instead of `opencv-python` saves ~100 MB. |
| Hosted | `HostedTranscriber`, `HostedFrameAnalyzer` | Stubs with an injected async client; unavailable (and skipped) without one. | 0 |

Rejected: PyTorch Whisper / transformers VLMs (1.5-4 GB, slow on CPU); Tesseract (needs a system binary, weaker on 7-segment displays); CLIP-style object recognition (~350 MB+, adds little over OCR + the user's own words for manual lookups).

All heavy imports are lazy; the core imports with the stdlib only. With nothing installed, the agent still works on hints and asks clarifying questions.

Pinned versions tested (Python 3.11): `faster-whisper==1.2.1`, `ctranslate2==4.8.2`, `rapidocr-onnxruntime==1.4.4`, `onnxruntime==1.30.0`, `numpy`, `Pillow` (tests only). Stdlib fallback path tested on 3.10 and 3.12.

**Whisper weights must be baked into the Docker image** (`THEME5_MODEL_DIR`, `THEME5_OFFLINE=1`): the grader may have no network, and Hugging Face downloads are blocked in this dev container (so the real-model test is opt-in with `THEME5_TEST_WHISPER=1`).

## Integration notes for other threads (no edits made to their files)

- **agent.py / plugins.py (scaffold):** drop-in today: `Agent(perception=MultimodalPerception())` implements `plugins.Perception` plus `setup()`. To get the full behaviour, `_on_frame` should (a) emit `job.ack` immediately instead of waiting, (b) ask `last_clarify` / `outcome.clarify` instead of the generic "picture is a bit unclear", (c) merge `outcome.slots` into state, (d) cancel perception jobs on interruption via `mm.cancel_all()`.
- **protocol.py (scaffold):** fold `protocol_media.py` field lists and tunables into protocol.py when convenient.
- **planner.py / tools.py:** planner maps a param named `model` to the `device` slot. Frames emit a separate `model` slot (model number). Suggest a `model` slot distinct from `device`, or use `multimodal.plan_manual_lookup` for manual tools.
- **sim/ (harness):** the mock manual_lookup should accept frame ref + device/model/error_code; `manual_text` understands `answer/text/content/steps/...` result shapes.
