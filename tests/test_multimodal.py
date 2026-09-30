# /mnt/project-files/theme5/tests/test_multimodal.py
"""Tests for theme5.multimodal and theme5.protocol_media.

Run from /mnt/project-files/theme5:  python -m pytest tests/test_multimodal.py -q
Heavy-model tests skip automatically when the model is not available offline.
"""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import math
import os
import struct
import sys
import time
import wave
import zlib

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from theme5 import multimodal as mm  # noqa: E402
from theme5.protocol import Event, ParamSpec, ToolSpec, parse_tool  # noqa: E402
from theme5.protocol_media import load_media, png_size  # noqa: E402

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_OCR = importlib.util.find_spec("rapidocr_onnxruntime") is not None and importlib.util.find_spec("PIL") is not None


# ---------------------------------------------------------------------------
# fixtures: synthetic WAV / PNG with the stdlib only
# ---------------------------------------------------------------------------

def make_wav(seconds: float = 0.5, sr: int = 16000, ch: int = 1, width: int = 2, amp: float = 0.5,
             freq: float = 440.0) -> bytes:
    n = int(seconds * sr)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(width)
        w.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            v = amp * math.sin(2 * math.pi * freq * i / sr)
            for _ in range(ch):
                if width == 1:
                    frames += bytes([int(128 + v * 127)])
                elif width == 2:
                    frames += struct.pack("<h", int(v * 32767))
                else:
                    frames += int(v * 8388607).to_bytes(3, "little", signed=True)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def make_float_wav(samples: list[float], sr: int = 8000) -> bytes:
    data = struct.pack(f"<{len(samples)}f", *samples)
    fmt = struct.pack("<HHIIHH", 3, 1, sr, sr * 4, 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


def make_png(w: int = 8, h: int = 6, gray: int = 200) -> bytes:
    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes([gray]) * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# fake backends
# ---------------------------------------------------------------------------

class FakeTranscriber:
    def __init__(self, name="fake-asr", text="book a flight to delhi", conf=0.9, delay=0.0, exc=None):
        self.name, self.text, self.conf, self.delay, self.exc = name, text, conf, delay, exc
        self.calls = 0
        self.warmed = False

    async def warmup(self):
        self.warmed = True

    async def transcribe(self, media, audio, prompt):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return mm.Transcript(self.text, self.conf, self.name)


class FakeAnalyzer:
    def __init__(self, name="fake-vision", text="", labels=(), conf=0.9, delay=0.0, exc=None, quality="ok"):
        self.name, self.text, self.labels, self.conf = name, text, tuple(labels), conf
        self.delay, self.exc, self.quality = delay, exc, quality
        self.calls = 0

    async def warmup(self):
        if self.exc:
            raise self.exc

    async def analyze(self, media):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return mm.FrameReading(self.text, self.labels, self.conf, self.name, self.quality)


def frame_event(ref="frame_7", **extra):
    return Event(type="frame", t=1000.0, payload={"frame_id": ref, "data": base64.b64encode(make_png()).decode(), **extra})


def audio_event(**extra):
    return Event(type="audio", t=1000.0, payload={"audio_id": "clip_1", "data": base64.b64encode(make_wav()).decode(), **extra})


# ---------------------------------------------------------------------------
# media decoding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width,ch", [(1, 2), (2, 1), (2, 2), (3, 1)])
def test_decode_wav_pcm_variants(width, ch):
    a = mm.decode_wav(make_wav(0.25, 8000, ch, width, amp=0.5))
    assert a.sample_rate == 8000
    assert abs(a.duration_s - 0.25) < 1e-3
    assert len(a.samples) == 2000
    assert -10.0 < a.rms_dbfs < -7.0  # sine at 0.5 amplitude = -9 dBFS


def test_decode_wav_float_and_truncation():
    a = mm.decode_wav(make_float_wav([0.0, 0.5, -0.5, 0.25] * 4000, sr=8000), max_s=1.0)
    assert a.duration_s == pytest.approx(1.0)
    assert len(a.samples) == 8000
    assert max(a.samples) == pytest.approx(0.5)


def test_decode_wav_rejects_garbage_and_detects_silence():
    with pytest.raises(ValueError):
        mm.decode_wav(b"not a wav at all")
    assert mm.decode_wav(make_wav(0.2, amp=0.0)).rms_dbfs < -100


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy needed")
def test_resample_to_16k():
    a = mm.decode_wav(make_wav(0.5, sr=8000))
    x = mm.resample_16k(a)
    assert len(x) == 8000 and str(x.dtype) == "float32"


def test_load_media_sources(tmp_path, monkeypatch):
    png = make_png()
    (tmp_path / "f.png").write_bytes(png)
    m = load_media(Event("frame", 0.0, {"path": str(tmp_path / "f.png"), "labels": ["washer", "4C"]}), "frame")
    assert m.data == png and m.hint_text == "washer, 4C" and m.ref.endswith("f.png")
    monkeypatch.setenv("THEME5_MEDIA_ROOT", str(tmp_path))
    assert load_media({"path": "f.png"}, "frame").data == png
    b64 = base64.b64encode(png).decode()
    assert load_media({"frame_id": "x", "image_base64": b64}, "frame").data == png
    assert load_media({"frame_id": "x", "data": "data:image/png;base64," + b64}, "frame").data == png
    missing = load_media({"path": "nope.png"}, "frame")
    assert missing.data is None and "unreadable" in (missing.error or "")
    au = load_media({"audio_id": "c1", "transcript": "hello", "confidence": 0.4, "sample_rate": 16000}, "audio")
    assert au.hint_text == "hello" and au.hint_confidence == 0.4 and au.sample_rate == 16000 and au.data is None


def test_png_size():
    assert png_size(make_png(12, 5)) == (12, 5)
    assert png_size(b"GIF89a....") is None


# ---------------------------------------------------------------------------
# grounding
# ---------------------------------------------------------------------------

def R(text="", labels=(), conf=0.9, quality="ok", source="t"):
    return mm.FrameReading(text, tuple(labels), conf, source, quality)


def test_ground_reads_explicit_code_model_device_part():
    f = mm.ground_frame("fr1", [R("ERROR4C\nWW90T534DAW"), R(labels=("washing machine", "drain filter"), source="hint")])
    assert f.error_code == "4C" and f.code_explicit
    assert f.model == "WW90T534DAW"
    assert f.device == "washing machine" and f.part == "drain filter"
    assert f.grounded and mm.frame_clarification(f) is None
    assert mm.perceived_slots(f) == {"frame": "fr1", "device": "washing machine", "part": "drain filter",
                                     "model": "WW90T534DAW", "error_code": "4C"}


def test_ground_error_code_forms():
    assert mm.ground_frame("f", [R("Error code: UE")]).error_code == "UE"
    assert mm.ground_frame("f", [R("E21")]).error_code == "E21"
    f = mm.ground_frame("f", [R("dE", conf=0.9)])
    assert f.error_code == "DE" and not f.code_explicit and f.confidence < 0.9
    assert mm.ground_frame("f", [R("4K UHD Smart TV")]).error_code is None


def test_ground_indicator_brand_and_synonyms():
    f = mm.ground_frame("f", [R("SAMSUNG fridge, red light blinking")])
    assert f.brand == "samsung" and f.device == "refrigerator" and "red light" in (f.indicator or "")


def test_multiple_devices_is_ambiguous():
    f = mm.ground_frame("f", [R(labels=("tv", "soundbar"))])
    assert f.device is None and f.devices == ("tv", "soundbar")
    q = mm.frame_clarification(f)
    assert q and "tv" in q and "soundbar" in q
    assert mm.frame_clarification(f, known_device="soundbar") is None

    async def go():
        p = mm.MultimodalProcessor([], [FakeAnalyzer(labels=("tv", "soundbar"))])
        out = await p.on_frame(frame_event(), known={"device": "soundbar"}).task
        assert out.status == "ok" and out.facts.device == "soundbar" and out.slots["device"] == "soundbar"
    run(go())


@pytest.mark.parametrize("readings,needle", [
    ([R(quality="dark")], "too dark"),
    ([R(quality="blank")], "couldn't make anything out"),
    ([R("")], "Which device"),
    ([R("Error 4C", conf=0.3)], "is the display showing 4C"),
    ([R("washing machine", conf=0.2)], "isn't clear"),
])
def test_clarify_instead_of_guessing(readings, needle):
    q = mm.frame_clarification(mm.ground_frame("f", readings))
    assert q is not None and needle.lower() in q.lower()


def test_known_device_targets_the_part_question():
    q = mm.frame_clarification(mm.ground_frame("f", [R("")]), known_device="dishwasher")
    assert "part of the dishwasher" in q


# ---------------------------------------------------------------------------
# processor: ack first, cancellable slow path, timeouts, degradation
# ---------------------------------------------------------------------------

def test_frame_ack_is_immediate_and_work_is_async():
    async def go():
        slow = FakeAnalyzer(text="Error 4C", labels=("washing machine",), delay=0.05)
        p = mm.MultimodalProcessor([], [slow])
        job = p.on_frame(frame_event())
        assert job.ack == "Let me look at that." and job.ack_kind == "ack"
        assert not job.done and slow.calls == 0  # nothing ran before the ack could be emitted
        out = await job.task
        assert out.status == "ok" and out.facts.error_code == "4C"
        assert out.slots["device"] == "washing machine" and out.slots["frame"] == "frame_7"
        assert p.latest_frame is out.facts and not p.jobs
    run(go())


def test_acks_rotate_to_avoid_repetition():
    async def go():
        p = mm.MultimodalProcessor([], [FakeAnalyzer(text="tv")], supersede_frames=False)
        acks = [p.on_frame(frame_event(f"f{i}")).ack for i in range(3)]
        await asyncio.sleep(0)
        p.cancel_all()
        return acks
    assert len(set(run(go()))) == 3


def test_cancel_job_and_supersede_older_frame():
    async def go():
        a = FakeAnalyzer(text="tv", delay=10)
        p = mm.MultimodalProcessor([], [a])
        j1 = p.on_frame(frame_event("f1"))
        await asyncio.sleep(0.01)
        j2 = p.on_frame(frame_event("f2"))  # new frame supersedes the old one
        await asyncio.gather(j1.task, return_exceptions=True)
        assert j1.task.cancelled()
        assert p.cancel(j2.id)
        with pytest.raises(asyncio.CancelledError):
            await j2.task
        assert p.cancel_all() == [] and p.latest_frame is None
    run(go())


def test_timeout_keeps_partial_result_and_marks_degraded():
    async def go():
        hint = FakeAnalyzer("hint", labels=("washing machine", "Error 4C"))
        stuck = FakeAnalyzer("stuck", delay=60)
        p = mm.MultimodalProcessor([], [hint, stuck], frame_timeout_s=0.05)
        t0 = time.perf_counter()
        out = await p.on_frame(frame_event()).task
        assert time.perf_counter() - t0 < 1.0
        assert out.degraded and "stuck: timeout" in out.notes
        assert out.status == "ok" and out.facts.error_code == "4C"
    run(go())


def test_scenario_budget_skips_heavy_backends():
    async def go():
        a = FakeAnalyzer(text="tv")
        p = mm.MultimodalProcessor([], [a], remaining_s=lambda: 5.0)  # less than the reserve
        out = await p.on_frame(frame_event()).task
        assert a.calls == 0 and out.degraded and out.status == "clarify"
    run(go())


def test_backend_crash_is_contained():
    async def go():
        boom = FakeAnalyzer("boom", exc=RuntimeError("segfault-ish"))
        ok = FakeAnalyzer("ok", text="Error E21", labels=("dishwasher",))
        out = await mm.MultimodalProcessor([], [boom, ok]).on_frame(frame_event()).task
        assert out.status == "ok" and out.facts.error_code == "E21" and "boom: error RuntimeError" in out.notes
    run(go())


def test_frame_with_no_backends_asks_to_reshow():
    async def go():
        out = await mm.MultimodalProcessor([], []).on_frame(frame_event()).task
        assert out.status == "clarify" and out.slots == {"frame": "frame_7"}
        assert "point the camera" in out.clarify
    run(go())


def test_audio_hint_preferred_then_fallback_chain():
    async def go():
        heavy = FakeTranscriber("heavy", text="should not run")
        p = mm.MultimodalProcessor([mm.HintTranscriber(), heavy], [])
        job = p.on_audio(audio_event(transcript="Change it to Friday", confidence=0.95))
        assert job.ack_kind == "filler"
        out = await job.task
        assert out.status == "ok" and out.text == "Change it to Friday" and heavy.calls == 0
        out2 = await p.on_audio(audio_event()).task  # no hint -> falls through to heavy
        assert out2.text == "should not run" and out2.source == "heavy"
    run(go())


def test_audio_low_confidence_asks_to_confirm():
    async def go():
        p = mm.MultimodalProcessor([FakeTranscriber(text="book to Dehli", conf=0.3)], [])
        out = await p.on_audio(audio_event()).task
        assert out.status == "clarify" and out.text == "book to Dehli" and "Did you say" in out.clarify
    run(go())


def test_audio_silence_and_unavailable_backends():
    async def go():
        silent = Event("audio", 0.0, {"audio_id": "s", "data": base64.b64encode(make_wav(amp=0.0)).decode()})
        t = FakeTranscriber()
        out = await mm.MultimodalProcessor([t], []).on_audio(silent).task
        assert out.status == "clarify" and "didn't hear anything" in out.clarify and t.calls == 0
        nob = mm.MultimodalProcessor([mm.HintTranscriber(), mm.HostedTranscriber()], [])
        out2 = await nob.on_audio(audio_event()).task
        assert out2.status == "clarify" and out2.degraded and "say it again" in out2.clarify
    run(go())


def test_hosted_stubs_with_injected_clients():
    async def asr(data, meta):
        return {"text": "turn it off", "confidence": 0.8}

    async def vlm(data, meta):
        return {"caption": "a Samsung washing machine showing error 5E", "labels": ["drain hose"], "confidence": 0.85}

    async def go():
        p = mm.MultimodalProcessor([mm.HostedTranscriber(asr)], [mm.HostedFrameAnalyzer(vlm)])
        a = await p.on_audio(audio_event()).task
        f = await p.on_frame(frame_event()).task
        assert a.text == "turn it off" and a.source == "hosted-asr"
        assert f.facts.error_code == "5E" and f.facts.part == "drain hose" and f.facts.brand == "samsung"
    run(go())


def test_warmup_records_status_and_never_raises():
    class Hang:
        name = "hang"

        async def warmup(self):
            await asyncio.sleep(60)

    async def go():
        p = mm.MultimodalProcessor([FakeTranscriber()], [FakeAnalyzer("bad", exc=ImportError("no cv2")), Hang()])
        st = await p.warmup(budget_s=0.05)
        assert st["fake-asr"] == "ready" and st["bad"].startswith("unavailable") and st["hang"] == "timeout"
    run(go())


def test_whisper_without_model_degrades_gracefully(tmp_path):
    """Offline + empty model dir: load fails fast, status says why, audio still yields a clarify."""
    async def go():
        w = mm.FasterWhisperTranscriber(model="tiny.en", download_root=str(tmp_path), local_files_only=True)
        p = mm.MultimodalProcessor([mm.HintTranscriber(), w], [])
        st = await p.warmup(budget_s=30)
        if st[w.name] == "ready":  # pragma: no cover - model somehow present
            pytest.skip("model available locally")
        assert st[w.name].startswith("unavailable")
        out = await p.on_audio(audio_event()).task
        assert out.status == "clarify" and any("faster-whisper: unavailable" in n for n in out.notes)
        p.close()
    run(go())


def test_timeouts_run_on_the_virtual_clock():
    """Under the harness's virtual-time loop a 12 s timeout costs no wall time."""
    vloop_path = os.path.join(ROOT, "sim", "vloop.py")
    if not os.path.exists(vloop_path):
        pytest.skip("sim/vloop.py not present")
    spec = importlib.util.spec_from_file_location("_vloop", vloop_path)
    vloop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vloop)

    async def go():
        p = mm.MultimodalProcessor([], [FakeAnalyzer("hint", labels=("oven",)), FakeAnalyzer("stuck", delay=500)])
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        out = await p.on_frame(frame_event()).task
        return out, loop.time() - t0

    w0 = time.perf_counter()
    out, virtual = vloop.run_virtual(go())
    assert virtual == pytest.approx(mm.FRAME_TIMEOUT_S, abs=0.01)
    assert time.perf_counter() - w0 < 2.0
    assert out.facts.device == "oven" and out.degraded


# ---------------------------------------------------------------------------
# manual_lookup flow with manifests the code has never seen
# ---------------------------------------------------------------------------

MANIFEST_A = [  # conventional names
    {"name": "manual_lookup", "description": "Look up the device manual for a frame", "read_only": True,
     "parameters": {"type": "object", "properties": {
         "frame_id": {"type": "string"}, "device": {"type": "string"}, "query": {"type": "string"}},
         "required": ["frame_id", "query"]}},
    {"name": "create_ticket", "description": "Open a support ticket", "parameters": {"type": "object", "properties": {
        "issue": {"type": "string"}}, "required": ["issue"]}},
]
MANIFEST_B = [  # unusual names, camelCase, enum, no explicit read-only marker
    {"name": "fetchApplianceHandbookSection", "description": "Troubleshooting guide excerpt",
     "params": {"applianceKind": {"type": "string", "enum": ["WASHER", "DRYER", "FRIDGE"], "required": True},
                "mdlSku": {"type": "string", "required": True},
                "faultCode": {"type": "string"},
                "snapshotRef": {"type": "string"}}},
]
MANIFEST_C = [  # list-style params, needs a part the frame may not show
    {"name": "docs.search", "description": "Search product documentation", "kind": "read",
     "parameters": [{"name": "component", "type": "string", "required": True},
                    {"name": "keywords", "type": "array"}, {"name": "page_limit", "type": "integer"}]},
    {"name": "book_technician", "description": "Book a technician visit", "parameters": [
        {"name": "date", "type": "string", "required": True}]},
]


def tools(manifest):
    return [parse_tool(t) for t in manifest]


def test_manifest_a_conventional():
    facts = mm.ground_frame("frame_7", [R("Error 4C"), R(labels=("washing machine",))])
    plan = mm.plan_manual_lookup(tools(MANIFEST_A), facts, "what does this code mean")
    assert plan.tool.name == "manual_lookup" and plan.ready
    assert plan.args == {"frame_id": "frame_7", "device": "washing machine", "query": "what does this code mean"}


def test_manifest_b_unusual_names_and_enum():
    facts = mm.ground_frame("fr9", [R("ERROR4C\nWW90T534DAW"), R(labels=("washer",))])
    t = tools(MANIFEST_B)
    assert t[0].state_modifying  # no read-only marker: safe default from protocol.py
    plan = mm.plan_manual_lookup(t, facts, None, tool=t[0])  # engine may pick it explicitly
    assert plan.ready
    assert plan.args == {"applianceKind": "WASHER", "mdlSku": "WW90T534DAW", "faultCode": "4C", "snapshotRef": "fr9"}
    # without the model number on camera, it is missing and we ask for the label
    facts2 = mm.ground_frame("fr10", [R("Error 4C washer")])
    plan2 = mm.plan_manual_lookup(t, facts2, None, tool=t[0])
    assert plan2.missing == ["mdlSku"] and "model number" in mm.clarify_for_missing(plan2)


def test_manifest_c_list_params_and_user_slots():
    t = tools(MANIFEST_C)
    assert mm.find_manual_tool(t).name == "docs.search"
    facts = mm.ground_frame("f", [R("Error 5E")])
    plan = mm.plan_manual_lookup(t, facts, None)
    assert plan.missing == ["component"] and "Which part" in mm.clarify_for_missing(plan)
    assert plan.args == {"keywords": ["error 5E"]}  # integer param left alone
    plan2 = mm.plan_manual_lookup(t, facts, None, known={"part": "drain filter"})
    assert plan2.ready and plan2.args["component"] == "drain filter"


def test_no_manual_tool_returns_none():
    assert mm.plan_manual_lookup(tools(MANIFEST_C)[1:], None, "x") is None


def test_param_field_mapping():
    assert mm.param_field("deviceModel") == "model"
    assert mm.param_field("image_ref") == "frame"
    assert mm.param_field("fault_code_str") == "error_code"
    assert mm.param_field("product_name") == "device"
    assert mm.param_field("page_limit") is None


def test_grounded_answer_uses_only_perception_and_result():
    facts = mm.ground_frame("f", [R("Error 4C"), R(labels=("washing machine",))])
    res = {"page": 42, "answer": "4C means a water supply problem. Check the tap is open and the inlet hose is not kinked."}
    ans = mm.grounded_answer(facts, res)
    assert ans.startswith("I can see error 4C on the washing machine.")
    assert "water supply problem" in ans and "(page 42)" in ans
    assert mm.grounded_answer(facts, {"results": []}) == "I can see error 4C on the washing machine. I couldn't find that in the manual."
    assert "Step one; Step two" in mm.grounded_answer(None, {"steps": ["Step one", "Step two"]})
    long = mm.grounded_answer(None, "A. " * 400)
    assert len(long) < 360


def test_end_to_end_frame_to_manual_answer():
    async def manual_lookup(args):  # stand-in for the harness mock tool
        assert args["frame_id"] == "frame_7" and args["device"] == "washing machine"
        return {"text": "Code 4C: no water supply. Open the tap and clean the inlet filter."}

    async def go():
        p = mm.MultimodalProcessor([], [FakeAnalyzer("hint", labels=("washing machine", "display: 4C"))])
        job = p.on_frame(frame_event(), query="what does this mean")
        spoken = [job.ack]
        out = await job.task
        assert out.status == "ok"
        plan = mm.plan_manual_lookup(tools(MANIFEST_A), out.facts, "what does this mean")
        assert plan.ready
        spoken.append(mm.grounded_answer(out.facts, await manual_lookup(plan.args)))
        return spoken

    spoken = run(go())
    assert spoken[0] == "Let me look at that."
    assert "4C" in spoken[1] and "inlet filter" in spoken[1]


# ---------------------------------------------------------------------------
# real offline OCR (skips if RapidOCR / Pillow missing)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_OCR, reason="rapidocr-onnxruntime and Pillow not installed")
def test_rapidocr_reads_display_code_and_model():
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (520, 220), "white")
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=36)
    d.text((20, 30), "ERROR 4C", fill="black", font=font)
    d.text((20, 120), "Model: WW90T534DAW", fill="black", font=font)
    b = io.BytesIO()
    im.save(b, "PNG")
    dark = io.BytesIO()
    Image.new("RGB", (64, 64), (5, 5, 5)).save(dark, "PNG")

    async def go():
        p = mm.MultimodalProcessor([], [mm.RapidOcrAnalyzer()])
        st = await p.warmup(budget_s=120)
        assert st["rapidocr"] == "ready"
        t0 = time.perf_counter()
        out = await p.on_frame(Event("frame", 0.0, {"frame_id": "ocr1", "data": base64.b64encode(b.getvalue()).decode()})).task
        dt = time.perf_counter() - t0
        out_dark = await p.on_frame(Event("frame", 0.0, {"frame_id": "d", "data": base64.b64encode(dark.getvalue()).decode()})).task
        p.close()
        return out, dt, out_dark

    out, dt, out_dark = run(go())
    assert out.facts.error_code == "4C" and out.facts.model == "WW90T534DAW"
    assert out.status == "ok" and dt < 10.0
    assert out_dark.status == "clarify" and "too dark" in out_dark.clarify


@pytest.mark.skipif(os.environ.get("THEME5_TEST_WHISPER") != "1", reason="set THEME5_TEST_WHISPER=1 with a local model")
def test_faster_whisper_real_model():  # pragma: no cover - needs model weights
    async def go():
        w = mm.FasterWhisperTranscriber()
        p = mm.MultimodalProcessor([w], [])
        st = await p.warmup(budget_s=280)
        assert st[w.name] == "ready", st
        out = await p.on_audio(audio_event()).task  # a pure tone: must not hallucinate a confident sentence
        assert out.status == "clarify" or out.confidence < 0.9
    run(go())


# ---------------------------------------------------------------------------
# adapter for the scaffold's plugins.Perception interface
# ---------------------------------------------------------------------------

def test_perception_adapter_contract():
    async def go():
        p = mm.MultimodalPerception(mm.MultimodalProcessor(
            [mm.HintTranscriber()], [FakeAnalyzer(labels=("washing machine", "Error 4C"))]))
        await p.setup()
        assert await p.transcribe(audio_event(transcript="yes please", confidence=0.9)) == ("yes please", 0.9)
        text, conf = await p.transcribe(audio_event(transcript="to dehli", confidence=0.3))
        assert text == "to dehli" and conf < mm.ASR_MIN_CONF
        assert await p.transcribe(audio_event()) == (None, 0.0)
        note, conf = await p.describe(frame_event())
        assert note == "washing machine; error 4C" and conf >= mm.FRAME_MIN_CONFIDENCE
        assert p.last_facts.error_code == "4C" and p.last_clarify is None
        p2 = mm.MultimodalPerception(mm.MultimodalProcessor([], [FakeAnalyzer(labels=("tv", "soundbar"))]))
        note, conf = await p2.describe(frame_event())
        assert conf == 0.0 and "Which one" in p2.last_clarify
    run(go())


def test_cancelling_the_awaiting_task_cancels_perception():
    async def go():
        a = FakeAnalyzer(text="tv", delay=30)
        p = mm.MultimodalPerception(mm.MultimodalProcessor([], [a]))
        outer = asyncio.ensure_future(p.describe(frame_event()))
        await asyncio.sleep(0.01)
        outer.cancel()
        await asyncio.gather(outer, return_exceptions=True)
        await asyncio.sleep(0)
        assert outer.cancelled() and not p.mm.jobs
    run(go())
