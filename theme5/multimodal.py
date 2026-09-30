# /mnt/project-files/theme5/theme5/multimodal.py
"""Multimodal perception: audio clips (WAV) and camera frames (PNG).

Contract with the engine (guide 3.2(5): process media behind conversational
acknowledgments, clarify ambiguous perceptions instead of guessing):

    job = mm.on_frame(event, query="what does this light mean")   # sync, no await
    emit speak(job.ack)                                          # fast path, immediately
    outcome = await job.task                                     # slow path, cancellable
    if outcome.status == "clarify": emit clarify(outcome.clarify)
    else: state.update(outcome.slots); plan = plan_manual_lookup(tools, outcome.facts, ...)

Backends are pluggable behind two small interfaces:
  Transcriber.transcribe(media, audio, prompt) -> Transcript
  FrameAnalyzer.analyze(media) -> FrameReading
Offline defaults: faster-whisper (ASR) and RapidOCR (frame text). Hosted stubs
take an injected async client. Harness-provided hints (reference transcript,
frame labels) are used first when present (SPEC U19/U20, ASSUMPTION).

Grounding is rule-based and deterministic: backends only return raw text and
confidence; this module decides what was perceived (device, part, model
number, error code, indicator) and never adds facts that were not perceived.
No LLM anywhere in this file.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import itertools
import math
import os
import re
import struct
from array import array
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Iterable, Mapping, Protocol, Sequence

from .protocol import (SPEAK_ACK, SPEAK_FILLER, WARMUP_CAP_S, Event, ToolSpec, audio_transcript, frame_caption,
                       frame_labels, media_ref)
from .protocol_media import (
    ASR_MIN_CONF, AUDIO_TIMEOUT_S, FRAME_MIN_CONFIDENCE, FRAME_TIMEOUT_S, MAX_AUDIO_S, PERCEPTION_ENV,
    SCENARIO_RESERVE_S, SILENCE_DBFS, MediaInput, load_media, png_size,
)

# ===========================================================================
# Data types
# ===========================================================================


class BackendUnavailable(RuntimeError):
    """A backend cannot serve this request (not installed, not configured, no data)."""


@dataclass(frozen=True)
class WavAudio:
    samples: Any  # mono float32 in [-1, 1]: numpy.ndarray if numpy is present, else array('f')
    sample_rate: int
    duration_s: float
    rms_dbfs: float


@dataclass(frozen=True)
class Transcript:
    text: str
    confidence: float  # 0..1
    source: str


@dataclass(frozen=True)
class FrameReading:
    """Raw output of a FrameAnalyzer: what it read/saw, not what it means."""
    text: str = ""  # OCR text or caption
    labels: tuple[str, ...] = ()  # object/scene labels
    confidence: float = 0.0  # 0..1
    source: str = ""
    quality: str = "ok"  # "ok" | "dark" | "blank" | "unreadable"


@dataclass(frozen=True)
class FrameFacts:
    """Grounded perception. Every non-None field was perceived in the frame."""
    ref: str
    device: str | None = None
    devices: tuple[str, ...] = ()  # all distinct devices seen (>1 means ambiguous)
    part: str | None = None
    model: str | None = None
    error_code: str | None = None
    code_explicit: bool = False  # code was preceded by "error"/"code" on screen
    brand: str | None = None
    indicator: str | None = None
    text: str = ""
    confidence: float = 0.0
    quality: str = "ok"
    source: str = ""

    @property
    def grounded(self) -> bool:
        return any((self.device, self.devices, self.part, self.model, self.error_code, self.indicator))


@dataclass
class PerceptionOutcome:
    job_id: str
    kind: str  # "audio" | "frame"
    ref: str
    status: str  # "ok" | "clarify"
    text: str = ""  # audio: transcript (tentative when status == "clarify")
    confidence: float = 0.0
    clarify: str | None = None  # question to ask when status == "clarify"
    facts: FrameFacts | None = None
    slots: dict[str, Any] = field(default_factory=dict)  # perceived slots to merge into state
    source: str = ""
    degraded: bool = False  # a timeout, missing backend or budget cut shaped this result
    notes: tuple[str, ...] = ()  # trace-friendly diagnostics


@dataclass
class PerceptionJob:
    id: str
    kind: str
    ref: str
    ack: str  # speak this immediately
    ack_kind: str  # protocol SPEAK_* kind
    task: "asyncio.Task[PerceptionOutcome]"
    started_at: float

    def cancel(self) -> bool:
        return self.task.cancel()

    @property
    def done(self) -> bool:
        return self.task.done()


# ===========================================================================
# Media decoding (stdlib, numpy used when present)
# ===========================================================================

try:  # optional
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - exercised only without numpy
    _np = None


def decode_wav(data: bytes, max_s: float = MAX_AUDIO_S) -> WavAudio:
    """RIFF/WAVE PCM 8/16/24/32-bit int or 32/64-bit float, any channel count.
    Mixes to mono and truncates to max_s. Raises ValueError on malformed input."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            pcm = body
            break
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None or len(fmt) < 16:
        raise ValueError("missing fmt or data chunk")
    tag, ch, sr, _, _, bits = struct.unpack("<HHIIHH", fmt[:16])
    if tag == 0xFFFE and len(fmt) >= 26:  # WAVE_FORMAT_EXTENSIBLE: real tag in subformat GUID
        tag = struct.unpack("<H", fmt[24:26])[0]
    if ch < 1 or sr < 1 or bits not in (8, 16, 24, 32, 64) or tag not in (1, 3):
        raise ValueError(f"unsupported wav format tag={tag} bits={bits} ch={ch}")
    width = bits // 8
    frame_bytes = width * ch
    n = min(len(pcm) // frame_bytes, int(max_s * sr))
    pcm = pcm[:n * frame_bytes]
    if _np is not None:
        if tag == 3:
            x = _np.frombuffer(pcm, dtype="<f4" if bits == 32 else "<f8").astype(_np.float32)
        elif bits == 8:
            x = (_np.frombuffer(pcm, dtype=_np.uint8).astype(_np.float32) - 128.0) / 128.0
        elif bits == 24:
            b = _np.frombuffer(pcm, dtype=_np.uint8).reshape(-1, 3).astype(_np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            v = _np.where(v >= 1 << 23, v - (1 << 24), v)
            x = v.astype(_np.float32) / float(1 << 23)
        else:
            x = _np.frombuffer(pcm, dtype="<i2" if bits == 16 else "<i4").astype(_np.float32)
            x /= float(1 << (bits - 1))
        x = x.reshape(-1, ch).mean(axis=1).astype(_np.float32) if ch > 1 else x
        rms = float(_np.sqrt(_np.mean(x * x))) if x.size else 0.0
        samples: Any = x
    else:
        vals = _decode_pcm_pure(pcm, tag, bits)
        mono = array("f", (sum(vals[i:i + ch]) / ch for i in range(0, len(vals), ch))) if ch > 1 else vals
        rms = math.sqrt(sum(v * v for v in mono) / len(mono)) if len(mono) else 0.0
        samples = mono
    dbfs = 20.0 * math.log10(rms) if rms > 1e-10 else -200.0
    return WavAudio(samples=samples, sample_rate=sr, duration_s=n / sr, rms_dbfs=dbfs)


def _decode_pcm_pure(pcm: bytes, tag: int, bits: int) -> array:
    if tag == 3:
        a = array("f" if bits == 32 else "d")
        a.frombytes(pcm)
        return array("f", a)
    if bits == 8:
        return array("f", ((b - 128) / 128.0 for b in pcm))
    if bits == 24:
        out = array("f")
        for i in range(0, len(pcm), 3):
            v = int.from_bytes(pcm[i:i + 3], "little", signed=True)
            out.append(v / float(1 << 23))
        return out
    a = array("h" if bits == 16 else "i")
    a.frombytes(pcm)
    scale = float(1 << (bits - 1))
    return array("f", (v / scale for v in a))


def resample_16k(audio: WavAudio) -> Any:
    """numpy float32 mono at 16 kHz (linear interpolation; adequate for ASR)."""
    if _np is None:
        raise BackendUnavailable("numpy required for ASR")
    x = _np.asarray(audio.samples, dtype=_np.float32)
    if audio.sample_rate == 16000 or x.size == 0:
        return x
    n_out = int(round(x.size * 16000 / audio.sample_rate))
    t_in = _np.arange(x.size, dtype=_np.float64) / audio.sample_rate
    t_out = _np.arange(n_out, dtype=_np.float64) / 16000.0
    return _np.interp(t_out, t_in, x).astype(_np.float32)


# ===========================================================================
# Backend interfaces
# ===========================================================================


class Transcriber(Protocol):
    name: str

    async def warmup(self) -> None: ...

    async def transcribe(self, media: MediaInput, audio: WavAudio | None, prompt: str | None) -> Transcript: ...


class FrameAnalyzer(Protocol):
    name: str

    async def warmup(self) -> None: ...

    async def analyze(self, media: MediaInput) -> FrameReading: ...


_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}  # process-wide: key -> loaded model
_LOAD_ERRORS: dict[tuple[Any, ...], str] = {}  # process-wide: key -> why loading failed


class _ExecutorBackend:
    """Runs blocking model code on one private worker thread so the event loop
    never blocks and model access is serialised. A cancelled job's thread runs
    to completion but its result is discarded."""
    name = "executor"

    def __init__(self) -> None:
        self._ex: concurrent.futures.ThreadPoolExecutor | None = None
        self._model: Any = None
        self._load_error: str | None = None
        self._load_lock: asyncio.Lock | None = None

    def _executor(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._ex is None:
            self._ex = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix=self.name)
        return self._ex

    async def _run(self, fn: Callable[..., Any], *args: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self._executor(), fn, *args)

    def _load(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def _cache_key(self) -> tuple[Any, ...] | None:
        """Loaded models are shared by every backend with the same key in this process,
        so a fresh Agent per scenario does not reload weights. None disables sharing."""
        return None

    async def warmup(self) -> None:
        await self._ensure()

    async def _ensure(self) -> Any:
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            raise BackendUnavailable(self._load_error)
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()
        async with self._load_lock:
            key = self._cache_key()
            if self._model is None and key is not None:
                self._model = _MODEL_CACHE.get(key)
                self._load_error = self._load_error or _LOAD_ERRORS.get(key)  # don't retry a failed load
            if self._model is None and self._load_error is None:
                try:
                    self._model = await self._run(self._load)
                    if key is not None:
                        _MODEL_CACHE[key] = self._model
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # ImportError, download blocked, bad model files ...
                    self._load_error = f"{self.name}: {e.__class__.__name__}: {str(e)[:160]}"
                    if key is not None:
                        _LOAD_ERRORS[key] = self._load_error
        if self._model is None:
            raise BackendUnavailable(self._load_error or f"{self.name}: not loaded")
        return self._model

    def close(self) -> None:
        if self._ex is not None:
            self._ex.shutdown(wait=False, cancel_futures=True)
            self._ex = None


# ---- audio backends -----------------------------------------------------------


class HintTranscriber:
    """Uses a transcript shipped with the clip (ASSUMPTION: the kit may provide one)."""
    name = "hint"

    async def warmup(self) -> None:
        return None

    async def transcribe(self, media: MediaInput, audio: WavAudio | None, prompt: str | None) -> Transcript:
        if not media.hint_text:
            raise BackendUnavailable("no transcript hint")
        return Transcript(media.hint_text.strip(), max(0.0, min(1.0, media.hint_confidence)), self.name)


class FasterWhisperTranscriber(_ExecutorBackend):
    """Offline ASR with faster-whisper (CTranslate2, int8 on CPU).

    Model: THEME5_WHISPER_MODEL (default base.en, ~145 MB; tiny.en ~75 MB is
    ~2x faster, small.en ~480 MB more accurate). Weights come from
    THEME5_MODEL_DIR if set (bake them into the Docker image; the grader may
    have no network). THEME5_OFFLINE=1 forbids downloads."""
    name = "faster-whisper"

    def __init__(self, model: str | None = None, compute_type: str = "int8", cpu_threads: int = 0,
                 download_root: str | None = None, local_files_only: bool | None = None) -> None:
        super().__init__()
        self.model_name = model or os.environ.get("THEME5_WHISPER_MODEL", "base.en")
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.download_root = download_root or os.environ.get("THEME5_MODEL_DIR")
        self.local_files_only = (os.environ.get("THEME5_OFFLINE") == "1") if local_files_only is None else local_files_only

    def _cache_key(self) -> tuple[Any, ...]:
        return (self.name, self.model_name, self.compute_type, self.cpu_threads, self.download_root)

    def _load(self) -> Any:
        from faster_whisper import WhisperModel  # heavy import stays off the import path
        m = WhisperModel(self.model_name, device="cpu", compute_type=self.compute_type,
                         cpu_threads=self.cpu_threads, download_root=self.download_root,
                         local_files_only=self.local_files_only)
        if _np is not None:  # one tiny decode so the first real request is warm
            list(m.transcribe(_np.zeros(8000, dtype=_np.float32), beam_size=1, without_timestamps=True)[0])
        return m

    def _decode(self, model: Any, x: Any, prompt: str | None) -> Transcript:
        segments, _info = model.transcribe(
            x, language="en" if self.model_name.endswith(".en") else None, beam_size=1, best_of=1,
            temperature=0.0, condition_on_previous_text=False, without_timestamps=True,
            vad_filter=False, initial_prompt=prompt or None,
        )
        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        if not segs or not text:
            return Transcript("", 0.0, self.name)
        tot = sum(max(s.end - s.start, 0.01) for s in segs)
        conf = sum(math.exp(min(0.0, s.avg_logprob)) * (1.0 - s.no_speech_prob) * max(s.end - s.start, 0.01)
                   for s in segs) / tot
        return Transcript(text, max(0.0, min(1.0, conf)), self.name)

    async def transcribe(self, media: MediaInput, audio: WavAudio | None, prompt: str | None) -> Transcript:
        if audio is None:
            raise BackendUnavailable("no decodable audio")
        model = await self._ensure()
        return await self._run(self._decode, model, resample_16k(audio), prompt)


class HostedTranscriber:
    """Stub for a hosted ASR model. Inject `client(wav_bytes, meta) -> {"text", "confidence"}`.
    ASSUMPTION: response keys. Without a client it is simply unavailable."""
    name = "hosted-asr"

    def __init__(self, client: Callable[[bytes, Mapping[str, Any]], Awaitable[Mapping[str, Any]]] | None = None) -> None:
        self.client = client

    async def warmup(self) -> None:
        return None

    async def transcribe(self, media: MediaInput, audio: WavAudio | None, prompt: str | None) -> Transcript:
        if self.client is None or media.data is None:
            raise BackendUnavailable("hosted ASR not configured")
        r = await self.client(media.data, {"ref": media.ref, "prompt": prompt})
        return Transcript(str(r.get("text", "")).strip(), float(r.get("confidence", 0.5)), self.name)


# ---- frame backends -------------------------------------------------------------


class HintFrameAnalyzer:
    """Uses labels/caption shipped with the frame (ASSUMPTION: the kit may provide them)."""
    name = "hint"

    async def warmup(self) -> None:
        return None

    async def analyze(self, media: MediaInput) -> FrameReading:
        if not media.hint_text:
            raise BackendUnavailable("no frame hint")
        labels = tuple(s.strip() for s in re.split(r"[,;\n]", media.hint_text) if s.strip())
        return FrameReading(text=media.hint_text, labels=labels, confidence=media.hint_confidence, source=self.name)


class RapidOcrAnalyzer(_ExecutorBackend):
    """Offline OCR (PP-OCR models on ONNX Runtime; weights ship inside the wheel,
    no download). Reads display codes, model labels and button text; also flags
    frames that are too dark or blank to read."""
    name = "rapidocr"

    def _cache_key(self) -> tuple[Any, ...]:
        return (self.name,)

    def _load(self) -> Any:
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR()

    def _read(self, engine: Any, data: bytes) -> FrameReading:
        quality = "ok"
        img = None
        try:
            import cv2  # dependency of rapidocr
            img = cv2.imdecode(_np.frombuffer(data, dtype=_np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            return FrameReading(source=self.name, quality="unreadable")
        gray = img.mean(axis=2)
        if float(gray.mean()) < 25.0:
            quality = "dark"
        elif float(gray.std()) < 4.0:
            quality = "blank"
        if quality != "ok":
            return FrameReading(source=self.name, quality=quality)
        result, _ = engine(img)
        lines = sorted(result or [], key=lambda r: (min(p[1] for p in r[0]), min(p[0] for p in r[0])))
        texts = [str(r[1]).strip() for r in lines if str(r[1]).strip()]
        if not texts:
            return FrameReading(source=self.name, confidence=0.0)
        weights = [max(1, len(str(r[1]))) for r in lines if str(r[1]).strip()]
        conf = sum(float(r[2]) * w for r, w in zip((r for r in lines if str(r[1]).strip()), weights)) / sum(weights)
        return FrameReading(text="\n".join(texts), confidence=conf, source=self.name)

    async def analyze(self, media: MediaInput) -> FrameReading:
        if media.data is None or (png_size(media.data) is None and media.data[:3] != b"\xff\xd8\xff"):
            raise BackendUnavailable("no decodable image")
        engine = await self._ensure()
        return await self._run(self._read, engine, media.data)


class HostedFrameAnalyzer:
    """Stub for a hosted vision model (VLM or cloud OCR). Inject
    `client(png_bytes, meta) -> {"caption"|"text", "labels", "confidence"}`.
    ASSUMPTION: response keys. The VLM output is still grounded by the rules
    below, so a chatty model cannot inject unperceived facts."""
    name = "hosted-vision"

    def __init__(self, client: Callable[[bytes, Mapping[str, Any]], Awaitable[Mapping[str, Any]]] | None = None) -> None:
        self.client = client

    async def warmup(self) -> None:
        return None

    async def analyze(self, media: MediaInput) -> FrameReading:
        if self.client is None or media.data is None:
            raise BackendUnavailable("hosted vision not configured")
        r = await self.client(media.data, {"ref": media.ref})
        labels = tuple(str(x) for x in (r.get("labels") or ()))
        text = str(r.get("text") or r.get("caption") or "")
        return FrameReading(text=text, labels=labels, confidence=float(r.get("confidence", 0.5)), source=self.name)


def default_transcribers() -> list[Transcriber]:
    return [HintTranscriber(), FasterWhisperTranscriber()]


def default_analyzers() -> list[FrameAnalyzer]:
    return [HintFrameAnalyzer(), RapidOcrAnalyzer()]


# ===========================================================================
# Grounding (pure, deterministic)
# ===========================================================================

_DEVICE_CANON: dict[str, str] = {
    "washing machine": "washing machine", "washer": "washing machine", "dryer": "dryer",
    "tumble dryer": "dryer", "refrigerator": "refrigerator", "fridge": "refrigerator",
    "tv": "tv", "television": "tv", "smart tv": "tv", "air conditioner": "air conditioner",
    "ac unit": "air conditioner", "oven": "oven", "microwave": "microwave", "dishwasher": "dishwasher",
    "router": "router", "printer": "printer", "phone": "phone", "smartphone": "phone", "laptop": "laptop",
    "monitor": "monitor", "soundbar": "soundbar", "vacuum": "vacuum", "robot vacuum": "vacuum",
    "watch": "watch", "smartwatch": "watch", "remote": "remote control", "remote control": "remote control",
    "air purifier": "air purifier", "induction cooktop": "cooktop", "cooktop": "cooktop",
    "projector": "projector", "tablet": "tablet", "earbuds": "earbuds", "car": "car",
}
_PARTS = (
    "lint filter", "drain filter", "drain pump", "water filter", "air filter", "filter", "door lock",
    "door seal", "door gasket", "door", "detergent drawer", "dispenser", "power button", "start button",
    "reset button", "power cord", "power supply", "control panel", "display", "screen", "knob", "dial",
    "hdmi port", "usb port", "ethernet port", "charging port", "port", "remote", "battery", "fan", "vent",
    "ice maker", "water inlet", "inlet hose", "drain hose", "hose", "compressor", "evaporator", "coil",
    "thermostat", "light bulb", "indicator light", "led", "tray", "drum", "agitator", "water tank",
    "dust bin", "brush", "cable", "antenna", "speaker", "keypad", "touch panel", "sensor", "heating element",
)
_BRANDS = ("samsung", "lg", "whirlpool", "bosch", "sony", "panasonic", "philips", "haier", "xiaomi",
           "apple", "hp", "canon", "epson", "tp-link", "netgear", "dyson", "ifb", "godrej", "voltas")
# ASSUMPTION: common letter-only appliance codes (Samsung/LG families) accepted without a digit.
_KNOWN_CODES = {"UE", "UB", "DE", "DC", "LE", "LC", "OE", "OF", "FE", "HE", "TE", "SD", "SUD", "NF",
                "PE", "CE", "AE", "BE", "HC", "FL", "OL", "IE", "DDC", "CL", "DOOR"}
_CODE_KW = re.compile(r"(?:error\s*code|fault\s*code|err(?:or)?|code|fault)\s*[:#=\-]?\s*([A-Z0-9]{1,4}(?:-[A-Z0-9]{1,2})?)\b", re.I)
_CODE_TOKEN = re.compile(r"^(?:[A-Z]{1,2}\d{1,3}[A-Z]?|\d{1,2}[A-Z]{1,2}\d?|[A-Z]{2,3}|[A-Z]{1,2}-[A-Z0-9]{1,3})$")
# ASSUMPTION: status messages appliances show instead of a code; read as the error code.
_DISPLAY_MSGS = ("no signal", "check filter", "clean filter", "door open", "low battery", "no water", "overheat")
_MODEL_KW = re.compile(r"\b(?:model|mdl|m/n|model\s*no\.?|model\s*number)\s*[:#.]?\s*([A-Z0-9][A-Z0-9/\-]{4,19})", re.I)
_INDICATOR = re.compile(
    r"\b((?:red|green|blue|orange|amber|yellow|white)\s+(?:light|led)|(?:blinking|flashing|flickering|solid)"
    r"(?:\s+(?:red|green|blue|orange|amber|yellow|white))?(?:\s+(?:light|led))?|(?:light|led)\s+(?:is\s+)?"
    r"(?:blinking|flashing|on|off))\b", re.I)
_NOT_MODEL = {"ERROR", "WARNING", "FILTER", "DRAIN", "RESET", "SAMSUNG"}


def _find_phrases(text: str, vocab: Iterable[str]) -> list[str]:
    """Vocabulary phrases in text order; longer phrases win over contained shorter ones."""
    low = f" {re.sub(r'[^a-z0-9]+', ' ', text.lower())} "
    found: list[tuple[int, str]] = []
    for v in sorted(vocab, key=len, reverse=True):
        key = f" {v} "
        i = low.find(key)
        if i >= 0:
            found.append((i, v))
            low = low.replace(key, " " + "|" * (len(key) - 2) + " ")  # keep offsets stable
    return [v for _, v in sorted(found)]


def _is_model(tok: str, labelled: bool = False) -> bool:
    """Model-number shape. A token printed after "Model" may be short (e.g. AR18B)."""
    t = tok.strip(".,:;()[]").upper()
    if not ((5 if labelled else 6) <= len(t) <= 20) or t in _NOT_MODEL or not re.fullmatch(r"[A-Z0-9][A-Z0-9/\-]+", t):
        return False
    return sum(c.isalpha() for c in t) >= 2 and sum(c.isdigit() for c in t) >= 2


def ground_frame(ref: str, readings: Sequence[FrameReading]) -> FrameFacts:
    """Merge backend readings and extract only what is literally present."""
    ok = [r for r in readings if r.quality == "ok"]
    quality = "ok" if ok or not readings else readings[0].quality
    text = "\n".join(t for t in (" ".join(filter(None, (r.text, *r.labels))) for r in ok) if t)
    conf = max((r.confidence for r in ok if (r.text or r.labels)), default=0.0)
    sources = "+".join(r.source for r in readings if r.source)
    if not text:
        return FrameFacts(ref=ref, quality=quality, confidence=0.0, source=sources)

    devices = []
    for d in _find_phrases(text, _DEVICE_CANON):
        c = _DEVICE_CANON[d]
        if c not in devices:
            devices.append(c)
    parts = [p for p in _find_phrases(text, _PARTS) if not (p in _DEVICE_CANON and _DEVICE_CANON[p] in devices)]
    brand = next(iter(_find_phrases(text, _BRANDS)), None)

    model = None
    mm = _MODEL_KW.search(text)
    if mm and _is_model(mm.group(1), labelled=True):
        model = mm.group(1).upper().strip(".,")
    else:  # OCR often drops the space in "MODEL AR18B" -> "MODELAR18B"
        toks = (re.sub(r"^(?:MODEL|MDL)[:#.]?", "", tok.strip(".,:;()[]").upper()) for tok in re.split(r"\s+", text))
        model = next((t for t in toks if _is_model(t)), None)
    code, explicit = None, False
    m = _CODE_KW.search(text)
    if m and (any(c.isdigit() for c in m.group(1)) or m.group(1).upper() in _KNOWN_CODES):
        code, explicit = m.group(1).upper(), True
    else:
        for tok in re.split(r"\s+", text):
            t = tok.strip(".,:;()[]!").upper()
            if t != model and _CODE_TOKEN.match(t) and (any(c.isdigit() for c in t) or t in _KNOWN_CODES) \
                    and t not in {"4K", "8K", "5G", "4G", "2G", "3G", "HD", "TV", "AC", "LED", "USB", "ON", "OFF"}:
                code = tok.strip(".,:;()[]!") if "-" in t else t  # keep "C-d0" as displayed
                break
        if code is None:
            msg = next(iter(_find_phrases(text, _DISPLAY_MSGS)), None)
            code = msg.upper() if msg else None

    if model and code and model == code:
        code = None

    ind = _INDICATOR.search(text)
    indicator = ind.group(1).lower() if ind else None
    if indicator and parts and parts[0] in ("led", "indicator light"):
        parts = parts[1:] or parts
    # Confidence: backend confidence, discounted for codes read without an "error" label.
    fconf = conf * (0.75 if code and not explicit else 1.0)
    return FrameFacts(
        ref=ref, device=devices[0] if len(devices) == 1 else None, devices=tuple(devices),
        part=parts[0] if parts else None, model=model, error_code=code, code_explicit=explicit,
        brand=brand, indicator=indicator, text=text, confidence=round(fconf, 3), quality=quality, source=sources,
    )


def frame_clarification(facts: FrameFacts, known_device: str | None = None) -> str | None:
    """One targeted question when the perception is ambiguous, else None."""
    if facts.quality == "dark":
        return "The picture is too dark for me to read. Could you turn on a light or move a little closer?"
    if facts.quality in ("blank", "unreadable"):
        return "I couldn't make anything out in that picture. Could you point the camera at the device again?"
    if len(facts.devices) > 1 and not (known_device and _canon_device(known_device) in facts.devices):
        a, b = facts.devices[0], facts.devices[1]
        return f"I can see a {a} and a {b}. Which one are you asking about?"
    if not facts.grounded:
        if known_device:
            return f"I can't make out which part of the {known_device} you mean. Could you point the camera at it?"
        return "I can't tell what that is from this angle. Which device is it, and which part are you asking about?"
    if facts.confidence < FRAME_MIN_CONFIDENCE:
        if facts.error_code:
            return f"Just to be sure, is the display showing {facts.error_code}?"
        if facts.model:
            return f"Is the model number {facts.model}?"
        what = facts.part or facts.device or facts.indicator
        return f"It looks like the {what}, but the picture isn't clear. Is that right?"
    return None


def _canon_device(d: str | None) -> str | None:
    if not d:
        return None
    return _DEVICE_CANON.get(d.strip().lower(), d.strip().lower())


def perceived_slots(facts: FrameFacts) -> dict[str, Any]:
    """Slots to merge into session state. Names follow planner.SLOT_SYNONYMS
    where one exists (device, error_code, frame); part/model/brand/indicator
    are new slot names (ASSUMPTION)."""
    out = {"frame": facts.ref, "device": facts.device, "part": facts.part, "model": facts.model,
           "error_code": facts.error_code, "brand": facts.brand, "indicator": facts.indicator}
    return {k: v for k, v in out.items() if v}


# ===========================================================================
# manual_lookup flow: frame facts -> tool args -> grounded answer
# ===========================================================================

_PARAM_FIELD = {  # exact normalised param name -> fact field
    "device": "device", "device_type": "device", "product": "device", "product_type": "device",
    "appliance": "device", "appliance_type": "device", "product_name": "device", "category": "device",
    "model": "model", "model_number": "model", "model_no": "model", "model_id": "model", "sku": "model",
    "device_model": "model", "product_model": "model", "product_id": "model", "part_number": "model",
    "part": "part", "part_name": "part", "component": "part", "section": "part", "feature": "part",
    "error_code": "error_code", "code": "error_code", "fault_code": "error_code", "error": "error_code",
    "display_code": "error_code", "frame": "frame", "frame_id": "frame", "image": "frame",
    "image_id": "frame", "image_ref": "frame", "photo": "frame", "picture": "frame", "frame_ref": "frame",
    "query": "query", "question": "query", "q": "query", "topic": "query", "issue": "query",
    "symptom": "query", "problem": "query", "search": "query", "keywords": "query", "text": "query",
    "brand": "brand", "manufacturer": "brand", "make": "brand", "indicator": "indicator", "light": "indicator",
}
_TOKEN_FIELD = {  # fallback: last matching token of the param name
    "model": "model", "sku": "model", "device": "device", "product": "device", "appliance": "device",
    "part": "part", "component": "part", "error": "error_code", "fault": "error_code", "code": "error_code",
    "frame": "frame", "image": "frame", "photo": "frame", "picture": "frame", "snapshot": "frame",
    "query": "query", "question": "query", "topic": "query", "issue": "query", "symptom": "query",
    "brand": "brand", "manufacturer": "brand", "indicator": "indicator", "light": "indicator", "led": "indicator",
}
_MANUAL_WORDS = ("manual", "lookup", "look_up", "guide", "troubleshoot", "documentation", "docs", "handbook",
                 "instructions", "support_article", "faq")


@dataclass
class ManualPlan:
    tool: ToolSpec
    args: dict[str, Any]
    missing: list[str]  # required params we could not ground
    mapping: dict[str, str]  # param -> fact field used

    @property
    def ready(self) -> bool:
        return not self.missing


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()).strip("_")


def param_field(param_name: str) -> str | None:
    n = _norm(param_name)
    if n in _PARAM_FIELD:
        return _PARAM_FIELD[n]
    for tok in reversed(n.split("_")):
        if tok in _TOKEN_FIELD:
            return _TOKEN_FIELD[tok]
    return None


def find_manual_tool(tools: Iterable[ToolSpec]) -> ToolSpec | None:
    """Best read-only tool that looks like a manual/troubleshooting lookup."""
    best, best_score = None, 0
    for t in tools:
        name, desc = _norm(t.name), t.description.lower()
        score = 2 * sum(w in name for w in _MANUAL_WORDS) + sum(w.replace("_", " ") in desc for w in _MANUAL_WORDS)
        score += sum(param_field(p.name) in ("frame", "model", "error_code", "part") for p in t.params)
        if t.state_modifying:
            score -= 3
        if score > best_score:
            best, best_score = t, score
    return best if best_score >= 2 else None


def default_query(facts: FrameFacts | None, user_query: str | None) -> str | None:
    if user_query and user_query.strip():
        return user_query.strip()
    if facts is None:
        return None
    if facts.error_code:
        return f"error {facts.error_code}"
    if facts.indicator:
        return facts.indicator
    return facts.part


def _coerce(value: Any, spec_type: str, enum: tuple[Any, ...] | None) -> Any:
    if value is None:
        return None
    if enum:
        v = str(value).lower()
        for e in enum:
            if str(e).lower() == v or _canon_device(str(e)) == _canon_device(v):
                return e
        return None
    t = spec_type.lower()
    if t in ("array", "list"):
        return [value]
    if t in ("integer", "number", "int", "float", "boolean", "bool", "object"):
        return None  # nothing perceived maps to these; leave for the planner
    return str(value)


def plan_manual_lookup(tools: Iterable[ToolSpec], facts: FrameFacts | None, user_query: str | None = None,
                       known: Mapping[str, Any] | None = None, tool: ToolSpec | None = None) -> ManualPlan | None:
    """Build validated args for a manual lookup from perceived facts, falling back
    to slots the user stated (known). Returns None if no suitable tool exists."""
    tool = tool or find_manual_tool(tools)
    if tool is None:
        return None
    known = dict(known or {})
    values: dict[str, Any] = {
        "device": (facts.device if facts else None) or _canon_device(known.get("device")),
        "model": (facts.model if facts else None) or known.get("model"),
        "part": (facts.part if facts else None) or known.get("part"),
        "error_code": (facts.error_code if facts else None) or known.get("error_code"),
        "frame": (facts.ref if facts else None) or known.get("frame"),
        "brand": (facts.brand if facts else None) or known.get("brand"),
        "indicator": (facts.indicator if facts else None) or known.get("indicator"),
        "query": default_query(facts, user_query or known.get("query")),
    }
    args: dict[str, Any] = {}
    mapping: dict[str, str] = {}
    missing: list[str] = []
    for p in tool.params:
        fld = param_field(p.name)
        v = _coerce(values.get(fld), p.type, p.enum) if fld else None
        if v is not None and v != "":
            args[p.name] = v
            mapping[p.name] = fld  # type: ignore[assignment]
        elif p.required:
            missing.append(p.name)
    return ManualPlan(tool=tool, args=args, missing=missing, mapping=mapping)


_MISSING_Q = {
    "device": "Which device is this?",
    "model": "Could you show me the label with the model number? It's usually on the back or inside the door.",
    "part": "Which part are you asking about? Could you point the camera at it?",
    "error_code": "What code is showing on the display?",
    "frame": "Could you show me the device on camera?",
    "query": "What would you like to know about it?",
    "brand": "Which brand is it?",
    "indicator": "Which light is on or blinking?",
}


def clarify_for_missing(plan: ManualPlan) -> str | None:
    if not plan.missing:
        return None
    p = plan.missing[0]
    fld = param_field(p)
    return _MISSING_Q.get(fld or "", f"What should I use for {p.replace('_', ' ')}?")


_TEXT_KEYS = ("answer", "text", "content", "instructions", "steps", "solution", "summary", "excerpt",
              "description", "section", "result", "message")


def manual_text(result: Any, _depth: int = 0) -> str:
    """Pull human-readable text out of an arbitrary manual_lookup result."""
    if result is None or _depth > 3:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, (list, tuple)):
        parts = [manual_text(x, _depth + 1) for x in result[:5]]
        parts = [p for p in parts if p]
        if parts and all(isinstance(x, str) for x in result[:5]):
            return "; ".join(parts)
        return parts[0] if parts else ""
    if isinstance(result, Mapping):
        for k in _TEXT_KEYS:
            if k in result:
                t = manual_text(result[k], _depth + 1)
                if t:
                    page = result.get("page")
                    return f"{t} (page {page})" if page is not None and _depth == 0 else t
        for v in result.values():
            if isinstance(v, (Mapping, list)):
                t = manual_text(v, _depth + 1)
                if t:
                    return t
    return ""


def _clip(s: str, limit: int = 320) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= limit:
        return s
    cut = s[:limit]
    end = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[:end + 1] if end > limit // 2 else cut.rstrip() + "...").strip()


def perceived_phrase(facts: FrameFacts | None, known_device: str | None = None) -> str:
    if facts is None:
        return ""
    dev = facts.device or _canon_device(known_device)
    on = f" on the {dev}" if dev else ""
    if facts.error_code:
        return f"I can see error {facts.error_code}{on}."
    if facts.indicator:
        return f"I can see a {facts.indicator}{on}."
    if facts.part and dev:
        return f"I can see the {facts.part} of the {dev}."
    if facts.model:
        return f"I can see model {facts.model}."
    if dev:
        return f"I can see the {dev}."
    return ""


def grounded_answer(facts: FrameFacts | None, result: Any, known_device: str | None = None) -> str:
    """Final answer built only from what was perceived plus the tool result."""
    seen = perceived_phrase(facts, known_device)
    body = manual_text(result)
    if not body:
        tail = "I couldn't find that in the manual."
        return f"{seen} {tail}".strip()
    return f"{seen} The manual says: {_clip(body)}".strip()


# ===========================================================================
# Processor: acknowledgment now, cancellable perception later
# ===========================================================================

_FRAME_ACKS = ("Let me look at that.", "Let me take a look.", "Okay, looking at it now.")
_AUDIO_ACKS = ("Got it, one moment.", "One sec, let me catch that.", "Okay, just a moment.")


class MultimodalProcessor:
    """Session-scoped. One instance per session; no module-level state.

    remaining_s: optional callable returning seconds left in the 120 s scenario
    budget (from the engine's watchdog). Jobs shrink their timeout to leave
    SCENARIO_RESERVE_S, and skip heavy backends when the budget is gone."""

    def __init__(self, transcribers: Sequence[Transcriber] | None = None,
                 analyzers: Sequence[FrameAnalyzer] | None = None, *,
                 audio_timeout_s: float = AUDIO_TIMEOUT_S, frame_timeout_s: float = FRAME_TIMEOUT_S,
                 remaining_s: Callable[[], float] | None = None, supersede_frames: bool = True) -> None:
        self.transcribers = list(default_transcribers() if transcribers is None else transcribers)
        self.analyzers = list(default_analyzers() if analyzers is None else analyzers)
        self.audio_timeout_s = audio_timeout_s
        self.frame_timeout_s = frame_timeout_s
        self.remaining_s = remaining_s
        self.supersede_frames = supersede_frames
        self.jobs: dict[str, PerceptionJob] = {}
        self.frames: dict[str, FrameFacts] = {}
        self.latest_frame: FrameFacts | None = None
        self.backend_status: dict[str, str] = {}
        self._ids = itertools.count(1)
        self._acks = {"frame": itertools.cycle(_FRAME_ACKS), "audio": itertools.cycle(_AUDIO_ACKS)}

    # ---- setup ---------------------------------------------------------------

    async def warmup(self, budget_s: float = WARMUP_CAP_S * 0.8) -> dict[str, str]:
        """Call from the 300 s setup hook. Never raises; records per-backend status."""
        backends = [*self.transcribers, *self.analyzers]

        async def one(b: Any) -> None:
            try:
                await b.warmup()
                self.backend_status[b.name] = "ready"
            except asyncio.CancelledError:
                self.backend_status[b.name] = "timeout"
                raise
            except Exception as e:
                self.backend_status[b.name] = f"unavailable: {str(e)[:120]}"

        tasks = [asyncio.ensure_future(one(b)) for b in backends]
        done, pending = await asyncio.wait(tasks, timeout=budget_s) if tasks else (set(), set())
        for t in pending:
            t.cancel()
        for b in backends:
            self.backend_status.setdefault(b.name, "timeout")
        return dict(self.backend_status)

    def close(self) -> None:
        self.cancel_all()
        for b in [*self.transcribers, *self.analyzers]:
            if hasattr(b, "close"):
                b.close()

    # ---- entry points (synchronous: call and emit job.ack with no await in between) ----

    def on_audio(self, ev: Event | Mapping[str, Any], prompt: str | None = None) -> PerceptionJob:
        media = load_media(ev, "audio")
        jid = f"mm_{next(self._ids):04d}"
        task = asyncio.ensure_future(self._audio(jid, media, prompt))
        return self._register(jid, "audio", media.ref, next(self._acks["audio"]), SPEAK_FILLER, task)

    def on_frame(self, ev: Event | Mapping[str, Any], query: str | None = None,
                 known: Mapping[str, Any] | None = None) -> PerceptionJob:
        media = load_media(ev, "frame")
        if self.supersede_frames:
            self.cancel_all("frame")
        jid = f"mm_{next(self._ids):04d}"
        task = asyncio.ensure_future(self._frame(jid, media, query, dict(known or {})))
        return self._register(jid, "frame", media.ref, next(self._acks["frame"]), SPEAK_ACK, task)

    def cancel(self, job_id: str) -> bool:
        j = self.jobs.pop(job_id, None)
        return bool(j and j.cancel())

    def cancel_all(self, kind: str | None = None) -> list[str]:
        ids = [j.id for j in self.jobs.values() if not j.done and (kind is None or j.kind == kind)]
        for i in ids:
            self.cancel(i)
        return ids

    # ---- internals -----------------------------------------------------------------

    def _register(self, jid: str, kind: str, ref: str, ack: str, ack_kind: str, task: asyncio.Future) -> PerceptionJob:
        loop = asyncio.get_running_loop()
        job = PerceptionJob(jid, kind, ref, ack, ack_kind, task, loop.time())  # type: ignore[arg-type]
        self.jobs[jid] = job
        task.add_done_callback(lambda _t, j=jid: self.jobs.pop(j, None))
        return job

    def _budget(self, base: float) -> float:
        if self.remaining_s is None:
            return base
        return max(0.0, min(base, self.remaining_s() - SCENARIO_RESERVE_S))

    async def _chain(self, backends: Sequence[Any], call: Callable[[Any], Awaitable[Any]], budget: float,
                     good: Callable[[Any], bool], notes: list[str], collect_all: bool = False) -> tuple[list[Any], bool]:
        """Try backends in order within a shared deadline. Returns (results, degraded)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + budget
        results: list[Any] = []
        degraded = False
        for b in backends:
            left = deadline - loop.time()
            if left <= 0:
                notes.append(f"{b.name}: skipped (budget)")
                degraded = True
                break
            try:
                r = await asyncio.wait_for(call(b), timeout=left)
            except BackendUnavailable as e:
                notes.append(f"{b.name}: unavailable ({str(e)[:80]})")
                continue
            except asyncio.TimeoutError:
                notes.append(f"{b.name}: timeout")
                degraded = True
                break
            except asyncio.CancelledError:
                raise
            except Exception as e:  # a backend bug must not kill the session
                notes.append(f"{b.name}: error {e.__class__.__name__}")
                degraded = True
                continue
            results.append(r)
            if good(r) and not collect_all:
                break
        return results, degraded

    async def _audio(self, jid: str, media: MediaInput, prompt: str | None) -> PerceptionOutcome:
        notes: list[str] = []
        audio: WavAudio | None = None
        if media.data is not None:
            try:
                audio = decode_wav(media.data)
            except (ValueError, struct.error) as e:
                notes.append(f"decode: {e}")
        elif media.error:
            notes.append(media.error)
        if audio is not None and audio.rms_dbfs < SILENCE_DBFS and not media.hint_text:
            return PerceptionOutcome(jid, "audio", media.ref, "clarify", confidence=0.0, source="energy",
                                     clarify="I didn't hear anything there. Could you say that again?",
                                     notes=(f"silence {audio.rms_dbfs:.1f} dBFS",))
        budget = self._budget(self.audio_timeout_s)
        results, degraded = await self._chain(
            self.transcribers, lambda b: b.transcribe(media, audio, prompt), budget,
            lambda r: bool(r.text) and r.confidence >= ASR_MIN_CONF, notes)
        heard = [r for r in results if r.text]
        best = max(heard, key=lambda r: r.confidence) if heard else None
        if best is None:
            return PerceptionOutcome(jid, "audio", media.ref, "clarify", source="none", degraded=True,
                                     clarify="Sorry, I couldn't make that out. Could you say it again?",
                                     notes=tuple(notes))
        if best.confidence < ASR_MIN_CONF:
            return PerceptionOutcome(jid, "audio", media.ref, "clarify", text=best.text, confidence=best.confidence,
                                     clarify=f"Sorry, I didn't catch that clearly. Did you say \"{_clip(best.text, 120)}\"?",
                                     source=best.source, degraded=degraded, notes=tuple(notes))
        return PerceptionOutcome(jid, "audio", media.ref, "ok", text=best.text, confidence=best.confidence,
                                 source=best.source, degraded=degraded, notes=tuple(notes))

    async def _frame(self, jid: str, media: MediaInput, query: str | None, known: dict[str, Any]) -> PerceptionOutcome:
        notes: list[str] = [media.error] if media.error else []
        budget = self._budget(self.frame_timeout_s)
        readings, degraded = await self._chain(
            self.analyzers, lambda b: b.analyze(media), budget, lambda r: False, notes, collect_all=True)
        facts = ground_frame(media.ref, readings)
        if not readings:
            facts = replace(facts, quality="unreadable")
        kd = _canon_device(known.get("device"))
        if facts.device is None and kd and kd in facts.devices:  # the user already said which one
            facts = replace(facts, device=kd)
        self.frames[media.ref] = facts
        self.latest_frame = facts
        question = frame_clarification(facts, known.get("device"))
        slots = perceived_slots(facts) if question is None else {"frame": facts.ref}
        return PerceptionOutcome(jid, "frame", media.ref, "clarify" if question else "ok",
                                 confidence=facts.confidence, clarify=question, facts=facts, slots=slots,
                                 source=facts.source, degraded=degraded, notes=tuple(notes))


# ===========================================================================
# Drop-in adapter for plugins.Perception (agent.py): Agent(perception=MultimodalPerception())
# ===========================================================================


def describe_facts(facts: FrameFacts) -> str:
    """Compact note for SessionState.frame_notes, e.g. 'washing machine; error 4C; model WW90T534DAW'."""
    bits = [facts.device or " or ".join(facts.devices) or None,
            f"part {facts.part}" if facts.part else None,
            f"error {facts.error_code}" if facts.error_code else None,
            f"model {facts.model}" if facts.model else None,
            facts.indicator, f"brand {facts.brand}" if facts.brand else None]
    return "; ".join(b for b in bits if b)


class MultimodalPerception:
    """Implements plugins.Perception (transcribe/describe -> (text, confidence))
    and the optional setup() warm-up hook, backed by MultimodalProcessor.
    Ambiguous perceptions come back with confidence 0 so the agent clarifies;
    the targeted question is in `last_clarify`, full facts in `last_facts`."""

    def __init__(self, processor: MultimodalProcessor | None = None) -> None:
        self.mm = processor or MultimodalProcessor()
        self.last_clarify: str | None = None
        self.last_facts: FrameFacts | None = None

    async def setup(self) -> None:
        await self.mm.warmup()

    async def transcribe(self, ev: Event) -> tuple[str | None, float]:
        out = await self.mm.on_audio(ev).task
        self.last_clarify = out.clarify
        if out.status == "ok":
            return out.text, out.confidence
        return (out.text or None), min(out.confidence, max(0.0, ASR_MIN_CONF - 0.01)) if out.text else 0.0

    async def describe(self, ev: Event) -> tuple[str | None, float]:
        out = await self.mm.on_frame(ev).task
        self.last_clarify = out.clarify
        self.last_facts = out.facts
        note = describe_facts(out.facts) if out.facts else ""
        if out.status == "ok":
            return note or None, out.confidence
        return note or "unclear frame", 0.0


# ===========================================================================
# Agent default: kit hints first, real ASR/OCR when the kit sends raw media only
# ===========================================================================


def perception_mode() -> str:
    """THEME5_PERCEPTION=hints restores the old hints-only perception; anything else is hybrid."""
    return "hints" if os.environ.get(PERCEPTION_ENV, "").strip().lower() in ("hints", "payload") else "hybrid"


class HybridPerception:
    """Default Agent perception (plugins.Perception + setup + facts).

    A media event that carries a transcript or labels/caption (ASSUMPTION: the
    kit may ship them, SPEC U19/U20) is read exactly like PayloadPerception.
    A raw clip or frame goes to MultimodalPerception (faster-whisper, RapidOCR).
    With no usable backend the audio path returns (None, 0) and the agent asks
    the user to repeat; frames degrade to "unclear frame" and a clarification.
    Backend budgets follow the scenario clock via `remaining_s`."""

    def __init__(self, processor: MultimodalProcessor | None = None,
                 remaining_s: Callable[[], float] | None = None) -> None:
        from .plugins import PayloadPerception  # plugins stays import-light; no cycle at module load
        self.payload = PayloadPerception()
        self.media = MultimodalPerception(processor or MultimodalProcessor(remaining_s=remaining_s))
        self._facts: dict[str, dict[str, Any]] = {}  # media ref -> slots perceived from raw pixels
        self.backend_status: dict[str, str] = {}

    async def setup(self) -> None:
        self.backend_status = await self.media.mm.warmup()

    @staticmethod
    def _has_transcript(ev: Event) -> bool:
        text, _ = audio_transcript(ev)
        return bool(text and text.strip())

    @staticmethod
    def _has_labels(ev: Event) -> bool:
        cap = frame_caption(ev)
        return bool(frame_labels(ev) or (cap and cap.strip()))

    async def transcribe(self, ev: Event) -> tuple[str | None, float]:
        if self._has_transcript(ev):
            return await self.payload.transcribe(ev)
        return await self.media.transcribe(ev)

    async def describe(self, ev: Event) -> tuple[str | None, float]:
        if self._has_labels(ev):
            return await self.payload.describe(ev)
        note, conf = await self.media.describe(ev)
        facts = self.media.last_facts
        if facts is not None and conf > 0.0:
            slots = {"device_model": facts.model, "error_code": facts.error_code}
            self._facts[media_ref(ev) or f"frame@{ev.t:g}"] = {k: v for k, v in slots.items() if v}
        return note, conf

    def facts(self, ev: Event) -> dict[str, Any]:
        if self._has_labels(ev):
            return self.payload.facts(ev)
        return dict(self._facts.get(media_ref(ev) or f"frame@{ev.t:g}", {}))

    def close(self) -> None:
        self.media.mm.close()
