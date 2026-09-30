# /mnt/project-files/theme5/theme5/protocol_media.py
"""Media-event field access and multimodal tunables.

Companion to protocol.py (owned by the scaffold thread). Everything here is an
ASSUMPTION about the unreleased evaluation kit (SPEC U19, U20) and is meant to
be folded into protocol.py by its owner; multimodal.py imports only from here
and from protocol.py, never raw kit field names.
"""
from __future__ import annotations

import base64
import binascii
import os
import struct
from dataclasses import dataclass
from typing import Any, Mapping

from .protocol import ASR_MIN_CONFIDENCE, Event, audio_transcript, frame_caption, media_ref

# ---------------------------------------------------------------------------
# Tunables. ASSUMPTION: chosen for the 120 s scenario cap, not given by the guide.
# ---------------------------------------------------------------------------
AUDIO_TIMEOUT_S = 15.0  # budget for one ASR job (all backends together)
FRAME_TIMEOUT_S = 12.0  # budget for one frame job
SCENARIO_RESERVE_S = 8.0  # never start perception with less than this left
MAX_AUDIO_S = 30.0  # longer clips are truncated before ASR
SILENCE_DBFS = -50.0  # RMS below this is treated as "heard nothing"
FRAME_MIN_CONFIDENCE = ASR_MIN_CONFIDENCE  # below this a frame reading is ambiguous (agent.py uses the same bar)
ASR_MIN_CONF = ASR_MIN_CONFIDENCE  # re-export; the single source is protocol.py
MEDIA_ROOT_ENV = "THEME5_MEDIA_ROOT"  # base dir for relative media paths
PERCEPTION_ENV = "THEME5_PERCEPTION"  # "hints" = trust kit transcripts/labels only (old default)

# ASSUMPTION: inline media may arrive base64-encoded under one of these keys.
_B64_KEYS = ("data", "b64", "base64", "bytes", "content", "audio_base64", "wav_base64",
             "image_base64", "png_base64", "frame_base64")
_PATH_KEYS = ("path", "file", "uri", "url")
_BASE_DIR_KEYS = ("base_dir", "root", "assets_dir")
_SR_KEYS = ("sample_rate", "sr", "rate")


@dataclass(frozen=True)
class MediaInput:
    kind: str  # "audio" | "frame"
    ref: str  # stable reference (id or path) used in snapshots and tool args
    data: bytes | None  # raw file bytes when resolvable
    hint_text: str | None  # harness-provided transcript / caption, if any
    hint_confidence: float
    sample_rate: int | None = None
    error: str | None = None  # why data is None


def _payload(ev_or_payload: Event | Mapping[str, Any]) -> Mapping[str, Any]:
    return ev_or_payload.payload if isinstance(ev_or_payload, Event) else ev_or_payload


def _decode_b64(v: Any) -> bytes | None:
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if not isinstance(v, str) or len(v) < 8:
        return None
    s = v.split(",", 1)[1] if v.startswith("data:") and "," in v else v
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        return None


def _resolve_path(p: Mapping[str, Any], raw: str) -> str:
    if raw.startswith("file://"):
        raw = raw[7:]
    if os.path.isabs(raw):
        return raw
    for k in _BASE_DIR_KEYS:
        if isinstance(p.get(k), str):
            return os.path.join(p[k], raw)
    root = os.environ.get(MEDIA_ROOT_ENV)
    return os.path.join(root, raw) if root else raw


def load_media(ev_or_payload: Event | Mapping[str, Any], kind: str) -> MediaInput:
    """Never raises. Returns bytes when found (inline or on disk) plus any hint."""
    p = _payload(ev_or_payload)
    ev = ev_or_payload if isinstance(ev_or_payload, Event) else Event(type=kind, t=0.0, payload=dict(p))
    ref = media_ref(ev) or "unknown"
    if kind == "audio":
        hint, conf = audio_transcript(ev)
    else:
        hint, conf = frame_caption(ev), 1.0
    sr = next((int(p[k]) for k in _SR_KEYS if isinstance(p.get(k), (int, float))), None)
    data: bytes | None = None
    err: str | None = None
    for k in _B64_KEYS:
        if k in p:
            data = _decode_b64(p[k])
            if data:
                break
    if data is None:
        raw = next((p[k] for k in _PATH_KEYS if isinstance(p.get(k), str)), None)
        if raw and not raw.startswith(("http://", "https://")):
            path = _resolve_path(p, raw)
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except OSError as e:
                err = f"unreadable {path}: {e.__class__.__name__}"
        else:
            err = "no inline data or local path"
    return MediaInput(kind=kind, ref=ref, data=data, hint_text=hint, hint_confidence=conf,
                      sample_rate=sr, error=err)


def png_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from the IHDR chunk; None if not a PNG."""
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return int(w), int(h)
