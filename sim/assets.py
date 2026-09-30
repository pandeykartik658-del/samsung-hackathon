# sim/assets.py
"""Deterministic synthetic media for scenarios (stdlib only).

Scans scenario JSON files for audio_clip / video_frame events and writes any
missing WAV / PNG they reference. The media are placeholders: WAVs contain
syllable-like tone bursts of the right duration (no real speech) and PNGs draw
one box per oracle label. Real perception must come from the official kit's
media; our scenarios carry oracle `transcript` / `labels` for offline testing.

    python -m sim.assets scenarios/ [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import wave
import zlib
from pathlib import Path
from typing import List, Tuple

SAMPLE_RATE = 16000


def _seed(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


def write_wav(path: Path, duration_ms: int, seed_key: str, sample_rate: int = SAMPLE_RATE) -> None:
    n = int(sample_rate * duration_ms / 1000)
    seed = _seed(seed_key)
    frames = bytearray()
    syll = int(sample_rate * 0.18)  # ~180 ms "syllables" with short gaps
    for i in range(n):
        k = i // syll
        pos = (i % syll) / syll
        f = 140 + (seed >> (k % 64)) % 120  # pitch per syllable, 140-260 Hz
        env = math.sin(math.pi * pos) if pos < 0.85 else 0.0
        v = 0.3 * env * (math.sin(2 * math.pi * f * i / sample_rate)
                         + 0.4 * math.sin(2 * math.pi * 2.7 * f * i / sample_rate))
        frames += struct.pack("<h", int(max(-1.0, min(1.0, v)) * 32767))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(frames))


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path: Path, width: int, height: int, boxes: List[Tuple[int, int, int, int, Tuple[int, int, int]]]) -> None:
    bg = (228, 228, 222)
    rows = []
    for y in range(height):
        row = bytearray(b"\x00")
        for x in range(width):
            c = bg
            for (x0, y0, x1, y1, col) in boxes:
                if x0 <= x < x1 and y0 <= y < y1:
                    c = col
            row += bytes(c)
        rows.append(bytes(row))
    raw = b"".join(rows)
    png = (b"\x89PNG\r\n\x1a\n"
           + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + _png_chunk(b"IDAT", zlib.compress(raw, 9))
           + _png_chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def frame_boxes(width: int, height: int, n: int, key: str):
    seed = _seed(key)
    boxes = []
    for i in range(max(1, n)):
        w = width // (n + 1)
        x0 = (i * width) // max(1, n) + width // 20
        col = (60 + (seed >> (8 * i)) % 120, 70 + (seed >> (8 * i + 3)) % 120, 80 + (seed >> (8 * i + 5)) % 120)
        boxes.append((x0, height // 5, min(width, x0 + w), height - height // 6, col))
    return boxes


def generate(scenario_dir: Path, force: bool = False) -> List[Path]:
    written = []
    files = [scenario_dir] if scenario_dir.is_file() else sorted(scenario_dir.glob("*.json"))
    for f in files:
        base = f.parent
        for ev in json.loads(f.read_text()).get("events", []):
            if "path" not in ev:
                continue
            out = base / ev["path"]
            if out.exists() and not force:
                continue
            if ev["type"] == "audio_clip":
                write_wav(out, int(ev["duration_ms"]), ev["path"], int(ev.get("sample_rate", SAMPLE_RATE)))
            elif ev["type"] == "video_frame":
                w, h = int(ev.get("width", 160)), int(ev.get("height", 120))
                write_png(out, w, h, frame_boxes(w, h, len(ev.get("labels", [])), ev["path"]))
            else:
                continue
            written.append(out)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=Path)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    for p in generate(a.path, a.force):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
