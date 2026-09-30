# bench/real_media.py
"""Rebuild a scenario directory with perceivable media, for raw-media benchmarking.

The generated suites ship placeholder media (sim/assets.py: tone bursts and
coloured boxes) plus oracle transcripts/labels, so `--strip-oracle` alone
measures nothing about ASR/OCR: no perception could read a placeholder. This
tool copies a scenario dir and replaces the media:

- audio: the oracle transcript spoken by espeak-ng (16 kHz mono WAV); the event's
  duration_ms is updated. Without espeak-ng on PATH the placeholder is kept and
  the clip is reported as "placeholder" (audio results are then meaningless).
- frames: a 640x480 appliance panel per oracle label, with the device name as a
  printed badge, a "MODEL <model>" sticker and the display code in an LCD box
  (needs Pillow). This is the text an OCR model can read on a real appliance; it
  cannot show what a vision model would get from the appliance's shape.

Oracle fields stay in the JSON (the scorer may read them); run the suite with
`--strip-oracle` so the agent never sees them:

    python -m bench.real_media bench/scenarios60 /tmp/real60
    python -m bench.run_suite /tmp/real60 --agent theme5.agent:Agent --strip-oracle --out runs/real60
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import wave
from array import array
from pathlib import Path
from typing import Any, Dict, List, Optional

from sim.assets import generate as generate_placeholders

W, H = 640, 480
PANEL_BG = [(214, 216, 212), (196, 204, 214)]


def espeak_bin() -> Optional[str]:
    return shutil.which("espeak-ng") or shutil.which("espeak")


def speak(text: str, out: Path, engine: str, rate_wpm: int = 160) -> int:
    """Synthesise `text` to a 16 kHz mono WAV at `out`; returns its duration in ms."""
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw.wav"
        subprocess.run([engine, "-v", "en-us", "-s", str(rate_wpm), "-w", str(raw), text],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with wave.open(str(raw), "rb") as r:
            sr, ch, sw, frames = r.getframerate(), r.getnchannels(), r.getsampwidth(), r.readframes(r.getnframes())
    if ch != 1 or sw != 2:
        raise ValueError(f"unexpected espeak output: {ch} ch, {sw * 8} bit")
    samples = array("h", frames)
    n_out = int(len(samples) * 16000 / sr)
    out_samples = array("h", (samples[min(len(samples) - 1, int(i * sr / 16000))] for i in range(n_out)))
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(out_samples.tobytes())
    return int(1000 * n_out / 16000)


def _font(size: int) -> Any:
    from PIL import ImageFont
    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def render_frame(labels: List[Dict[str, Any]], out: Path) -> None:
    """One panel per label: device badge, model sticker, LCD display code."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H), (236, 236, 230))
    d = ImageDraw.Draw(img)
    n = max(1, len(labels))
    pw = W // n
    for i, lab in enumerate(labels or [{}]):
        x0, x1 = i * pw + 12, (i + 1) * pw - 12
        d.rectangle([x0, 20, x1, H - 20], fill=PANEL_BG[i % 2], outline=(90, 90, 90), width=3)
        name = str(lab.get("label") or "").replace("_", " ").upper()
        size = 34 if n == 1 else 24
        if name:
            d.text((x0 + 18, 50), name, fill=(20, 20, 20), font=_font(size))
        if lab.get("model"):
            d.rectangle([x0 + 14, 120, x1 - 14, 175], fill=(250, 250, 250), outline=(60, 60, 60), width=2)
            d.text((x0 + 24, 132), f"MODEL {lab['model']}", fill=(10, 10, 10), font=_font(size - 6))
        if lab.get("text"):
            d.rectangle([x0 + 14, 230, x1 - 14, 330], fill=(18, 30, 22))
            d.text((x0 + 34, 250), str(lab["text"]), fill=(120, 255, 150), font=_font(52 if n == 1 else 40))
    conf = min((float(lab.get("confidence", 1.0)) for lab in labels), default=1.0)
    if conf < 0.6:  # the scenario marks this frame as unclear: blur it the way a shaky camera would
        from PIL import ImageFilter
        img = img.filter(ImageFilter.GaussianBlur(radius=4 + 10 * (0.6 - conf)))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def convert(src: Path, dst: Path) -> Dict[str, int]:
    stats = {"audio_tts": 0, "audio_placeholder": 0, "frames": 0, "frames_placeholder": 0}
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    engine = espeak_bin()
    try:
        import PIL  # noqa: F401
        have_pil = True
    except ImportError:
        have_pil = False
    for f in sorted(src.glob("*.json")):
        doc = json.loads(f.read_text())
        for ev in doc.get("events", []):
            if "path" not in ev:
                continue
            out = dst / ev["path"]
            if ev.get("type") == "audio_clip" and ev.get("transcript") and engine:
                ev["duration_ms"] = speak(str(ev["transcript"]), out, engine)
                ev["sample_rate"] = 16000
                stats["audio_tts"] += 1
            elif ev.get("type") == "video_frame" and have_pil:
                render_frame(list(ev.get("labels") or []), out)
                ev["width"], ev["height"] = W, H
                stats["frames"] += 1
            elif ev.get("type") in ("audio_clip", "video_frame"):
                stats["audio_placeholder" if ev["type"] == "audio_clip" else "frames_placeholder"] += 1
        (dst / f.name).write_text(json.dumps(doc, indent=1))
    generate_placeholders(dst)  # anything not re-rendered keeps a placeholder
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    a = ap.parse_args(argv)
    stats = convert(a.src, a.dst)
    print(json.dumps(stats))
    if stats["audio_placeholder"]:
        print("WARNING: espeak-ng not found; audio clips are placeholders", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
