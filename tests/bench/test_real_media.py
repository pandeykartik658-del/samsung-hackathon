# tests/bench/test_real_media.py
from __future__ import annotations

import importlib.util
import json
import wave

import pytest

from bench import real_media as rm

HAS_PIL = importlib.util.find_spec("PIL") is not None


def _scenario(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    doc = {"id": "x1", "events": [
        {"id": "e1", "t_ms": 0, "type": "audio_clip", "clip_id": "c1", "path": "assets/c1.wav",
         "sample_rate": 16000, "duration_ms": 900, "end_of_turn": True, "transcript": "Make it Kochi."},
        {"id": "e2", "t_ms": 10, "type": "video_frame", "frame_id": "f1", "path": "assets/f1.png",
         "width": 160, "height": 120, "labels": [{"label": "dryer", "model": "DV90T", "confidence": 0.92, "text": "HE"}]},
        {"id": "e3", "t_ms": 20, "type": "text_chunk", "text": "hi"},
    ]}
    (src / "x1.json").write_text(json.dumps(doc))
    return src


def test_convert_keeps_oracles_and_writes_every_asset(tmp_path, monkeypatch):
    monkeypatch.setattr(rm, "espeak_bin", lambda: None)  # no TTS: audio stays a placeholder
    dst = tmp_path / "dst"
    stats = rm.convert(_scenario(tmp_path), dst)
    assert stats["audio_placeholder"] == 1 and stats["audio_tts"] == 0
    doc = json.loads((dst / "x1.json").read_text())
    assert doc["events"][0]["transcript"] == "Make it Kochi."  # scorer may read oracles; harness strips them
    assert doc["events"][1]["labels"][0]["model"] == "DV90T"
    assert (dst / "assets/c1.wav").exists() and (dst / "assets/f1.png").exists()


@pytest.mark.skipif(not HAS_PIL, reason="Pillow not installed")
def test_frames_are_rendered_large_with_text(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setattr(rm, "espeak_bin", lambda: None)
    dst = tmp_path / "dst"
    assert rm.convert(_scenario(tmp_path), dst)["frames"] == 1
    assert Image.open(dst / "assets/f1.png").size == (rm.W, rm.H)
    ev = json.loads((dst / "x1.json").read_text())["events"][1]
    assert (ev["width"], ev["height"]) == (rm.W, rm.H)


@pytest.mark.skipif(rm.espeak_bin() is None, reason="espeak-ng not installed")
def test_tts_audio_is_16k_mono_and_duration_updated(tmp_path):  # pragma: no cover - needs espeak-ng
    dst = tmp_path / "dst"
    assert rm.convert(_scenario(tmp_path), dst)["audio_tts"] == 1
    ev = json.loads((dst / "x1.json").read_text())["events"][0]
    with wave.open(str(dst / "assets/c1.wav")) as w:
        assert (w.getframerate(), w.getnchannels()) == (16000, 1)
        assert abs(ev["duration_ms"] - 1000 * w.getnframes() / 16000) < 2
