# tests/bench/test_asr_probe.py
from __future__ import annotations

import json

from bench import asr_probe
from sim.assets import write_wav
from theme5 import multimodal as mm


class FakeASR:
    def __init__(self):
        self.closed = False

    async def warmup(self):
        return None

    async def transcribe(self, media, audio, prompt):
        return mm.Transcript("make it kochi", 0.4, "fake")

    def close(self):
        self.closed = True


def test_probe_prints_heard_vs_oracle(tmp_path, monkeypatch, capsys):
    write_wav(tmp_path / "assets/c1.wav", 400, "k")
    (tmp_path / "s.json").write_text(json.dumps({"events": [
        {"type": "audio_clip", "clip_id": "c1", "path": "assets/c1.wav", "transcript": "Make it Kochi."}]}))
    monkeypatch.setattr(asr_probe, "FasterWhisperTranscriber", FakeASR)
    assert asr_probe.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Make it Kochi." in out and "make it kochi" in out and "1 clips, 1 below" in out
