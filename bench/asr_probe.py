# bench/asr_probe.py
"""Print what the default ASR hears for every audio clip in a scenario dir, next to the oracle.

    python -m bench.asr_probe /tmp/real60 [--limit 20]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

from theme5.multimodal import FasterWhisperTranscriber, asr_prompt, decode_wav
from theme5.protocol_media import ASR_MIN_CONF, MediaInput


async def probe(root: Path, limit: int) -> int:
    asr = FasterWhisperTranscriber()
    await asr.warmup()
    n = low = 0
    for f in sorted(root.glob("*.json")):
        for ev in json.loads(f.read_text()).get("events", []):
            if ev.get("type") != "audio_clip" or n >= limit:
                continue
            data = (root / ev["path"]).read_bytes()
            media = MediaInput(kind="audio", ref=ev.get("clip_id", ""), data=data, hint_text=None,
                               hint_confidence=0.0)
            t = await asr.transcribe(media, decode_wav(data), asr_prompt())
            n += 1
            low += t.confidence < ASR_MIN_CONF
            print(f"{t.confidence:.2f}  oracle={ev.get('transcript')!r}\n      heard ={t.text!r}")
    print(f"{n} clips, {low} below ASR_MIN_CONF={ASR_MIN_CONF}")
    asr.close()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=Path)
    ap.add_argument("--limit", type=int, default=1000)
    a = ap.parse_args(argv)
    return asyncio.run(probe(a.path, a.limit))


if __name__ == "__main__":
    sys.exit(main())
