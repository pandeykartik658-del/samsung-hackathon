# /mnt/project-files/theme5/theme5/plugins.py
"""Optional slow-path plug-ins. The agent never awaits these on the fast path;
each has a rule/payload-based default so the agent runs with zero extra deps."""
from __future__ import annotations

import asyncio
from typing import Any, Protocol

from .protocol import LLM_TIMEOUT_S, Event, ToolSpec, audio_transcript, frame_caption, frame_labels


class LLMPlugin(Protocol):
    async def parse(self, utterance: str, snapshot: dict[str, Any], tools: list[ToolSpec]) -> dict[str, Any] | None:
        """Return {"intent": str|None, "slots": {...}} or None if unsure."""


class NullLLM:
    async def parse(self, utterance: str, snapshot: dict[str, Any], tools: list[ToolSpec]) -> dict[str, Any] | None:
        return None


async def llm_parse(llm: LLMPlugin | None, utterance: str, snapshot: dict[str, Any], tools: list[ToolSpec],
                    timeout: float = LLM_TIMEOUT_S) -> dict[str, Any] | None:
    """Bounded call: any timeout, exception or malformed output falls back to None."""
    if llm is None:
        return None
    try:
        out = await asyncio.wait_for(llm.parse(utterance, snapshot, tools), timeout)
    except Exception:  # noqa: BLE001 - plug-in failures must never break the session
        return None
    if not isinstance(out, dict) or not isinstance(out.get("slots", {}), dict):
        return None
    return out


class Perception(Protocol):
    async def transcribe(self, ev: Event) -> tuple[str | None, float]:
        """(text or None, confidence 0..1)"""

    async def describe(self, ev: Event) -> tuple[str | None, float]:
        """(caption or None, confidence 0..1)"""


class PayloadPerception:
    """Default: trust transcripts/captions the harness ships with the media.
    ASSUMPTION: the kit supplies them; otherwise plug in real ASR/vision here."""

    async def transcribe(self, ev: Event) -> tuple[str | None, float]:
        return audio_transcript(ev)

    AMBIGUITY_MARGIN = 0.15  # ASSUMPTION: top-2 labels this close => ambiguous

    async def describe(self, ev: Event) -> tuple[str | None, float]:
        cap = frame_caption(ev)
        labels = sorted((d for d in frame_labels(ev) if d["confidence"] is not None), key=lambda d: -d["confidence"])
        if labels:
            conf = labels[0]["confidence"]
            if len(labels) > 1 and conf - labels[1]["confidence"] < self.AMBIGUITY_MARGIN:
                conf = min(conf, 0.3)
        else:
            conf = float(ev.payload.get("confidence", 1.0 if cap else 0.0))
        if ev.payload.get("ambiguous"):
            conf = min(conf, 0.3)
        return cap, conf

    def facts(self, ev: Event) -> dict[str, Any]:
        """Slots readable straight off a confident frame (model number, display code)."""
        labels = sorted(frame_labels(ev), key=lambda d: -(d["confidence"] if d["confidence"] is not None else 1.0))
        if not labels:
            return {}
        top = labels[0]
        out: dict[str, Any] = {}
        if top["model"]:
            out["device_model"] = str(top["model"])
        text = next((d["text"] for d in labels if d["text"]), None)
        if text:
            out["error_code"] = str(text).strip()  # keep the display's own case ("C-d0")
        return out
