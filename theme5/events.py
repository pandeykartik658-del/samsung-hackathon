# /mnt/project-files/theme5/theme5/events.py
"""Typed events and actions covering every input and output in guide §3.1.

protocol.py turns raw kit JSON into `protocol.Event`; `typed()` lifts that
into the dataclasses below. Actions are built as dataclasses and serialised
by the engine (protocol.make_action + to_wire), so no module hand-writes
wire dicts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

from . import protocol as P
from .protocol import ToolResult, ToolSpec

# ============================================================== inputs (§3.1)


@dataclass(frozen=True)
class ToolManifest:
    """Scenario tool manifest (read-only vs state-modifying tools)."""
    t: float
    tools: tuple[ToolSpec, ...]
    reference_date: str | None = None


@dataclass(frozen=True)
class TextChunk:
    """Transcribed text chunk; `end_of_turn` marks the end of the user turn."""
    t: float
    text: str
    end_of_turn: bool = False
    cumulative: bool | None = None


@dataclass(frozen=True)
class EndOfTurn:
    """Standalone end-of-turn marker (U07)."""
    t: float


@dataclass(frozen=True)
class AudioClip:
    """Raw WAV clip; transcript/confidence only if the kit ships them (U19)."""
    t: float
    ref: str | None
    transcript: str | None = None
    confidence: float = 1.0
    end_of_turn: bool = True


@dataclass(frozen=True)
class VideoFrame:
    """PNG frame; caption/labels only if the kit ships them (U20)."""
    t: float
    ref: str | None
    caption: str | None = None
    labels: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class Interruption:
    """Barge-in signal, optionally carrying the new words (U08)."""
    t: float
    text: str = ""


@dataclass(frozen=True)
class ToolResultEvent:
    """Asynchronous tool result or fault for a call_id."""
    t: float
    result: ToolResult


@dataclass(frozen=True)
class SessionEnd:
    t: float


@dataclass(frozen=True)
class UnknownEvent:
    t: float
    raw_type: str


InputEvent = Union[ToolManifest, TextChunk, EndOfTurn, AudioClip, VideoFrame, Interruption,
                   ToolResultEvent, SessionEnd, UnknownEvent]


def typed(ev: P.Event) -> InputEvent:
    """protocol.Event -> typed input event."""
    if ev.type == P.EV_MANIFEST:
        ref = ev.payload.get("date") or ev.payload.get("today")
        return ToolManifest(ev.t, tuple(P.parse_manifest(ev)), None if ref is None else str(ref))
    if ev.type == P.EV_TEXT:
        return TextChunk(ev.t, P.text_of(ev), P.is_end_of_turn(ev), P.is_cumulative(ev))
    if ev.type == P.EV_EOT:
        return EndOfTurn(ev.t)
    if ev.type == P.EV_AUDIO:
        txt, conf = P.audio_transcript(ev)
        eot = P.is_end_of_turn(ev) or "end_of_turn" not in ev.payload
        return AudioClip(ev.t, P.media_ref(ev), txt, conf, eot)
    if ev.type == P.EV_FRAME:
        return VideoFrame(ev.t, P.media_ref(ev), P.frame_caption(ev), tuple(P.frame_labels(ev)))
    if ev.type == P.EV_INTERRUPT:
        return Interruption(ev.t, P.text_of(ev))
    if ev.type == P.EV_TOOL_RESULT:
        return ToolResultEvent(ev.t, P.parse_tool_result(ev))
    if ev.type == P.EV_END:
        return SessionEnd(ev.t)
    return UnknownEvent(ev.t, ev.raw_type)


# ============================================================= outputs (§3.1)


@dataclass(frozen=True)
class Speak:
    """Spoken output. kind: filler | ack | progress | info (U21)."""
    text: str
    kind: str = P.SPEAK_ACK


@dataclass(frozen=True)
class ToolCall:
    """Non-blocking tool call with explicit call_id and plan generation."""
    call_id: str
    tool: str
    args: dict[str, Any]
    generation: int
    idempotency_key: str | None = None


@dataclass(frozen=True)
class Cancel:
    """Cancellation of an in-flight call."""
    call_id: str
    reason: str = "superseded"


@dataclass(frozen=True)
class Clarify:
    """Clarification request, optionally naming the slot it asks for."""
    text: str
    slot: str | None = None


@dataclass(frozen=True)
class FinalResponse:
    """Final response; the engine guarantees a valid State Snapshot."""
    text: str
    snapshot: dict[str, Any] | None = None


Action = Union[Speak, ToolCall, Cancel, Clarify, FinalResponse]


@dataclass(frozen=True)
class StateSnapshot:
    intent: str | None
    slots: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return P.snapshot_payload(self.intent, self.slots)


def action_fields(a: Action) -> tuple[str, dict[str, Any]]:
    """(internal action type, fields) for protocol.make_action."""
    if isinstance(a, Speak):
        return P.ACT_SPEAK, {"text": a.text, "kind": a.kind}
    if isinstance(a, ToolCall):
        f: dict[str, Any] = {"call_id": a.call_id, "tool": a.tool, "args": dict(a.args)}
        if P.EMIT_CALL_META:
            f["generation"] = a.generation
            if a.idempotency_key:
                f["idempotency_key"] = a.idempotency_key
        return P.ACT_TOOL_CALL, f
    if isinstance(a, Cancel):
        return P.ACT_CANCEL, {"call_id": a.call_id, "reason": a.reason}
    if isinstance(a, Clarify):
        return P.ACT_CLARIFY, {"text": a.text, "slot": a.slot}
    if isinstance(a, FinalResponse):
        return P.ACT_FINAL, {"text": a.text}
    raise TypeError(f"not an action: {a!r}")
