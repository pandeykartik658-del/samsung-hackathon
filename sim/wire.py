# sim/wire.py
"""Harness-side wire format for events (harness -> agent) and actions (agent -> harness).

Guide section 3.1 names the stream contents but not the JSON field names, so
every field name below is an ASSUMPTION. They are kept in this one module so the
harness can be re-pointed at the official kit (or at the agent's protocol.py)
by editing only this file and sim/adapter.py.

Aligned with theme5/protocol.py (parse_event / parse_tool_result / parse_manifest).

Events (dicts, all carry `type`, `event_id`, `t` = harness virtual ms):
  tool_manifest  tools: [tool_def], date: reference date, session: {...}
  text_chunk     text, end_of_turn
  audio_clip     clip_id, path, sample_rate, duration_ms, end_of_turn, [transcript]
  video_frame    frame_id, path, width, height, [labels]
  interrupt      reason
  tool_result    call_id, tool, ok, status: ok|error|timeout|cancelled, result | error, retryable
  session_end    -

  [transcript] / [labels] are oracle annotations for offline testing only
  (ASSUMPTION: the real kit ships raw WAV/PNG). `--strip-oracle` removes them.

Actions (dicts, all carry `type`; agents may add `action_id`, `t`, `state_snapshot`):
  speak           text, kind: filler|ack|progress|info|answer
  tool_call       call_id, tool, args
  cancel          call_id
  clarify         text, [slot]
  final_response  text, state_snapshot: {intent: str|null, slots: {}}
The harness normalises aliases (final_response -> final, state_snapshot ->
snapshot, tool_call name/arguments -> tool/args) before validation; traces keep the normalised form under `kind`
and every original field in `data`.
"""
from __future__ import annotations

from typing import Any, Dict, List

EVENT_TYPES = (
    "tool_manifest", "text_chunk", "audio_clip", "video_frame",
    "interrupt", "tool_result", "session_end",
)
ACTION_TYPES = ("speak", "tool_call", "cancel", "clarify", "final")
ACTION_ALIASES = {"final_response": "final", "say": "speak", "clarification": "clarify", "cancellation": "cancel"}
SPEAK_KINDS = ("filler", "ack", "progress", "info", "answer")
TOOL_STATUSES = ("ok", "error", "timeout", "cancelled")
ORACLE_FIELDS = ("transcript", "labels")

# User-input events after which response latency is measured (guide 5: "following
# user input or interruption"). ASSUMPTION: only end-of-turn chunks count.
def is_user_turn_boundary(ev: Dict[str, Any]) -> bool:
    t = ev.get("type")
    if t == "interrupt":
        return True
    if t in ("text_chunk", "audio_clip"):
        return bool(ev.get("end_of_turn"))
    return False


def is_substantive(action: Dict[str, Any]) -> bool:
    """ASSUMPTION: fillers are not 'substantive spoken actions' (guide 5)."""
    t = action.get("type")
    if t in ("clarify", "final"):
        return True
    return t == "speak" and action.get("kind", "answer") != "filler"


def _is_str(v: Any) -> bool:
    return isinstance(v, str) and v != ""


def validate_snapshot(snap: Any) -> List[str]:
    errs: List[str] = []
    if not isinstance(snap, dict):
        return ["snapshot must be an object"]
    if "intent" not in snap:
        errs.append("snapshot.intent missing")
    elif snap["intent"] is not None and not isinstance(snap["intent"], str):
        errs.append("snapshot.intent must be string or null")
    if not isinstance(snap.get("slots"), dict):
        errs.append("snapshot.slots must be an object")
    return errs


def validate_action(a: Any) -> List[str]:
    """Return a list of schema errors (empty when valid)."""
    if not isinstance(a, dict):
        return ["action must be a JSON object"]
    t = a.get("type")
    if t not in ACTION_TYPES:
        return [f"unknown action type {t!r}"]
    errs: List[str] = []
    if t == "speak":
        if not _is_str(a.get("text")):
            errs.append("speak.text must be a non-empty string")
        if a.get("kind", "answer") not in SPEAK_KINDS:
            errs.append(f"speak.kind must be one of {SPEAK_KINDS}")
    elif t == "tool_call":
        if not _is_str(a.get("call_id")):
            errs.append("tool_call.call_id must be a non-empty string")
        if not _is_str(a.get("tool")):
            errs.append("tool_call.tool must be a non-empty string")
        if not isinstance(a.get("args", {}), dict):
            errs.append("tool_call.args must be an object")
    elif t == "cancel":
        if not _is_str(a.get("call_id")):
            errs.append("cancel.call_id must be a non-empty string")
    elif t == "clarify":
        if not _is_str(a.get("text")):
            errs.append("clarify.text must be a non-empty string")
    elif t == "final":
        if not isinstance(a.get("text", ""), str):
            errs.append("final.text must be a string")
        errs.extend(validate_snapshot(a.get("snapshot")))
    return errs


def normalize_action(a: Any) -> Any:
    """Map protocol.py names onto the harness's canonical names, keeping all fields."""
    if not isinstance(a, dict):
        return a
    out = dict(a)
    t = out.get("type")
    out["type"] = ACTION_ALIASES.get(t, t)
    if "snapshot" not in out and "state_snapshot" in out:
        out["snapshot"] = out["state_snapshot"]
    if out["type"] == "tool_call":
        if "tool" not in out and "name" in out:
            out["tool"] = out["name"]
        if "args" not in out and "arguments" in out:
            out["args"] = out["arguments"]
    return out


def strip_oracle(ev: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in ev.items() if k not in ORACLE_FIELDS}
