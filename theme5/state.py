# /mnt/project-files/theme5/theme5/state.py
"""Session-scoped state: intent, slots, correction log. No cross-session memory."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .protocol import snapshot_payload


@dataclass
class SessionState:
    intent: str | None = None
    slots: dict[str, Any] = field(default_factory=dict)
    version: int = 0  # bumps on every change; used to detect stale work
    corrections: list[tuple[str, Any, Any]] = field(default_factory=list)
    frames: list[str] = field(default_factory=list)  # frame refs, newest last
    frame_notes: dict[str, str] = field(default_factory=dict)
    completed: set[str] = field(default_factory=set)  # intents finished this session

    def set_intent(self, intent: str | None) -> bool:
        if intent == self.intent:
            return False
        self.intent = intent
        self.version += 1
        return True

    def update(self, changes: dict[str, Any], correction: bool = False) -> set[str]:
        """Localised update: only the named slots change. Returns changed names."""
        changed: set[str] = set()
        for k, v in changes.items():
            old = self.slots.get(k)
            if old == v:
                continue
            if v is None:
                self.slots.pop(k, None)
            else:
                self.slots[k] = v
            changed.add(k)
            if correction and old is not None:
                self.corrections.append((k, old, v))
        if changed:
            self.version += 1
        return changed

    def add_frame(self, ref: str, note: str | None = None) -> None:
        self.frames.append(ref)
        if note:
            self.frame_notes[ref] = note
        self.update({"frame": ref})

    def reset_task(self) -> None:
        """User abandoned the task: drop intent and task slots."""
        self.intent = None
        self.slots.clear()
        self.version += 1

    def snapshot(self) -> dict[str, Any]:
        return snapshot_payload(self.intent, dict(self.slots))
