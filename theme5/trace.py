# /mnt/project-files/theme5/theme5/trace.py
"""Agent-side trace. The kit scores its own trace (G§5); ours records why we
did things (dropped stale results, blocked duplicates, floor changes) so the
viewer and our tests can see decisions the wire does not show."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TraceLog(list):
    """A list of dict records (actions and `_`-prefixed internal notes)."""

    def note(self, kind: str, t: float, **data: Any) -> dict[str, Any]:
        rec = {"type": f"_{kind}", "t": round(float(t), 3), **data}
        self.append(rec)
        return rec

    def of(self, kind: str) -> list[dict[str, Any]]:
        key = kind if kind.startswith("_") else f"_{kind}"
        return [r for r in self if r.get("type") == key]

    def actions(self) -> list[dict[str, Any]]:
        return [r for r in self if not str(r.get("type", "")).startswith("_")]

    def dump(self, path: str | Path) -> None:
        with Path(path).open("w") as f:
            for r in self:
                f.write(json.dumps(r, default=str) + "\n")
