# /mnt/project-files/theme5/viewer/embed_sample.py
"""Embed a JSONL trace into trace_viewer.html as its built-in sample.

    python viewer/embed_sample.py                      # re-embed viewer/sample_trace.jsonl
    python viewer/embed_sample.py path/to/trace.jsonl  # ship the viewer with a real run baked in

The viewer stays a single self-contained file: the trace lives in a
<script type="application/x-ndjson" id="sample-trace"> block.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HTML = HERE / "trace_viewer.html"
SAMPLE = HERE / "sample_trace.jsonl"
_BLOCK = re.compile(r'(<script type="application/x-ndjson" id="sample-trace">\n)(.*?)(\n</script>)', re.S)


def embed(html: str, jsonl: str) -> str:
    lines = [ln for ln in jsonl.splitlines() if ln.strip()]
    for i, ln in enumerate(lines, 1):
        json.loads(ln)  # refuse to embed a broken trace
    body = "\n".join(lines).replace("</", "<\\/")  # keep a stray </script> from closing the block
    new, n = _BLOCK.subn(lambda m: m.group(1) + body + m.group(3), html, count=1)
    if n != 1:
        raise ValueError("sample-trace block not found in viewer HTML")
    return new


def embedded(html: str) -> str:
    m = _BLOCK.search(html)
    if not m:
        raise ValueError("sample-trace block not found in viewer HTML")
    return m.group(2).replace("<\\/", "</")


def main(argv: list[str]) -> int:
    src = Path(argv[1]) if len(argv) > 1 else SAMPLE
    HTML.write_text(embed(HTML.read_text(encoding="utf-8"), src.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"embedded {src} into {HTML}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
