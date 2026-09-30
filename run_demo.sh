#!/usr/bin/env bash
# run_demo.sh
# End-to-end demo on the local virtual-clock harness:
#   1. the 9 public-style scenarios (pass/fail checks + rubric estimate)
#   2. the 60-scenario adversarial bench (rubric table + 5 worst failures)
#   3. a copy of the trace viewer with one real trace embedded, for the walkthrough
# Usage: bash run_demo.sh [scenario id for the viewer, default s02_text_destination_change]
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
fi
TRACE_ID="${1:-s02_text_destination_change}"
OUT=runs/demo

"$PY" - <<'PYCHECK'
import sys
if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
    sys.exit(f"Python 3.10-3.12 required, found {sys.version.split()[0]}")
PYCHECK

echo "== 1/3  public-style scenarios (sim harness, virtual clock)"
"$PY" -m sim.run scenarios --report --out "$OUT/scenarios"

echo
echo "== 2/3  60-scenario adversarial bench (rubric replica, multimodal x1.5)"
"$PY" -m bench.run_suite bench/scenarios60 --agent theme5.agent:Agent --out "$OUT/bench" \
  | grep -E '^(MEAN|SUITE|Suite)' || true
echo "full table and worst failures: $OUT/bench/REPORT.md"

echo
echo "== 3/3  trace viewer"
TRACE="$OUT/scenarios/traces/$TRACE_ID.jsonl"
"$PY" - "$TRACE" "$OUT/trace_viewer.html" <<'PYVIEW'
import sys
from pathlib import Path
sys.path.insert(0, "viewer")
import embed_sample
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
html = Path("viewer/trace_viewer.html").read_text(encoding="utf-8")
dst.write_text(embed_sample.embed(html, src.read_text(encoding="utf-8")), encoding="utf-8")
print(f"open {dst} in a browser (trace: {src.name}; any other trace loads via the file picker)")
PYVIEW
