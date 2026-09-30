# Requests from the sim/scenarios thread to other threads

Found while running `python -m sim.run scenarios/ --report` on 2026-09-29.

## Agent package (theme5/)
1. `scenarios/01_text_reroute.json` .. `05_unseen_tool.json` were written into `scenarios/`, which this
   thread owns, in a different format. `sim.run` skips them with a warning. Please move them to e.g.
   `theme5/fixtures/` (or port them to the sim format).
2. `theme5/harness.py` + `theme5/scorer.py` duplicate `sim/`. Suggest keeping them as unit-test fixtures only
   and using `python -m sim.run` for end-to-end runs.
3. Tool-call field names changed mid-session (`tool`/`args` -> `name`/`arguments`). The harness accepts both,
   but please pin one in protocol.py.
4. Behaviour the canonical scenarios catch (re-run 18:05, agent now passes 8/9):
   - s09: `manual_lookup` is never called with the error code after the user answers
     "The washer. It's showing 4C." (needs device_model WW90T and a query containing 4C).
   Fixed since the first run: s01, s03, s06, s07, s08.
   Convention the scenarios assume (ASSUMPTION): snapshot slot names = the tool's parameter names.

## bench/scorer.py
All three requests done by the scorer thread (18:00). sim.run now takes its rubric from bench.scorer.

## viewer/trace_viewer.html
1. Harness traces use `dir: "sys"` for bookkeeping records (tool_started, tool_completed, tool_cancelled,
   write_committed, cancel_ignored, scenario_start/end); `bench/scorer.py` relies on that. Please add
   `"sys"` to `DIR_META` (or a System lane) so they are not flagged as unknown. Everything else in
   TRACE_FORMAT.md already matches (`t_ms`, `dir` in/out, body in `data`, `read_only` on manifest tools,
   `status: "cancelled"` results).
