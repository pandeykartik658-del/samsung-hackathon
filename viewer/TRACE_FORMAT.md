# /mnt/project-files/theme5/viewer/TRACE_FORMAT.md
# Trace format the viewer reads

No trace writer exists in `sim/` yet, so everything here is an ASSUMPTION. The viewer
accepts the spellings of both `sim/wire.py` (harness) and `theme5/protocol.py` (agent),
so it should load the harness's traces unchanged once they exist. If the harness settles
on something else, only `normalizeRow()` / `EVENT_KIND` / `ACTION_KIND` in
`trace_viewer.html` need to change.

## File
- JSONL, one JSON object per line. Also accepted: a JSON array, or `{"trace": [...]}`.
- Bad lines are skipped and listed under "Trace checks"; the viewer never stops on them.

## Record (preferred shape, used by `sample_trace.jsonl`)
```json
{"ts_ms": 1030, "dir": "out", "action": {"type": "tool_call", "call_id": "c1", "tool": "search_flights", "args": {}}}
{"ts_ms": 3400, "dir": "in",  "event":  {"type": "tool_result", "call_id": "c1", "status": "ok", "result": {}}}
{"ts_ms": 0,    "dir": "meta", "meta":  {"scenario_id": "..."}}
```
| Field | Accepted | Notes |
|---|---|---|
| time | `ts_ms`, `t_ms`, `time_ms`, `t`, `ts`, `timestamp` (ms); `t_s`, `ts_s`, `time_s` (s) | Wrapper time wins over body time. Missing: previous record's time, flagged. |
| direction | `dir` / `direction` / `stream` / `source`: `in`/`event`/`harness`/`user`, `out`/`action`/`agent`, `meta`, `sys` | `sys` = harness bookkeeping (tool_started, write_committed, scenario_start/end...): listed in the log, not drawn; `scenario_start` data is the scenario header. Missing: inferred from the type. |
| body | `event`, `action`, `data`, `payload`, `meta`, or the record itself (bare wire event/action) | |
| type | `type` (then `event`, `action`, wrapper `type` / `kind`) | Aliases below. Unknown types go to the System lane and are flagged. |

## Types
- Events: `tool_manifest`/`manifest`/`tools`, `text_chunk`/`text`/`transcript`, `audio_clip`/`audio`
  (`duration_ms` draws a bar, `transcript` shown if present), `video_frame`/`frame`/`image`
  (`labels` shown if present), `interrupt`/`barge_in`, `tool_result`/`tool_response`, `session_end`/`end_session`.
- Actions: `speak` (`kind`: filler draws lighter), `tool_call` (`call_id`, `tool`, `args`, optional `retry_of`),
  `cancel` (`call_id`), `clarify`, `final`/`final_response`, standalone `snapshot`.
- Snapshot: `state_snapshot` or `snapshot` on any action, `{"intent": str|null, "slots": {}}`.
- Tool mode from the manifest: `read_only: true` or a kind containing "read" is read-only;
  anything else counts as state-modifying (same rule as the spec).

## Derived by the viewer
- **Stale result**: a `tool_result` whose call was cancelled at or before it arrived, `status: "cancelled"`, or `stale: true`.
- **Trace checks**: bad JSON, missing/out-of-order timestamps, unknown types, results for unknown call ids,
  duplicate results, calls never resolved or cancelled, finals without a snapshot, and more than one successful
  call of the same state-modifying tool (possible double booking).
