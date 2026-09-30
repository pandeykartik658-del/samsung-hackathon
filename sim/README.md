# sim/: Theme 05 simulation harness

Mirrors guide section 4: virtual-clock streaming harness, deterministic replay,
async mock tools with latency and fault injection, full JSONL trace.

```
python -m sim.run scenarios/ --report                 # team agent (theme5.agent:Agent)
python -m sim.run scenarios/ --agent pkg.mod:Factory  # any agent with async run(inbox, outbox)
python -m sim.run scenarios/ --only s07 -v --report   # one scenario, every check
python -m sim.assets scenarios/                       # (re)generate placeholder WAV/PNG
python -m pytest tests/sim -q
```
Outputs go to `runs/latest/`: `traces/<id>.jsonl`, `report.json`, `report.md`.
Exit code 1 if any scenario fails its `expected` checks.

| File | Role |
|---|---|
| vloop.py | asyncio loop whose clock jumps to the next timer: agent sleeps/timeouts run on virtual time. `--charge-compute` adds real CPU time for latency realism. 120 s wall cap. |
| harness.py | replays events, serves tool calls, honours cancels, ends on quiescence or `duration_ms`. |
| mock_tools.py | flight_search, book_flight, create_ticket, manual_lookup + unseen tools from the scenario; fixtures; faults `error` / `timeout` / `slow` by call index or args; writes commit only on completion. |
| wire.py | event/action field names (ASSUMPTION, aligned with theme5/protocol.py; aliases accepted). |
| adapter.py | loads the agent, passes `clock=` (virtual ms) if the constructor accepts it. |
| scenario.py | scenario schema and validation. |
| scoring.py | `expected` block checks (pass/fail) + rubric estimate. `bench/scorer.py` is reported alongside when present. |
| trace.py | `{seq, t_ms, dir: in/out/sys, kind, data}` JSONL. |

ASSUMPTIONS (official kit unreleased): all wire field names; a cancelled call is acked with a
`tool_result` of status `cancelled`; audio/frame events carry oracle `transcript`/`labels`
(`--strip-oracle` removes them; the WAV/PNG files are placeholders, not speech or photos);
latency is measured to the first non-filler speak/clarify/final; cancel grace defaults to 300 ms.
