# /mnt/project-files/theme5/theme5/cli.py
"""CLI.
  python -m theme5 run [scenarios] [sim.run options]   sim/ replay scored by bench/scorer.py
  python -m theme5 serve                       JSONL events on stdin -> JSONL actions on stdout
ASSUMPTION: the official kit will import Agent directly or talk JSONL; both are covered."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .agent import Agent
from .protocol import dumps


async def _serve() -> None:
    loop = asyncio.get_running_loop()
    in_q: asyncio.Queue = asyncio.Queue()
    out_q: asyncio.Queue = asyncio.Queue()
    agent = Agent()
    run = loop.create_task(agent.run(in_q, out_q))

    async def pump_out() -> None:
        while True:
            a = await out_q.get()
            sys.stdout.write(dumps(a) + "\n")
            sys.stdout.flush()

    out_task = loop.create_task(pump_out())
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        line = line.strip()
        if line:
            try:
                await in_q.put(json.loads(line))
            except json.JSONDecodeError:
                continue
    await in_q.put(None)
    await run
    while not out_q.empty():
        await asyncio.sleep(0)
    out_task.cancel()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["run"]:
        from sim.run import main as sim_main  # dev-only: sim/ ships alongside the package
        rest = argv[1:] or ["scenarios"]
        if "--agent" not in rest:
            rest += ["--agent", "theme5.agent:Agent"]
        return sim_main(rest)
    ap = argparse.ArgumentParser(prog="theme5")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="replay scenarios through sim/ (options passed to sim.run)")
    sub.add_parser("serve", help="JSONL events on stdin, JSONL actions on stdout")
    ap.parse_args(argv)
    asyncio.run(_serve())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
