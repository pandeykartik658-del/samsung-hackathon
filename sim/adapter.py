# sim/adapter.py
"""Bridge between the harness wire format (dicts) and an agent implementation.

Agent contract (guide 3: "two asynchronous queues"):
    agent.run(inbox: asyncio.Queue, outbox: asyncio.Queue) -> coroutine
The harness puts events on `inbox` and reads actions from `outbox`.

`load_agent("pkg.mod:attr")` accepts, for `attr`:
  - a class or zero-arg factory returning an object with async `run(inbox, outbox)`
  - an async function `fn(inbox, outbox)`

Codec (ASSUMPTION until the agent's protocol.py is final): by default events are
passed as plain dicts and actions are accepted as dicts, dataclasses, or objects
with `to_dict()` / `model_dump()`. A codec module exposing
`decode_event(dict) -> obj` and/or `encode_action(obj) -> dict` can be given with
`--codec pkg.mod`; if not given, `<agent package>.protocol` is tried automatically.
"""
from __future__ import annotations

import dataclasses
import importlib
import inspect
from typing import Any, Callable, Dict, Optional


class Codec:
    def __init__(self, decode_event: Optional[Callable[[Dict[str, Any]], Any]] = None,
                 encode_action: Optional[Callable[[Any], Dict[str, Any]]] = None):
        self._decode = decode_event
        self._encode = encode_action

    def event_out(self, ev: Dict[str, Any]) -> Any:
        return self._decode(ev) if self._decode else ev

    def action_in(self, obj: Any) -> Any:
        if self._encode is not None and not isinstance(obj, dict):
            return self._encode(obj)
        return to_plain(obj)


def to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return obj
    for meth in ("to_dict", "model_dump", "dict"):
        f = getattr(obj, meth, None)
        if callable(f):
            return f()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return obj  # left as-is; the harness logs it as an invalid action


def load_codec(spec: Optional[str], agent_spec: Optional[str] = None) -> Codec:
    mod = None
    if spec:
        mod = importlib.import_module(spec)
    elif agent_spec:
        pkg = agent_spec.split(":")[0].split(".")[0]
        try:
            mod = importlib.import_module(f"{pkg}.protocol")
        except ImportError:
            mod = None
    if mod is None:
        return Codec()
    return Codec(getattr(mod, "decode_event", None), getattr(mod, "encode_action", None))


class _FnAgent:
    def __init__(self, fn):
        self._fn = fn

    async def run(self, inbox, outbox):
        await self._fn(inbox, outbox)


def accepts_kw(fn: Any, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def resolve(obj: Any, **kw: Any) -> Any:
    """Turn a class / factory / coroutine function into an object with run().
    Keyword args (e.g. clock) are passed only if the constructor accepts them."""
    if inspect.iscoroutinefunction(obj):
        return _FnAgent(obj)
    if inspect.isclass(obj) or callable(obj) and not hasattr(obj, "run"):
        inst = obj(**{k: v for k, v in kw.items() if accepts_kw(obj, k)})
        if inspect.iscoroutinefunction(getattr(inst, "run", None)):
            return inst
        raise TypeError(f"{obj!r} did not produce an object with async run(inbox, outbox)")
    if inspect.iscoroutinefunction(getattr(obj, "run", None)):
        return obj
    raise TypeError(f"cannot use {obj!r} as an agent")


DEFAULT_AGENT = "theme5.agent:Agent"


def load_agent_factory(spec: str = DEFAULT_AGENT) -> Callable[..., Any]:
    """Return a factory producing a fresh agent per scenario (session-scoped state).
    The harness calls it as factory(clock=<fn returning virtual ms>)."""
    mod_name, _, attr = spec.partition(":")
    mod = importlib.import_module(mod_name)
    target = getattr(mod, attr or "make_agent")
    return lambda **kw: resolve(target, **kw)
