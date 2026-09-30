# /mnt/project-files/theme5/tests/test_plugins.py
import asyncio

from theme5.plugins import NullLLM, PayloadPerception, llm_parse
from theme5.protocol import parse_event


class SlowLLM:
    async def parse(self, utterance, snapshot, tools):
        await asyncio.sleep(1)
        return {"intent": "x", "slots": {}}


class BrokenLLM:
    async def parse(self, utterance, snapshot, tools):
        raise RuntimeError("boom")


class GoodLLM:
    async def parse(self, utterance, snapshot, tools):
        return {"intent": "get_weather", "slots": {"location": "Goa"}}


def test_llm_parse_bounded_and_fallbacks():
    run = asyncio.run
    assert run(llm_parse(None, "hi", {}, [])) is None
    assert run(llm_parse(NullLLM(), "hi", {}, [])) is None
    assert run(llm_parse(SlowLLM(), "hi", {}, [], timeout=0.01)) is None
    assert run(llm_parse(BrokenLLM(), "hi", {}, [])) is None
    assert run(llm_parse(GoodLLM(), "hi", {}, []))["slots"] == {"location": "Goa"}


def test_payload_perception():
    p = PayloadPerception()
    a = parse_event({"type": "audio", "path": "x.wav", "transcript": "hello", "confidence": 0.4})
    assert asyncio.run(p.transcribe(a)) == ("hello", 0.4)
    assert asyncio.run(p.transcribe(parse_event({"type": "audio", "path": "x.wav"}))) == (None, 1.0)
    f = parse_event({"type": "frame", "frame_id": "f", "labels": ["tv", "remote"]})
    assert asyncio.run(p.describe(f)) == ("tv, remote", 1.0)
    amb = parse_event({"type": "frame", "frame_id": "f", "caption": "?", "ambiguous": True})
    assert asyncio.run(p.describe(amb))[1] <= 0.3
    close = parse_event({"type": "frame", "labels": [{"label": "washing_machine", "confidence": 0.49},
                                                     {"label": "dryer", "confidence": 0.47}]})
    assert asyncio.run(p.describe(close))[1] <= 0.3
    sure = parse_event({"type": "frame", "labels": [{"label": "display", "model": "WW90T", "text": "4C", "confidence": 0.93}]})
    assert asyncio.run(p.describe(sure))[1] == 0.93
    assert p.facts(sure) == {"device_model": "WW90T", "error_code": "4C"}
