# /mnt/project-files/theme5/viewer/test_viewer.py
"""Tests for the trace viewer.

The embed/sample checks are pure Python. The browser checks drive the real HTML
in headless Chromium through Playwright and are skipped when it is unavailable.
Run: python -m pytest viewer/test_viewer.py -q
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import importlib.util

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("embed_sample", HERE / "embed_sample.py")
embed_sample = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(embed_sample)

HTML = HERE / "trace_viewer.html"
SAMPLE = HERE / "sample_trace.jsonl"


# ---------------------------------------------------------------- pure Python
def _lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip()]


def test_embedded_sample_matches_file():
    assert _lines(embed_sample.embedded(HTML.read_text())) == _lines(SAMPLE.read_text())


def test_frame_labels_render_objects_not_object_object():
    """Frame labels may be {label, model, confidence} objects (scenario s09); the summary must format them."""
    html = HTML.read_text()
    assert "function frameLabel(" in html
    assert "b.labels.map(frameLabel)" in html
    assert "b.labels.join(" not in html

def test_sample_records_have_time_and_direction():
    for ln in _lines(SAMPLE.read_text()):
        rec = json.loads(ln)
        assert isinstance(rec["ts_ms"], (int, float))
        assert rec["dir"] in ("in", "out", "meta")
        body = rec.get("event") or rec.get("action") or rec.get("meta")
        assert isinstance(body, dict)


def test_embed_rejects_broken_json():
    with pytest.raises(json.JSONDecodeError):
        embed_sample.embed(HTML.read_text(), '{"ts_ms": 0}\n{broken\n')


def test_embed_escapes_script_close_and_roundtrips():
    trace = '{"ts_ms":0,"dir":"out","action":{"type":"speak","text":"</script><b>x</b>"}}'
    html = embed_sample.embed(HTML.read_text(), trace)
    block = html.split('id="sample-trace">', 1)[1].split("\n</script>", 1)[0]
    assert "</script>" not in block
    assert json.loads(embed_sample.embedded(html))["action"]["text"] == "</script><b>x</b>"


def test_embed_requires_block():
    with pytest.raises(ValueError):
        embed_sample.embed("<html></html>", "{}")


# ---------------------------------------------------------------- browser
@pytest.fixture(scope="module")
def page():
    sync_api = pytest.importorskip("playwright.sync_api")
    pw = sync_api.sync_playwright().start()
    browser = None
    candidates = [None, os.environ.get("PW_CHROMIUM"), "/opt/pw-browsers/chromium"]
    for exe in candidates:
        if exe is not None and not os.path.exists(exe):
            continue
        try:
            browser = pw.chromium.launch(executable_path=exe) if exe else pw.chromium.launch()
            break
        except Exception:  # noqa: BLE001 - try the next candidate
            continue
    if browser is None:
        pw.stop()
        pytest.skip("no launchable Chromium")
    pg = browser.new_page(viewport={"width": 1280, "height": 900})
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("dialog", lambda d: (errors.append("dialog: " + d.message), d.dismiss()))
    pg.goto(HTML.as_uri())
    pg.errors = errors
    yield pg
    browser.close()
    pw.stop()


def _load(page, text: str) -> dict:
    return page.evaluate(
        """t => { const ok = TraceViewer.load(t, 'test');
                  const m = TraceViewer.state.model;
                  return { ok, recs: m.recs.length, warnings: m.warnings,
                    calls: m.calls.map(c => ({ id: c.id, tool: c.tool, status: c.status, stale: c.stale,
                                               cancelAt: c.cancelAt, mode: c.mode })),
                    snaps: m.snapshots.length, changes: m.snapshots.filter(s => s.isChange).length,
                    t1: m.t1, items: document.querySelectorAll('#chart .item').length }; }""",
        text,
    )


def test_sample_loads_and_marks_stale(page):
    page.errors.clear()
    r = _load(page, SAMPLE.read_text())
    assert r["ok"] and page.errors == []
    assert r["recs"] == 24
    calls = {c["id"]: c for c in r["calls"]}
    assert calls["c1"]["stale"] and calls["c1"]["cancelAt"] == 3080
    assert calls["c2"]["status"] == "timeout" and not calls["c2"]["stale"]
    assert calls["c4"]["mode"] == "W" and calls["c1"]["mode"] == "R"
    assert r["snaps"] == 11 and r["changes"] == 4
    assert any("heartbeat" in w for w in r["warnings"])
    assert page.locator("details.snap").count() == 4
    assert page.locator("#log tr").count() == 24 + 1


def test_seek_hides_future_and_play_advances(page):
    _load(page, SAMPLE.read_text())
    page.evaluate("TraceViewer.seek(3500)")
    future = page.locator("#chart .item.future").count()
    total = page.locator("#chart .item").count()
    assert 0 < future < total
    assert "search_flights (c2" in page.locator("#nowTools").inner_text()
    page.click("#playBtn")
    page.wait_for_timeout(400)
    page.click("#playBtn")
    assert page.evaluate("TraceViewer.state.now") > 3500


def test_play_from_end_restarts(page):
    _load(page, SAMPLE.read_text())
    page.click("#playBtn")
    page.wait_for_timeout(100)
    page.click("#playBtn")
    assert page.evaluate("TraceViewer.state.now") < 1000


def test_bare_wire_dialect_and_bad_lines_do_not_crash(page):
    page.errors.clear()
    lines = [
        {"type": "tool_manifest", "ts_ms": 0, "tools": [{"name": "lookup", "kind": "read"}]},
        {"type": "text_chunk", "ts_ms": 100, "text": "hi", "end_of_turn": True},
        {"type": "tool_call", "t": 150, "call_id": "k1", "tool": "lookup", "args": {}},
        {"type": "tool_result", "ts_ms": 90, "call_id": "zz", "status": "ok"},  # out of order + orphan
        {"type": "mystery", "ts_ms": 200},
        {"type": "final", "ts_ms": 300, "text": "done", "snapshot": {"intent": "x", "slots": {"a": 1}}},
        {"type": "speak", "text": "no timestamp"},
        {"type": "final_response", "ts_ms": 320, "text": "again"},
    ]
    text = "\n".join(json.dumps(x) for x in lines) + "\n{not json\n42\n"
    r = _load(page, text)
    assert r["ok"] and page.errors == []
    w = " | ".join(r["warnings"])
    for needle in ("not valid JSON", "no timestamp", "out of order", "never issued", "unknown type",
                   "never got a result", "without a state snapshot"):
        assert needle in w, needle
    calls = {c["id"]: c for c in r["calls"]}
    assert calls["k1"]["mode"] == "R" and calls["k1"]["status"] == "pending"


def test_json_array_and_seconds_time(page):
    page.errors.clear()
    arr = [{"dir": "in", "t_s": 0.5, "data": {"type": "audio", "duration_ms": 800}},
           {"dir": "agent", "t_s": 1.0, "data": {"type": "clarify", "text": "Which one?"}}]
    r = _load(page, json.dumps(arr))
    assert r["ok"] and r["recs"] == 2 and page.errors == []
    assert r["t1"] == pytest.approx(1300)


def test_duplicate_state_modifying_success_is_flagged(page):
    lines = [
        {"ts_ms": 0, "dir": "in", "event": {"type": "tool_manifest", "tools": [{"name": "book", "read_only": False}]}},
        {"ts_ms": 10, "dir": "out", "action": {"type": "tool_call", "call_id": "b1", "tool": "book", "args": {}}},
        {"ts_ms": 20, "dir": "out", "action": {"type": "tool_call", "call_id": "b2", "tool": "book", "args": {}}},
        {"ts_ms": 30, "dir": "in", "event": {"type": "tool_result", "call_id": "b1", "status": "ok"}},
        {"ts_ms": 40, "dir": "in", "event": {"type": "tool_result", "call_id": "b2", "status": "ok"}},
    ]
    r = _load(page, "\n".join(json.dumps(x) for x in lines))
    assert any("possible duplicate side effect" in w for w in r["warnings"])


def test_empty_input_is_rejected_without_crash(page):
    page.errors.clear()
    ok = page.evaluate("TraceViewer.load('   ', 'empty')")
    assert ok is False
    assert page.errors and page.errors[0].startswith("dialog:")


def test_sys_bookkeeping_records_are_not_unknown(page):
    page.errors.clear()
    lines = [
        {"dir": "sys", "kind": "scenario_start", "t_ms": 0, "data": {"scenario_id": "s_sys"}},
        {"dir": "out", "kind": "tool_call", "t_ms": 5, "data": {"type": "tool_call", "call_id": "c", "name": "f", "arguments": {}}},
        {"dir": "sys", "kind": "tool_started", "t_ms": 6, "data": {"call_id": "c"}},
        {"dir": "in", "kind": "tool_result", "t_ms": 9, "data": {"type": "tool_result", "call_id": "c", "status": "ok"}},
        {"dir": "sys", "kind": "write_committed", "t_ms": 9, "data": {"call_id": "c"}},
    ]
    r = _load(page, "\n".join(json.dumps(x) for x in lines))
    assert r["ok"] and page.errors == []
    assert not any("unknown type" in w for w in r["warnings"])
    assert page.evaluate("TraceViewer.state.model.meta.body.scenario_id") == "s_sys"
    assert r["calls"][0]["tool"] == "f"


TRACES = HERE.parent / "runs" / "latest" / "traces"


@pytest.mark.skipif(not TRACES.is_dir(), reason="no harness run in runs/latest")
def test_real_harness_traces_load(page):
    for f in sorted(TRACES.glob("*.jsonl")):
        page.errors.clear()
        r = _load(page, f.read_text())
        assert r["ok"] and page.errors == [], f.name
        assert not any("unknown type" in w or "not valid JSON" in w for w in r["warnings"]), (f.name, r["warnings"])
