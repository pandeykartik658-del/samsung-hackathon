# /mnt/project-files/theme5/tests/test_state.py
from theme5.state import SessionState


def test_localized_update_only_touches_named_slots():
    s = SessionState()
    s.update({"origin": "Delhi", "destination": "Mumbai"})
    v = s.version
    changed = s.update({"destination": "Pune"}, correction=True)
    assert changed == {"destination"}
    assert s.slots == {"origin": "Delhi", "destination": "Pune"}
    assert s.version == v + 1
    assert s.corrections == [("destination", "Mumbai", "Pune")]


def test_noop_update_does_not_bump_version():
    s = SessionState()
    s.update({"a": 1})
    v = s.version
    assert s.update({"a": 1}) == set() and s.version == v


def test_none_removes_slot_and_intent_changes():
    s = SessionState()
    s.update({"a": 1})
    s.update({"a": None})
    assert "a" not in s.slots
    assert s.set_intent("x") and not s.set_intent("x")


def test_frames_and_snapshot_and_reset():
    s = SessionState()
    s.add_frame("f1", "a tv")
    s.set_intent("lookup")
    assert s.snapshot() == {"intent": "lookup", "slots": {"frame": "f1"}}
    assert s.frame_notes["f1"] == "a tv"
    s.reset_task()
    assert s.snapshot() == {"intent": None, "slots": {}}
