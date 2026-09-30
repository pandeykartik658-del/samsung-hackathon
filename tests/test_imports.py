# /mnt/project-files/theme5/tests/test_imports.py
"""Every module in the package imports cleanly (no hidden side effects,
no circular imports)."""
import importlib
import pkgutil

import theme5


def test_import_everything():
    names = sorted(m.name for m in pkgutil.iter_modules(theme5.__path__) if m.name != "__main__")
    for required in ("protocol", "clock", "events", "engine", "fastpath", "slowpath", "coordinator", "slots",
                     "tools", "multimodal", "trace", "cli", "agent"):
        assert required in names, required
    for n in names:
        importlib.import_module(f"theme5.{n}")
