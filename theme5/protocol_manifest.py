# /mnt/project-files/theme5/theme5/protocol_manifest.py
"""Compatibility shim: manifest parsing now lives in protocol.py (the one
ASSUMPTION adapter). Import from theme5.protocol in new code."""
from .protocol import (  # noqa: F401
    ParamDef, ToolDef, from_toolspec, parse_manifest_defs, parse_params, parse_tool_def,
)
