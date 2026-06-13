"""Strip vendored agent instructions from materialized repos.

Used by `interfaces.build` after cloning/copying interface source: a repo's own
`.claude/` (settings, hooks, project memory), `CLAUDE.md`, and `.mcp.json` —
e.g. hopsworks-api ships a `.claude/` — must never be auto-loaded as directives
by the agent working in or near that source. Functional VCS metadata (`.git`)
is left alone, since rebuilds rely on the pinned checkout.
"""
from __future__ import annotations

import shutil
from pathlib import Path

_AGENT_PLUMBING_DIRS = (".claude",)
_AGENT_PLUMBING_FILES = ("CLAUDE.md", ".mcp.json")


def strip_agent_plumbing(dst: Path) -> None:
    """Recursively remove `.claude/`, `CLAUDE.md`, and `.mcp.json` from `dst`.

    Use after cloning/copying ANY repo so its own agent instructions are never
    auto-loaded as directives by Claude Code.
    """
    dst = Path(dst)
    for d in _AGENT_PLUMBING_DIRS:
        for p in dst.rglob(d):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
    for f in _AGENT_PLUMBING_FILES:
        for p in dst.rglob(f):
            if p.is_file():
                p.unlink(missing_ok=True)
