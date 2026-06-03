#!/usr/bin/env python3
"""Claude Code PreToolUse hook. Reads JSON from stdin, classifies the tool
call as cli / mcp / sdk / python / bash / other, appends one JSONL line to
TESTBED_COMMAND_LOG, and blocks calls that try to escape the boundary or
run unobservable background work.

Exit 0 → allow. Non-zero → block (the tool call is rejected by claude-code
and an error is shown to the model).

Blocked:
  - Tool-input flags `dangerouslyDisableSandbox: true`, `run_in_background: true`.
  - `Read`/`Write`/`Edit` paths outside the boundary (env `TESTBED_BOUNDARY`),
    or pointing at known-sensitive $HOME subdirs. This is the only way to
    enforce Read/Write/Edit path denies under `bypassPermissions` — Claude
    Code's `permissions.deny` patterns for those tools are silently skipped
    in bypass mode, but PreToolUse hooks fire regardless of mode.
"""
from __future__ import annotations

import json
import os
import pwd
import re
import shlex
import sys
import time

PYTHON_PREFIXES = ("python", "python3", "uv run python", "uv run", "pip", "pip3")


def classify(tool_name: str, tool_input: dict) -> str:
    if tool_name.startswith("mcp__"):
        return "mcp"
    if tool_name != "Bash":
        return "other"

    command = (tool_input.get("command") or "").strip()
    if not command:
        return "bash"

    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    if not tokens:
        return "bash"

    first = tokens[0]
    cli_binary = os.environ.get("TESTBED_CLI_BINARY") or None
    if cli_binary and (first == cli_binary or first.endswith(f"/{cli_binary}")):
        return "cli"
    is_python = (
        first in PYTHON_PREFIXES
        or first.endswith(("/python", "/python3", "/pip", "/pip3"))
        or first.endswith(".py")
    )
    if is_python:
        sdk_module = os.environ.get("TESTBED_SDK_MODULE") or None
        if sdk_module:
            m = re.escape(sdk_module)
            if re.search(rf"\b(?:import|from)\s+{m}\b|-m\s+{m}\b", command):
                return "sdk"
        return "python"
    return "bash"


# Tool-input flags that escape our intended boundaries. Blocking via hook
# (the only mechanism available — settings.json can't deny by parameter).
FORBIDDEN_FLAGS = {
    "dangerouslyDisableSandbox": (
        "DENIED: dangerouslyDisableSandbox is forbidden. The sandbox stays "
        "ON. If a write to your cwd is failing with EPERM, check `pwd`, "
        "create parents with `os.makedirs(..., exist_ok=True)`, and keep "
        "paths relative — DO NOT try to bypass the sandbox."
    ),
    "run_in_background": (
        "DENIED: run_in_background is forbidden. Run commands synchronously; "
        "chain with `&&` instead of detaching."
    ),
}


# Tools we intercept for path-based deny. Their tool_input contains
# `file_path` (absolute, resolved by Claude before submitting).
_PATH_TOOLS = ("Read", "Write", "Edit", "NotebookEdit", "MultiEdit")


def _real_home() -> str:
    """User's REAL home from /etc/passwd, ignoring any $HOME redirect."""
    return pwd.getpwuid(os.getuid()).pw_dir


def _path_violates_boundary(file_path: str) -> str | None:
    """Return a rejection reason if `file_path` is outside the boundary or
    points at a sensitive $HOME subdir; None if allowed."""
    if not file_path:
        return None
    p = os.path.realpath(file_path) if os.path.isabs(file_path) else os.path.realpath(
        os.path.join(os.getcwd(), file_path)
    )
    # Boundary check: TESTBED_BOUNDARY is the engineer's challenge dir.
    boundary = os.environ.get("TESTBED_BOUNDARY")
    if boundary:
        b = os.path.realpath(boundary)
        if not (p == b or p.startswith(b + "/")):
            home = _real_home()
            # Allowed escapes — both are Claude Code's OWN infrastructure, not
            # user data:
            #   1. ~/.claude (Claude's config + tokens we already trust).
            #   2. Background-task output files. When Claude auto-moves a long
            #      foreground command to a background task, it writes the output
            #      to <tmp>/claude-*/<cwd-slug>/<session>/tasks/<id>.output —
            #      outside the boundary. Denying the read leaves the agent
            #      blind to its own command. The path embeds the cwd slug, so
            #      this allowance stays scoped to THIS run's own tasks.
            cwd_slug = os.getcwd().replace(os.sep, "-")
            is_task_output = (
                "/tasks/" in p and p.endswith(".output") and cwd_slug in p
            )
            if not p.startswith(home + "/.claude/") and not is_task_output:
                return f"path is outside the engineer boundary ({b})"
    # Always block dotfiles + dot-dirs at $HOME root (secrets), regardless of
    # whether a boundary is set. Catches ~/.ssh, ~/.aws, ~/.gnupg, .kaggle,
    # .gitconfig, .netrc, .zshrc — every secret-bearing path on macOS.
    home = _real_home()
    rel = p[len(home) + 1:] if p.startswith(home + "/") else None
    if rel and rel.startswith("."):
        # ~/.claude is allowed (Claude's config + tokens we already trust).
        if rel == ".claude" or rel.startswith(".claude/"):
            return None
        return f"path targets a sensitive $HOME location ({home}/{rel.split('/')[0]})"
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}

    # Block forbidden tool-input flags. Stderr is shown to the model so it
    # learns why the call was rejected and (hopefully) stops trying.
    for flag, msg in FORBIDDEN_FLAGS.items():
        if tool_input.get(flag) is True:
            print(msg, file=sys.stderr, flush=True)
            return 2  # non-zero → block the tool call

    # Path-based deny for Read/Write/Edit. Enforces what `permissions.deny`
    # silently skips under bypassPermissions.
    if tool_name in _PATH_TOOLS:
        # NotebookEdit uses `notebook_path`; everything else uses `file_path`.
        path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        reason = _path_violates_boundary(path)
        if reason:
            print(f"DENIED: {tool_name}({path!r}) — {reason}. Stay inside your "
                  f"working directory (or ~/.claude); rewrite the call to a "
                  f"path under cwd.", file=sys.stderr, flush=True)
            return 2

    # Log the (allowed) call.
    log_path = os.environ.get("TESTBED_COMMAND_LOG")
    if log_path:
        record = {
            "timestamp": time.time(),
            "session_id": payload.get("session_id"),
            "tool_name": tool_name,
            "category": classify(tool_name, tool_input),
            "tool_input": tool_input,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
