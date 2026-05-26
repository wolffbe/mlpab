#!/usr/bin/env python3
"""Claude Code PreToolUse hook. Reads JSON from stdin, classifies the tool
call as cli / mcp / sdk / python / bash / other, and appends one JSONL line
to the file pointed at by TESTBED_COMMAND_LOG.

Claude Code invokes this script before every tool call; exit 0 lets the call
proceed. Any non-zero exit would block the tool, which we never want here.

The CLI binary that counts as a "cli" call is set via TESTBED_CLI_BINARY;
the SDK module via TESTBED_SDK_MODULE. Both default to unset — a missing env
var means the corresponding bucket simply gets no matches for this run.
"""
from __future__ import annotations

import json
import os
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


def main() -> int:
    log_path = os.environ.get("TESTBED_COMMAND_LOG")
    if not log_path:
        return 0
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}
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
