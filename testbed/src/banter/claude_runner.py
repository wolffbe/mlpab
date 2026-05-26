"""Spawn `claude -p` to drive a challenge.

We set up a project-scoped .claude/settings.json with a PreToolUse hook so
every tool call is logged, then run `claude -p` against the user's configured
Anthropic credentials (api-key or login) and capture the full stream-json
transcript. Cost + token totals come from the transcript's `result` event.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TESTBED_ROOT = Path(__file__).resolve().parents[2]
HOOK_SCRIPT = TESTBED_ROOT / "hooks" / "log_tool_call.py"
# Shared pip download cache so deps Claude installs into a per-run venv
# (torch, sklearn, etc.) are not re-downloaded for every run. The per-run
# venv stays empty on creation — this only avoids the network round-trip.
PIP_CACHE_DIR = TESTBED_ROOT / "cache" / "pip"

# Rate-limit retry budget. If claude -p exits with a 429/529-style error we
# sleep with exponential backoff and try again until total elapsed wall-clock
# reaches 12h, then give up.
RATE_LIMIT_RETRY_WINDOW_S = 12 * 3600
RATE_LIMIT_BASE_BACKOFF_S = 2
RATE_LIMIT_MAX_BACKOFF_S = 3600
_RATE_LIMIT_TOKENS = ("rate_limit", "rate limit", "overloaded", "429", "529")


@dataclass
class ClaudeResult:
    exit_code: int
    wall_time_s: float
    transcript_path: Path
    stderr_path: Path


def _write_settings(run_dir: Path, command_log: Path) -> None:
    # `matcher` is a regex; ".*" matches every tool name. (`-p` mode silently
    # ignores invalid settings, so the field must be present and well-formed.)
    settings = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": ".*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"python3 {HOOK_SCRIPT}",
                        }
                    ],
                }
            ]
        }
    }
    settings_dir = run_dir / ".claude"
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "settings.json").write_text(json.dumps(settings, indent=2))
    # Mirror the env var into the settings dir for debugging by hand.
    (settings_dir / "command_log_path.txt").write_text(str(command_log))


def _write_mcp_config(run_dir: Path, mcp_servers: dict[str, Any]) -> Path | None:
    if not mcp_servers:
        return None
    cfg_path = run_dir / ".mcp.json"
    cfg_path.write_text(json.dumps({"mcpServers": mcp_servers}, indent=2))
    return cfg_path


def run(
    prompt: str,
    run_dir: Path,
    auth: str,
    model: str,
    cli_binary: str | None,
    sdk_module: str | None,
    mcp_servers: dict[str, Any],
    command_log: Path,
    timeout_s: int = 60 * 60,
) -> ClaudeResult:
    """Spawn `claude -p`.

    auth == "api-key": leave ANTHROPIC_API_KEY in env (claude-code reads it).
    auth == "login":   strip ANTHROPIC_API_KEY so claude-code uses stored OAuth
                       credentials from `claude /login`.
    """
    if shutil.which("claude") is None:
        raise RuntimeError("`claude` CLI not found on PATH. Install Claude Code first.")

    _write_settings(run_dir, command_log)
    mcp_cfg = _write_mcp_config(run_dir, mcp_servers)

    transcript_path = run_dir / "transcript.jsonl"
    stderr_path = run_dir / "claude.stderr.log"
    command_log.touch()

    env = os.environ.copy()
    # ANTHROPIC_BASE_URL would route claude-code through a proxy; we don't use
    # one, so strip it in case the parent shell set one.
    env.pop("ANTHROPIC_BASE_URL", None)
    if auth == "login":
        env.pop("ANTHROPIC_API_KEY", None)
    env["ANTHROPIC_MODEL"] = model
    env["TESTBED_COMMAND_LOG"] = str(command_log)
    if cli_binary:
        env["TESTBED_CLI_BINARY"] = cli_binary
    if sdk_module:
        env["TESTBED_SDK_MODULE"] = sdk_module

    # Activate the per-run venv so Claude's `python` / `pip` invocations land
    # there (and not in the system Python, which may be too new for ML deps).
    venv_bin = run_dir / "venv" / "bin"
    if venv_bin.is_dir():
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(run_dir / "venv")

    PIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    env["PIP_CACHE_DIR"] = str(PIP_CACHE_DIR)

    # Absolute path — `--settings` is resolved relative to cwd (=run_dir),
    # not the caller's cwd, so a relative path would miss.
    settings_file = (run_dir / ".claude" / "settings.json").resolve()
    cmd = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        model,
        "--permission-mode",
        "bypassPermissions",
        # Force the per-run settings (including the PreToolUse hook) to load.
        # `-p` mode silently ignores project settings unless asked explicitly.
        "--settings",
        str(settings_file),
        "--setting-sources",
        "project,local,user",
    ]
    if mcp_cfg:
        cmd.extend(["--mcp-config", str(mcp_cfg)])

    start = time.monotonic()
    attempt = 0
    exit_code = 1
    while True:
        attempt += 1
        # Append (don't truncate) so each retry's stream-json events accumulate
        # — parse_transcript_usage already keeps the LAST result event.
        with open(transcript_path, "ab") as out, open(stderr_path, "ab") as err:
            proc = subprocess.Popen(cmd, cwd=run_dir, env=env, stdout=out, stderr=err)
            try:
                exit_code = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
                exit_code = 124  # conventional timeout code

        if exit_code == 0 or not _last_result_is_rate_limited(transcript_path):
            break
        elapsed = time.monotonic() - start
        backoff = min(
            RATE_LIMIT_BASE_BACKOFF_S * (2 ** (attempt - 1)),
            RATE_LIMIT_MAX_BACKOFF_S,
        )
        if elapsed + backoff > RATE_LIMIT_RETRY_WINDOW_S:
            print(
                f"[claude_runner] rate-limited after {attempt} attempts "
                f"({elapsed:.0f}s elapsed); 12h retry budget exhausted, giving up.",
                flush=True,
            )
            break
        print(
            f"[claude_runner] rate-limited on attempt {attempt} "
            f"({elapsed:.0f}s elapsed); sleeping {backoff}s before retry.",
            flush=True,
        )
        time.sleep(backoff)
    wall = time.monotonic() - start

    return ClaudeResult(
        exit_code=exit_code,
        wall_time_s=wall,
        transcript_path=transcript_path,
        stderr_path=stderr_path,
    )


def _last_result_is_rate_limited(transcript_path: Path) -> bool:
    """Inspect the last `result` event in the stream-json transcript and
    decide whether the run stopped on a 429/529-style condition."""
    if not transcript_path.exists():
        return False
    last_result: dict[str, Any] | None = None
    for line in transcript_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            last_result = event
    if not last_result or not last_result.get("is_error"):
        return False
    blob = " ".join(
        str(last_result.get(k) or "") for k in ("api_error_status", "subtype", "result")
    ).lower()
    return any(token in blob for token in _RATE_LIMIT_TOKENS)
