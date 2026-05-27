"""Spawn `claude -p` to drive a challenge.

We set up a project-scoped .claude/settings.json with a PreToolUse hook so
every tool call is logged, then run `claude -p` against the user's configured
Anthropic credentials (api-key or login) and capture the full stream-json
transcript. Cost + token totals come from the transcript's `result` event.

The same rate-limit retry helper (`run_with_retry`) is used by the engineer
(`run` in this module) AND the autoresearch researcher — every `claude -p`
invocation in the testbed goes through it, so a 429/529 anywhere triggers
exponential back-off within the same 12h retry budget.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


TESTBED_ROOT = Path(__file__).resolve().parents[2]
# Hook script ships inside the package (src/banter/hooks/).
HOOK_SCRIPT = Path(__file__).resolve().parent / "hooks" / "log_tool_call.py"
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
    extra_env: dict[str, str] | None = None,
) -> ClaudeResult:
    """Spawn `claude -p` (the engineer instance).

    auth == "api-key": leave ANTHROPIC_API_KEY in env (claude-code reads it).
    auth == "login":   strip ANTHROPIC_API_KEY so claude-code uses stored OAuth
                       credentials from `claude /login`.

    `extra_env` (e.g. an interface's credential keys) is layered onto the
    engineer's environment so the CLI/SDK/MCP server can authenticate.
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

    # Interface credential keys (from the manifest) override inherited values so
    # the engineer's CLI/SDK/MCP server authenticates against the configured host.
    if extra_env:
        env.update({k: v for k, v in extra_env.items() if v})

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

    exit_code, wall = run_with_retry(
        cmd=cmd,
        cwd=run_dir,
        env=env,
        transcript_path=transcript_path,
        stderr_path=stderr_path,
        timeout_s=timeout_s,
        log_prefix="claude_runner",
    )

    return ClaudeResult(
        exit_code=exit_code,
        wall_time_s=wall,
        transcript_path=transcript_path,
        stderr_path=stderr_path,
    )


def run_with_retry(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    transcript_path: Path,
    stderr_path: Path,
    timeout_s: int | None = None,
    on_line: Callable[[str], None] | None = None,
    log_prefix: str = "claude_runner",
) -> tuple[int, float]:
    """Run a `claude -p` subprocess with exponential rate-limit back-off.

    Used by both the engineer and the autoresearch researcher.

    - Appends stdout to `transcript_path` (stream-json) and stderr to
      `stderr_path`. Appending — not truncating — so each retry's events
      accumulate; `_last_result_is_rate_limited` always inspects the LAST
      `result` event.
    - When `on_line` is given, stdout is piped through and each line is
      forwarded to the callback (for live streaming display).
    - On rate-limit detection (429/529/overloaded in the last `result`),
      sleeps `min(2 * 2^(attempt-1), 3600)` seconds and retries until the
      12h total budget is exhausted.

    Returns `(exit_code, total_wall_time_s)` including sleep time.
    """
    start = time.monotonic()
    attempt = 0
    exit_code = 1
    while True:
        attempt += 1
        if on_line is not None:
            with open(transcript_path, "ab") as tf, open(stderr_path, "ab") as sf:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(cwd),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=sf,
                    text=True,
                )
                try:
                    for raw_line in proc.stdout:  # type: ignore[union-attr]
                        raw_line = raw_line.rstrip("\n")
                        tf.write((raw_line + "\n").encode())
                        tf.flush()
                        if raw_line.strip():
                            try:
                                on_line(raw_line)
                            except Exception:
                                pass
                    if timeout_s is not None:
                        exit_code = proc.wait(timeout=timeout_s)
                    else:
                        exit_code = proc.wait()
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
                    exit_code = 124
                except KeyboardInterrupt:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise
        else:
            with open(transcript_path, "ab") as out, open(stderr_path, "ab") as err:
                proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=out, stderr=err)
                try:
                    if timeout_s is not None:
                        exit_code = proc.wait(timeout=timeout_s)
                    else:
                        exit_code = proc.wait()
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
                    exit_code = 124
                except KeyboardInterrupt:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise

        if exit_code == 0 or not _last_result_is_rate_limited(transcript_path):
            break
        elapsed = time.monotonic() - start
        backoff = min(
            RATE_LIMIT_BASE_BACKOFF_S * (2 ** (attempt - 1)),
            RATE_LIMIT_MAX_BACKOFF_S,
        )
        if elapsed + backoff > RATE_LIMIT_RETRY_WINDOW_S:
            print(
                f"[{log_prefix}] rate-limited after {attempt} attempts "
                f"({elapsed:.0f}s elapsed); {RATE_LIMIT_RETRY_WINDOW_S // 3600}h retry budget "
                f"exhausted, giving up.",
                flush=True,
            )
            break
        print(
            f"[{log_prefix}] rate-limited on attempt {attempt} "
            f"({elapsed:.0f}s elapsed); sleeping {backoff}s before retry.",
            flush=True,
        )
        time.sleep(backoff)
    return exit_code, time.monotonic() - start


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
