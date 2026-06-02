"""Spawn `claude -p` to drive a challenge.

Sets up a project-scoped .claude/settings.json (PreToolUse hook + sandbox +
denies), runs `claude -p`, and captures the stream-json transcript. Cost +
token totals come from the transcript's `result` event.

`run_with_retry` (shared with the autoresearch researcher) handles 429/529
and 5xx back-off across a 6h budget.
"""
from __future__ import annotations

import json
import os
import pwd
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from banter import streaming


TESTBED_ROOT = Path(__file__).resolve().parents[2]
# Real user home from /etc/passwd, NOT $HOME — autoresearch redirects $HOME
# into the run dir, so by the time `banter run` loads this module its
# environ.HOME is already <run>/. Using that as HOME_DIR would emit a
# denyRead rooted at <run>/, which then blocks the engineer's own writes
# inside its boundary (parent-dir lookups for open(..., 'w') need read).
HOME_DIR = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
# Hook script ships inside the package (src/banter/hooks/).
HOOK_SCRIPT = Path(__file__).resolve().parent / "hooks" / "log_tool_call.py"

# Tools blocked for both researcher and engineer: scheduling/async (end the
# `-p` turn or run unobservably), nested subagents (cost-attribution loss),
# plan/worktree modes (unused), and a few meta tools. `mcp__*` tools stay
# allowed — including whichever platform MCP servers are loaded.
COMMON_DENY = [
    # Background scheduling / async — calling these in -p ends the turn.
    "ScheduleWakeup",
    "CronCreate", "CronDelete", "CronList",
    "Task", "TaskCreate", "TaskUpdate", "TaskGet", "TaskList", "TaskOutput", "TaskStop",
    "RemoteTrigger", "Monitor", "PushNotification",
    # Nested subagents — bloat context, break per-call cost attribution.
    "Agent",
    # Plan / worktree modes unused in autoresearch / benchmark.
    "EnterPlanMode", "ExitPlanMode",
    "EnterWorktree", "ExitWorktree",
    # Meta / helper tools not needed for the loop.
    "ToolSearch", "ShareOnboardingGuide",
]

# Engineer-only denies. Web access kills reproducibility for the deterministic
# ML task. Researcher keeps them — may look up library docs.
ENGINEER_ONLY_DENY = ["WebFetch", "WebSearch"]


def deny_patterns_for(boundary: Path) -> list[str]:
    """Build `permissions.deny` patterns that DO fire under `bypassPermissions`.

    Covers Bash escape patterns (cd .., cat/ls dotfiles in $HOME) — these
    are the ones the engineer can't bypass. Read/Write/Edit `file_path`
    patterns would be silently skipped in bypass mode, so those are
    enforced by the PreToolUse hook instead (see `hooks/log_tool_call.py`).
    """
    boundary.resolve()  # accepted for future use
    home = str(HOME_DIR)
    return [
        "Bash(cd *..*)",
        f"Bash(cat {home}/.*)",  f"Bash(cat {home}/.*/**)",
        f"Bash(ls {home}/.*)",   f"Bash(ls {home}/.*/**)",
    ]


def oauth_token_from_keychain() -> str | None:
    """Return the Claude Code OAuth access token from macOS Keychain, or None.

    None on non-macOS hosts, or when the user hasn't run `claude /login`.
    """
    try:
        out = subprocess.run(
            ["/usr/bin/security", "find-generic-password",
             "-s", "Claude Code-credentials", "-w"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout
        return json.loads(out)["claudeAiOauth"]["accessToken"]
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
        return None


# File-based fallback for the OAuth token. Claude Code's Bash tool strips
# CLAUDE_CODE_OAUTH_TOKEN from child env, so the chain
# autoresearch → researcher → bash → banter run → engineer loses it.
# autoresearch writes the token to <run>/.claude-oauth (mode 0600) and
# points BANTER_TOKEN_CACHE at it; custom BANTER_* vars survive the hop.
TOKEN_CACHE_FILENAME = ".claude-oauth"
TOKEN_CACHE_ENV = "BANTER_TOKEN_CACHE"


def write_token_cache(token: str, run_path: Path) -> Path:
    """Write the OAuth token to ``run_path/.claude-oauth`` (mode 0600).

    Returns the cache path. The caller is responsible for forwarding it via
    the ``BANTER_TOKEN_CACHE`` env var so child banter invocations find it.
    """
    cache = run_path / TOKEN_CACHE_FILENAME
    cache.write_text(token)
    cache.chmod(0o600)
    return cache


def read_token_cache() -> str | None:
    """Read the OAuth token from the cache pointed at by BANTER_TOKEN_CACHE."""
    path = os.environ.get(TOKEN_CACHE_ENV)
    if not path:
        return None
    try:
        s = Path(path).read_text().strip()
        return s or None
    except (FileNotFoundError, OSError):
        return None


def resolve_oauth_token() -> str | None:
    """Get the OAuth token via the most reliable available path.

    Order: env (already-propagated by upstream), BANTER_TOKEN_CACHE file,
    Keychain (works from the user's shell but fails silently in some nested
    subprocess contexts). Returns the token string or None.
    """
    return (os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
            or read_token_cache()
            or oauth_token_from_keychain())


# Rate-limit / transient-error retry budget. On a matching exit-error we
# exp-backoff up to RATE_LIMIT_MAX_BACKOFF_S and retry until total wall-clock
# reaches RATE_LIMIT_RETRY_WINDOW_S, then give up.
RATE_LIMIT_RETRY_WINDOW_S = 6 * 3600
RATE_LIMIT_BASE_BACKOFF_S = 2
RATE_LIMIT_MAX_BACKOFF_S = 3600

# When set (by autoresearch), `run_with_retry` appends every rate-limit sleep
# (in seconds) to this file so the compute-time budget can exclude waiting.
# Custom BANTER_* vars survive the researcher → Bash → engineer hop, so the
# researcher and all its engineer subprocesses accumulate into one ledger.
RATE_LIMIT_LEDGER_ENV = "BANTER_RATELIMIT_LEDGER"


def _record_rate_limit_wait(env: dict[str, str], seconds: float) -> None:
    """Append `seconds` to the rate-limit-wait ledger named by env, if set.

    Best-effort: a failed write must never abort the retry loop.
    """
    path = env.get(RATE_LIMIT_LEDGER_ENV)
    if not path:
        return
    try:
        with open(path, "a") as f:
            f.write(f"{seconds}\n")
    except OSError:
        pass


def read_rate_limit_wait(ledger: Path) -> float:
    """Sum the rate-limit-wait ledger (seconds). Missing file → 0.0; any
    unparseable line is skipped rather than discarding the whole total."""
    total = 0.0
    try:
        lines = ledger.read_text().splitlines()
    except OSError:
        return 0.0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            total += float(line)
        except ValueError:
            continue
    return total


# Anthropic rate limits (429/529, "overloaded") + transient 5xx. Without 5xx
# in here, a mid-session 500 burns a whole version on retry.
_RATE_LIMIT_TOKENS = (
    "rate_limit", "rate limit", "overloaded",
    "429", "529",
    "500", "502", "503", "504",
    "internal server error", "bad gateway", "service unavailable", "gateway timeout",
)


@dataclass
class ClaudeResult:
    exit_code: int
    wall_time_s: float
    transcript_path: Path
    stderr_path: Path


# Baseline outbound hosts every engineer needs regardless of platform:
# Claude itself (api.anthropic.com) + loopback for any local platform server.
# Platform-specific hosts come from each `platforms/<platform>/<interface>/config.yaml`
# via the `allowed_domains:` key.
BASE_ALLOWED_DOMAINS = (
    "127.0.0.1", "localhost",
    "api.anthropic.com", "*.anthropic.com",
    "statsig.anthropic.com", "console.anthropic.com",
)


def _write_settings(
    run_dir: Path,
    command_log: Path,
    version_dir: Path | None = None,
    allowed_domains: list[str] | None = None,
) -> None:
    """Write the engineer's project-scoped `.claude/settings.json`.

    `version_dir` (autoresearch `<run>/v<N>`) overrides the boundary;
    otherwise it falls back to `run_dir` (benchmark challenge dir).

    Confinement: Claude Code's built-in sandbox kernel-confines Bash
    subprocesses to the boundary. `permissions.deny` adds tool-name and
    Bash escape denies. Read/Write/Edit are NOT kernel-confined in
    bypassPermissions — soft (prompt + cwd) only.

    Network is unrestricted (no `allowedDomains`); `allowLocalBinding`
    covers platform servers (mlkit binds 127.0.0.1:8765). `.claude/`
    state lands inside `boundary` via the HOME redirect in :func:`run`.
    """
    # Engineer's world = its challenge folder (`run_dir`). Each challenge is
    # a separate engineer invocation and is isolated from siblings (same
    # version) and from all other versions / runs. allowRead also reaches
    # the technical infra the engineer needs to function: ~/.claude, shared
    # Python libs (where `./venv/` symlinks resolve), the mle-bench cache
    # (where `./data/` resolves), and the PreToolUse hook script.
    # `version_dir` is accepted for back-compat but no longer used — it would
    # leak sibling challenges.
    # Engineer's world = its challenge folder. Everything it needs lives
    # inside the boundary: venv (materialized by `runner._make_venv`),
    # challenge data (cloned by `mlebench_wrapper.prepare`), and the
    # PreToolUse hook script (copied below). allowRead therefore only needs
    # the boundary itself plus ~/.claude for Claude Code's own config.
    _ = version_dir  # accepted for back-compat; not used
    boundary = run_dir.resolve()
    settings_dir = run_dir / ".claude"
    settings_dir.mkdir(parents=True, exist_ok=True)
    # Copy the hook script into the boundary so the PreToolUse hook can
    # invoke it without reading outside the sandbox.
    hooks_dir = settings_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    local_hook = hooks_dir / HOOK_SCRIPT.name
    shutil.copy(HOOK_SCRIPT, local_hook)
    local_hook.chmod(0o755)
    # Sandbox: NETWORK gated, filesystem FULLY OPEN. Enabling the sandbox at
    # all triggers default filesystem restrictions even without a `filesystem`
    # block — so we explicitly opt back out with allowRead/Write `["/"]`,
    # then layer just the network allowlist. Result:
    #   - Outbound: only baseline (Claude API + loopback) + whatever the
    #     platform declares in its `allowed_domains:` config. No pypi /
    #     HF / kaggle / github / arbitrary HTTPS during the run.
    #   - Filesystem: unrestricted. Confinement falls to the PreToolUse
    #     hook (path-based deny for Read/Write/Edit, enforced regardless
    #     of permission mode) + `permissions.deny` (Bash escape patterns
    #     + tool-name denies).
    domain_list = list(BASE_ALLOWED_DOMAINS) + list(allowed_domains or [])
    settings = {
        "hooks": {
            "PreToolUse": [{
                "matcher": ".*",  # regex over tool names
                "hooks": [{"type": "command", "command": f"python3 {local_hook}"}],
            }]
        },
        "sandbox": {
            "enabled": True,
            "autoAllowBashIfSandboxed": True,
            "filesystem": {
                "allowRead":  ["/"],
                "allowWrite": ["/"],
            },
            "network": {
                "allowLocalBinding": True,
                "allowedDomains": domain_list,
            },
        },
        "permissions": {
            "deny": list(COMMON_DENY) + list(ENGINEER_ONLY_DENY) + deny_patterns_for(boundary),
        },
    }
    (settings_dir / "settings.json").write_text(json.dumps(settings, indent=2))
    # Mirror the command log path for debugging.
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
    version_dir: Path | None = None,
    allowed_domains: list[str] | None = None,
) -> ClaudeResult:
    """Spawn `claude -p` (the engineer).

    Always forwards a Keychain OAuth token (when present) since the
    redirected HOME has no on-disk credentials. `auth="login"` additionally
    strips ANTHROPIC_API_KEY so OAuth wins; `auth="api-key"` keeps both
    and lets claude-code pick. `extra_env` layers platform credentials.
    """
    if shutil.which("claude") is None:
        raise RuntimeError("`claude` CLI not found on PATH. Install Claude Code first.")

    _write_settings(run_dir, command_log, version_dir=version_dir,
                    allowed_domains=allowed_domains)
    mcp_cfg = _write_mcp_config(run_dir, mcp_servers)

    transcript_path = run_dir / "transcript.jsonl"
    stderr_path = run_dir / "claude.stderr.log"
    command_log.touch()

    env = os.environ.copy()
    env.pop("ANTHROPIC_BASE_URL", None)  # strip any parent-shell proxy
    # HOME → boundary so `.claude/` state lands inside the run dir.
    boundary = (version_dir or run_dir).resolve()
    env["HOME"] = str(boundary)
    # Auth: redirected HOME has no on-disk creds. resolve_oauth_token() tries
    # env → token cache file → Keychain in order; the cache is what makes the
    # researcher → bash → banter run → engineer chain work (env is stripped
    # by Claude Code's Bash tool).
    token = resolve_oauth_token()
    if token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    if auth == "login":
        env.pop("ANTHROPIC_API_KEY", None)
    if not env.get("ANTHROPIC_API_KEY") and not env.get("CLAUDE_CODE_OAUTH_TOKEN"):
        print("[claude_runner] WARNING: no auth available (no ANTHROPIC_API_KEY, "
              "no token in env/cache/Keychain). Engineer will fail. Re-run "
              "banter from a shell where `claude /login` has been run.",
              flush=True)
    env["ANTHROPIC_MODEL"] = model
    env["TESTBED_COMMAND_LOG"] = str(command_log)
    # The PreToolUse hook reads this to enforce Read/Write/Edit path denies
    # — those are silently skipped under bypassPermissions, but the hook
    # fires regardless and rejects calls outside the boundary.
    env["TESTBED_BOUNDARY"] = str(boundary)
    if cli_binary:
        env["TESTBED_CLI_BINARY"] = cli_binary
    if sdk_module:
        env["TESTBED_SDK_MODULE"] = sdk_module

    # Activate per-run venv so claude's python/pip find ML deps via .pth.
    venv_bin = run_dir / "venv" / "bin"
    if venv_bin.is_dir():
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(run_dir / "venv")

    # Skip pip cache (per-run venv inherits via .pth; cache dir unwritable).
    env["PIP_NO_CACHE_DIR"] = "1"

    # Platform credentials override inherited values.
    if extra_env:
        env.update({k: v for k, v in extra_env.items() if v})

    # Absolute path: `--settings` resolves against cwd (=run_dir).
    settings_file = (run_dir / ".claude" / "settings.json").resolve()
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "stream-json", "--verbose",
        "--model", model,
        "--permission-mode", "bypassPermissions",
        # `-p` mode skips project settings unless we ask for them explicitly.
        "--settings", str(settings_file),
        "--setting-sources", "project,local,user",
    ]
    if mcp_cfg:
        cmd.extend(["--mcp-config", str(mcp_cfg)])

    # `runner.run` wraps the call in `streaming.tee_to` for stream.log;
    # full transcript still lands in transcript.jsonl regardless.
    exit_code, wall = run_with_retry(
        cmd=cmd,
        cwd=run_dir,
        env=env,
        transcript_path=transcript_path,
        stderr_path=stderr_path,
        timeout_s=timeout_s,
        on_line=streaming.make_printer("engineer"),
        log_prefix="claude_runner",
    )

    # `run_with_retry` only writes stderr_path when there's actual stderr
    # output, so no empty-file cleanup is needed here.

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

    - Appends stdout to `transcript_path` (stream-json). Stderr is buffered
      to a temp file and only PROMOTED to `stderr_path` if non-empty, so
      the run dir stays clean when nothing went wrong.
    - When `on_line` is given, stdout is piped through and each line is
      forwarded to the callback (for live streaming display).
    - On rate-limit / 5xx detection in the last `result`, sleeps
      `min(2 * 2^(attempt-1), 3600)` seconds and retries until the 6h
      total budget is exhausted.
    - Each rate-limit sleep is recorded to the `BANTER_RATELIMIT_LEDGER` file
      (when that env var is set) so the autoresearch compute-time budget can
      EXCLUDE waiting-on-rate-limit time from the session wall clock.

    Returns `(exit_code, total_wall_time_s)` including sleep time.
    """
    import tempfile

    start = time.monotonic()
    attempt = 0
    exit_code = 1
    # Buffer stderr to a temp file outside the run dir; we only move it into
    # place at the END if it has actual content.
    err_tmp = tempfile.NamedTemporaryFile(prefix="banter-err-", delete=False)
    err_tmp_path = Path(err_tmp.name)
    err_tmp.close()
    while True:
        attempt += 1
        if on_line is not None:
            with open(transcript_path, "ab") as tf, open(err_tmp_path, "ab") as sf:
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
            with open(transcript_path, "ab") as out, open(err_tmp_path, "ab") as err:
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
        # Record the rate-limit wait so the autoresearch compute-time budget
        # can subtract it from session wall clock (waiting != computing).
        _record_rate_limit_wait(env, backoff)

    # Promote the temp stderr buffer to `stderr_path` only if non-empty;
    # otherwise drop it so the run dir doesn't carry an empty log file.
    try:
        if err_tmp_path.exists() and err_tmp_path.stat().st_size > 0:
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            err_tmp_path.replace(stderr_path)
        else:
            err_tmp_path.unlink(missing_ok=True)
    except OSError:
        pass

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
