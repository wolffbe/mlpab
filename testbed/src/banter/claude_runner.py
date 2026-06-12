"""Spawn `claude -p` to drive a task run.

Sets up a project-scoped .claude/settings.json (PreToolUse hook + sandbox +
denies), runs `claude -p`, and captures the stream-json transcript (cost +
token totals come from its `result` event). `run_with_retry` handles 429/529
and 5xx back-off across a 6h budget.
"""
from __future__ import annotations

import json
import os
import pwd
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from banter import streaming


TESTBED_ROOT = Path(__file__).resolve().parents[2]
# Real user home from /etc/passwd, NOT $HOME — the runner redirects $HOME into
# the run dir, so environ.HOME is already <run>/ by the time this module loads.
# Using that as HOME_DIR would root a denyRead at <run>/, blocking the agent's
# own writes inside its boundary (open(..,'w') parent-dir lookups need read).
HOME_DIR = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
# Hook script ships inside the package (src/banter/hooks/).
HOOK_SCRIPT = Path(__file__).resolve().parent / "hooks" / "log_tool_call.py"

# Tools denied for the agent: scheduling/async (end the `-p`
# turn or run unobservably), nested subagents (cost-attribution loss), plan/
# worktree modes (unused), and a few meta tools. `mcp__*` stay allowed —
# including whichever platform MCP servers are loaded.
COMMON_DENY = [
    # Background scheduling / async — calling these in -p ends the turn.
    "ScheduleWakeup",
    "CronCreate", "CronDelete", "CronList",
    "Task", "TaskCreate", "TaskUpdate", "TaskGet", "TaskList", "TaskOutput", "TaskStop",
    "RemoteTrigger", "Monitor", "PushNotification",
    # Nested subagents — bloat context, break per-call cost attribution.
    "Agent",
    # Plan / worktree modes unused in treatment runs.
    "EnterPlanMode", "ExitPlanMode",
    "EnterWorktree", "ExitWorktree",
    # Meta / helper tools not needed for the loop.
    "ToolSearch", "ShareOnboardingGuide",
]

# Agent-only denies. Web access kills reproducibility for the deterministic
# ML task.
AGENT_ONLY_DENY = ["WebFetch", "WebSearch"]

# Compute libraries whose LOCAL execution the agent hook forbids. Remote-only
# model: training must be pushed to the platform as a remote job via the
# interface, never run locally — a local python command importing any of these is
# blocked (go remote or give up). Light glue (pandas/numpy/csv/json — downloading
# remote results, formatting deliverables) is intentionally NOT here. Overridable
# per run via `compute_deny`; surfaced to the hook as TESTBED_COMPUTE_DENY.
DEFAULT_COMPUTE_DENY = [
    "torch", "torchvision", "tensorflow", "keras", "sklearn", "scikit_learn",
    "xgboost", "lightgbm", "catboost", "jax", "flax", "transformers",
]

# Bootstrapped into the agent's run venv (as `_banter_apilog.py` + a `.pth`
# that imports it): wraps `requests.Session.send` to append every outbound
# request's {method, path, src} to BANTER_API_LOG. The Hopsworks SDK, `hops` CLI,
# and MCP server all run in this venv and all use `requests`, so this captures
# every REST call to the cluster for endpoint-coverage scoring. Best-effort —
# wrapped so it can never break startup/requests.
#
# `src` ATTRIBUTES the call to the interface it came THROUGH by walking the stack:
# a frame in a first-party `.mcp` subtree → "mcp"; `.cli` → "cli"; any other
# first-party frame → "sdk"; none → "other" (raw `requests` the agent hand-
# rolled, NOT via the interface). The scorer counts a whitelist hit only when
# `src` is the interface under test, so off-interface traffic can't inflate
# `whitelist_hits`. First-party roots come from BANTER_IFACE_SDK (set by `run()`
# from the wheel's top_level.txt); the `.mcp`/`.cli` split uses the SDK's in-tree
# subpackage convention (one wheel ships SDK + CLI + MCP). Server-side Job code
# runs on the cluster, not this venv, so its calls never reach the shim — by design.
# NOTE: a `.pth` import line (run by `site` for EVERY site dir), NOT
# `sitecustomize.py` — a base interpreter (e.g. Homebrew python) often ships its
# own stdlib `sitecustomize.py` that precedes the venv on sys.path and would
# shadow ours, silently disabling the shim.
_API_LOG_SHIM = r'''
import os as _os
_banter_log = _os.environ.get("BANTER_API_LOG")
if _banter_log:
    try:
        import json as _json, time as _time, threading as _thr, sys as _sys
        from urllib.parse import urlsplit as _urlsplit
        import requests as _rq
        _banter_lock = _thr.Lock()
        _banter_orig_send = _rq.Session.send
        _banter_roots = set(
            p for p in (_os.environ.get("BANTER_IFACE_SDK") or "").split(",") if p
        )
        def _banter_origin():
            # Attribute to the interface whose code is on the stack. `.mcp`/`.cli`
            # subpackages of a first-party root win over the SDK client they wrap;
            # a request with no first-party frame is "other" (not via the interface).
            saw_sdk = saw_mcp = saw_cli = False
            f = _sys._getframe()
            while f is not None:
                parts = (f.f_globals.get("__name__") or "").split(".")
                if parts and parts[0] in _banter_roots:
                    saw_sdk = True
                    sub = parts[1] if len(parts) > 1 else ""
                    if sub == "mcp":
                        saw_mcp = True
                    elif sub == "cli":
                        saw_cli = True
                f = f.f_back
            if saw_mcp:
                return "mcp"
            if saw_cli:
                return "cli"
            return "sdk" if saw_sdk else "other"
        def _banter_send(self, request, **kw):
            try:
                with _banter_lock, open(_banter_log, "a") as _f:
                    _f.write(_json.dumps({
                        "ts": _time.time(),
                        "method": (getattr(request, "method", "") or "").upper(),
                        "path": _urlsplit(getattr(request, "url", "") or "").path,
                        "src": _banter_origin(),
                    }) + "\n")
            except Exception:
                pass
            return _banter_orig_send(self, request, **kw)
        _rq.Session.send = _banter_send
    except Exception:
        pass
'''


def _first_party_roots(venv_dir: Path) -> list[str]:
    """First-party package roots of the interface dist (for api-log `src`
    attribution), read from the installed wheel's `top_level.txt`.

    The interface is the only local-wheel dist that ships an `mcp` and/or `cli`
    subpackage in-tree (one wheel backs SDK, CLI, and MCP — only the entry point
    differs). So we pick the dist-info whose top-level packages expose those
    subpackages and return ALL of its top-level packages. Empty if not found
    (shim then tags "other")."""
    for sp in (venv_dir / "lib").glob("python*/site-packages"):
        for tl in sp.glob("*.dist-info/top_level.txt"):
            try:
                pkgs = tl.read_text().split()
            except OSError:
                continue
            for pkg in pkgs:
                if (sp / pkg / "mcp").is_dir() or (sp / pkg / "cli").is_dir():
                    return pkgs
    return []


def _install_api_log_shim(venv_dir: Path) -> None:
    """Bootstrap the request-logging shim into the run venv's OWN site-packages
    (not the shared base venv) so only the agent's python is instrumented.

    Uses a `.pth` import line, not `sitecustomize.py`: `site` runs `.pth` import
    lines for every site dir, so this fires even when the base interpreter ships
    its own `sitecustomize.py` (which would shadow a venv-local one). Best-effort."""
    for sp in (venv_dir / "lib").glob("python*/site-packages"):
        try:
            (sp / "_banter_apilog.py").write_text(_API_LOG_SHIM)
            (sp / "_banter_apilog.pth").write_text("import _banter_apilog\n")
        except OSError:
            pass


def deny_patterns_for(boundary: Path) -> list[str]:
    """Build `permissions.deny` patterns that DO fire under `bypassPermissions`.

    Covers Bash escape patterns (cd .., cat/ls dotfiles in $HOME) — the ones the
    agent can't bypass. Read/Write/Edit `file_path` patterns are silently
    skipped in bypass mode, so those are enforced by the PreToolUse hook instead
    (see `hooks/log_tool_call.py`).
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
# CLAUDE_CODE_OAUTH_TOKEN from child env, so a Bash → `banter run` → agent
# chain loses it. Callers can write the token to <run>/.claude-oauth (mode 0600)
# and point BANTER_TOKEN_CACHE at it; custom BANTER_* vars survive the hop.
TOKEN_CACHE_FILENAME = ".claude-oauth"
TOKEN_CACHE_ENV = "BANTER_TOKEN_CACHE"


def write_token_cache(token: str, run_path: Path) -> Path:
    """Write the OAuth token to ``run_path/.claude-oauth`` (mode 0600).

    Returns the cache path. Caller must forward it via the ``BANTER_TOKEN_CACHE``
    env var so child banter invocations find it.
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

    Order: env (a deliberate override — point CLAUDE_CODE_OAUTH_TOKEN in .env
    at a LONG-LIVED `claude setup-token` token for headless sessions), then
    Keychain (the freshest short-lived credential — an interactive claude
    refreshes it there), then the BANTER_TOKEN_CACHE file (a session-start
    snapshot; STALEST source, so last — preferring it over Keychain once froze
    an expired token for ~30 runs). Returns the token string or None.
    """
    return (os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
            or oauth_token_from_keychain()
            or read_token_cache())


# Rate-limit / transient-error retry budget. On a matching exit-error we exp-
# backoff up to RATE_LIMIT_MAX_BACKOFF_S and retry until total wall-clock reaches
# RATE_LIMIT_RETRY_WINDOW_S, then give up.
RATE_LIMIT_RETRY_WINDOW_S = 6 * 3600
RATE_LIMIT_BASE_BACKOFF_S = 2
RATE_LIMIT_MAX_BACKOFF_S = 3600

# Anthropic rate limits (429/529, "overloaded") + transient 5xx. Without the 5xx
# tokens, a mid-session 500 burns a whole run on retry.
_RATE_LIMIT_TOKENS = (
    "rate_limit", "rate limit", "overloaded",
    "429", "529",
    "500", "502", "503", "504",
    "internal server error", "bad gateway", "service unavailable", "gateway timeout",
)

# Auth failures. The injected CLAUDE_CODE_OAUTH_TOKEN can expire mid-session on a
# long run (a multi-hour session once died on a 401). On these we re-pull a
# fresh token from the Keychain (Claude Code refreshes it there) and retry a
# bounded number of times — distinct from the rate-limit backoff loop.
_AUTH_ERROR_TOKENS = (
    "401", "invalid authentication", "authentication_error",
    "unauthorized", "invalid x-api-key", "invalid bearer token", "oauth token",
)
# How many times to refresh-and-retry on an auth error before giving up.
AUTH_RETRY_MAX_ATTEMPTS = 3


@dataclass
class ClaudeResult:
    exit_code: int
    wall_time_s: float
    transcript_path: Path
    stderr_path: Path
    # Seconds slept in rate-limit back-off during this run. Included in
    # `wall_time_s`; subtract to get compute time (waiting != computing).
    rate_limit_wait_s: float = 0.0
    # Per-tool-call durations ({tool_name, tool_input, seconds}), stamped live
    # by `ToolTimer` as stream lines arrive. `results.split_tool_time`
    # classifies them into the CSV's platform_time_s / local_time_s split.
    # Empty for the codex engine (its raw events aren't span-timed yet).
    tool_spans: list[dict[str, Any]] = field(default_factory=list)


class ToolTimer:
    """Per-tool-call durations from a live stream-json feed.

    The stream-json events carry NO timestamps, so the only clock is line
    ARRIVAL time: a `tool_use` block (assistant event) opens a span; the
    matching `tool_result` (by tool_use_id, in a later user event) closes it.
    `claude -p` blocks on each tool synchronously, so arrival deltas track
    execution time faithfully (plus negligible harness overhead).

    A `result` event ends one `claude -p` attempt: any still-open spans close
    there, so the rate-limit back-off sleep BETWEEN retry attempts never
    inflates a span. `finalize()` closes spans still open at process death
    (e.g. a wall-clock kill mid tool call) at the current clock.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._open: dict[str, dict[str, Any]] = {}
        self.spans: list[dict[str, Any]] = []

    def _close(self, span: dict[str, Any], now: float) -> None:
        span["seconds"] = round(now - span.pop("start"), 3)
        self.spans.append(span)

    def observe(self, raw_line: str) -> None:
        now = self._clock()
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            return
        etype = event.get("type")
        if etype == "assistant":
            for block in (event.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use" \
                        and block.get("id"):
                    self._open[block["id"]] = {
                        "tool_name": block.get("name", ""),
                        "tool_input": block.get("input") or {},
                        "start": now,
                    }
        elif etype == "user":
            for block in (event.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    span = self._open.pop(block.get("tool_use_id"), None)
                    if span is not None:
                        self._close(span, now)
        elif etype == "result":
            for span in self._open.values():
                self._close(span, now)
            self._open.clear()

    def finalize(self) -> list[dict[str, Any]]:
        if self._open:
            now = self._clock()
            for span in self._open.values():
                self._close(span, now)
            self._open.clear()
        return self.spans


# Baseline outbound hosts every agent needs regardless of platform: Claude
# itself (api.anthropic.com) + loopback for any local platform server. Platform-
# specific hosts come from each `configs/platforms/<platform>/<interface>.yaml`
# via the `allowed_domains:` key.
BASE_ALLOWED_DOMAINS = (
    "127.0.0.1", "localhost",
    "api.anthropic.com", "*.anthropic.com",
    "statsig.anthropic.com", "console.anthropic.com",
)


def _write_settings(
    run_dir: Path,
    command_log: Path,
    allowed_domains: list[str] | None = None,
    sandbox_excluded_commands: list[str] | None = None,
) -> None:
    """Write the agent's project-scoped `.claude/settings.json`.

    Confinement: Claude Code's built-in sandbox kernel-confines Bash subprocesses
    to the boundary (the run dir). `permissions.deny` adds tool-name and Bash
    escape denies. Read/Write/Edit are NOT kernel-confined in bypassPermissions —
    soft (prompt + cwd) only.

    `allowLocalBinding` covers platform servers bound on loopback. `.claude/`
    state lands inside the boundary via the HOME redirect in :func:`run`.
    """
    # Agent's world = its run folder (`run_dir`); each run is a
    # separate, isolated agent invocation. Everything it needs lives inside the
    # boundary: venv (`runner._make_venv`), task data (`evals_provider.prepare`),
    # and the PreToolUse hook script (copied below).
    boundary = run_dir.resolve()
    settings_dir = run_dir / ".claude"
    settings_dir.mkdir(parents=True, exist_ok=True)
    # Copy the hook script into the boundary so the PreToolUse hook can invoke it
    # without reading outside the sandbox.
    hooks_dir = settings_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    local_hook = hooks_dir / HOOK_SCRIPT.name
    shutil.copy(HOOK_SCRIPT, local_hook)
    local_hook.chmod(0o755)
    # Sandbox: NETWORK gated, filesystem FULLY OPEN. Enabling the sandbox triggers
    # default filesystem restrictions even without a `filesystem` block, so we
    # explicitly opt back out with allowRead/Write `["/"]`, then layer just the
    # network allowlist. Result:
    #   - Outbound: only baseline (Claude API + loopback) + the platform's
    #     `allowed_domains:`. No pypi / HF / github / arbitrary HTTPS.
    #   - Filesystem: unrestricted. Confinement falls to the PreToolUse hook
    #     (path-based Read/Write/Edit deny, enforced regardless of permission
    #     mode) + `permissions.deny` (Bash escapes + tool-name denies).
    domain_list = list(BASE_ALLOWED_DOMAINS) + list(allowed_domains or [])
    # Go binaries (e.g. the databricks CLI) cannot verify TLS inside Seatbelt
    # (trustd unreachable → `x509: OSStatus -26276`); the documented fix is to
    # run them OUTSIDE the sandbox via `excludedCommands`. An excluded command
    # bypasses the domain allowlist, so manifests list only binaries that pin
    # their own endpoint; the PreToolUse hook + permissions.deny still apply.
    excluded = list(sandbox_excluded_commands or [])
    settings = {
        "hooks": {
            "PreToolUse": [{
                "matcher": ".*",  # regex over tool names
                # ABSOLUTE path: claude resolves hook commands from its own cwd
                # (the run dir), so a relative run_dir (e.g. the CLI single-run
                # form's default --runs-root) would double the path and block
                # every tool call.
                "hooks": [{"type": "command", "command": f"python3 {local_hook.resolve()}"}],
            }]
        },
        "sandbox": {
            "enabled": True,
            "autoAllowBashIfSandboxed": True,
            "excludedCommands": excluded,
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
            "deny": list(COMMON_DENY) + list(AGENT_ONLY_DENY) + deny_patterns_for(boundary),
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
    cli_subcommand: str | None = None,
    timeout_s: int | None = 60 * 60,   # None → NO wall-clock cap
    extra_env: dict[str, str] | None = None,
    allowed_domains: list[str] | None = None,
    interface: str | None = None,
    compute_deny: list[str] | None = None,
    instance_allowlist: list[str] | None = None,
    sandbox_excluded_commands: list[str] | None = None,
    platform: str | None = None,
) -> ClaudeResult:
    """Spawn `claude -p` (the agent).

    Always forwards a Keychain OAuth token (when present) since the redirected
    HOME has no on-disk credentials. `auth="login"` additionally strips
    ANTHROPIC_API_KEY so OAuth wins; `auth="api-key"` keeps both and lets
    claude-code pick. `extra_env` layers platform credentials.
    """
    if shutil.which("claude") is None:
        raise RuntimeError("`claude` CLI not found on PATH. Install Claude Code first.")

    _write_settings(run_dir, command_log,
                    allowed_domains=allowed_domains,
                    sandbox_excluded_commands=sandbox_excluded_commands)
    mcp_cfg = _write_mcp_config(run_dir, mcp_servers)

    transcript_path = run_dir / "transcript.jsonl"
    stderr_path = run_dir / "claude.stderr.log"
    command_log.touch()

    env = os.environ.copy()
    env.pop("ANTHROPIC_BASE_URL", None)  # strip any parent-shell proxy
    # HOME → boundary so `.claude/` state lands inside the run dir.
    boundary = run_dir.resolve()
    env["HOME"] = str(boundary)
    # Auth: the redirected HOME has no credentials, and claude cannot reach
    # the Keychain credential from a redirected HOME (verified: "Not logged
    # in" even with the host ~/.claude.json copied in) — so an env token is
    # REQUIRED. resolve_oauth_token() picks, in order: a deliberate env
    # override (set CLAUDE_CODE_OAUTH_TOKEN in .env to a long-lived
    # `claude setup-token` token — the robust choice for headless sessions),
    # the Keychain (freshest short-lived credential, re-read EVERY run so a
    # mid-session refresh is picked up), then the session-start cache file.
    # Expiry of a short-lived token mid-run is handled by the 401 retry below.
    token = resolve_oauth_token()
    if token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    if auth == "login":
        env.pop("ANTHROPIC_API_KEY", None)
    if not env.get("ANTHROPIC_API_KEY") and not env.get("CLAUDE_CODE_OAUTH_TOKEN"):
        print("[banter] WARNING: no auth available (no ANTHROPIC_API_KEY, "
              "no token in env/cache/Keychain). Agent will fail. Re-run "
              "banter from a shell where `claude /login` has been run.",
              flush=True)
    env["ANTHROPIC_MODEL"] = model
    env["TESTBED_COMMAND_LOG"] = str(command_log)
    # PreToolUse hook reads this to enforce Read/Write/Edit path denies — silently
    # skipped under bypassPermissions, but the hook fires regardless and rejects
    # calls outside the boundary.
    env["TESTBED_BOUNDARY"] = str(boundary)
    if cli_binary:
        env["TESTBED_CLI_BINARY"] = cli_binary
    if cli_subcommand:
        env["TESTBED_CLI_SUBCOMMAND"] = cli_subcommand
    if sdk_module:
        env["TESTBED_SDK_MODULE"] = sdk_module
    # Platform name for the hook's denial messages ("Training must run on
    # <platform>") — the hook is platform-agnostic and must not hardcode one.
    if platform and platform != "none":
        env["TESTBED_PLATFORM"] = platform
    # Remote-only enforcement, only for the real delegation interfaces. The hook
    # needs the active interface (so a non-active interface's use is an escape) plus
    # the compute libraries whose LOCAL execution is forbidden (training must run
    # remotely). A none/none baseline trains locally by design, so the hook no-ops.
    if interface in ("cli", "mcp", "sdk"):
        env["TESTBED_INTERFACE"] = interface
        env["TESTBED_COMPUTE_DENY"] = ",".join(compute_deny or DEFAULT_COMPUTE_DENY)
    # Cost control (manifest `instance_allowlist`, e.g. the AWS Free Tier set):
    # the hook denies any other `ml.<family>.<size>` token in a tool call.
    if instance_allowlist:
        env["TESTBED_INSTANCE_ALLOW"] = ",".join(instance_allowlist)

    # Activate per-run venv so claude's python/pip find ML deps via .pth.
    venv_bin = run_dir / "venv" / "bin"
    if venv_bin.is_dir():
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(run_dir / "venv")
        # REST-endpoint coverage: log every outbound request from the agent's
        # SDK/CLI/MCP (all use `requests` in this venv) to api_calls.jsonl, which
        # `results.endpoint_hits` scores against the configured white/blacklist.
        env["BANTER_API_LOG"] = str(run_dir / "api_calls.jsonl")
        # First-party package roots → the shim attributes each call's `src`
        # (mcp/cli/sdk/other) so coverage counts only calls made THROUGH the
        # interface under test, not hand-rolled `requests`.
        roots = _first_party_roots(run_dir / "venv")
        if roots:
            env["BANTER_IFACE_SDK"] = ",".join(roots)
        elif interface in ("cli", "mcp", "sdk"):
            # No installed dist ships an mcp/cli subpackage → the shim can't
            # attribute calls and tags everything "other", silently zeroing
            # whitelist_hits. Warn rather than fail quietly so a misbuilt venv
            # doesn't look like an agent that simply never used the interface.
            print(f"[banter] WARNING: no first-party interface roots in "
                  f"{run_dir / 'venv'}; API calls will all be attributed 'other' "
                  f"and whitelist_hits will be 0 for interface {interface!r}.",
                  flush=True)
        _install_api_log_shim(run_dir / "venv")

    # Skip pip cache (per-run venv inherits via .pth; cache dir unwritable).
    env["PIP_NO_CACHE_DIR"] = "1"

    # Keep long foreground commands (e.g. a remote training job the agent waits
    # on) SYNCHRONOUS. Two SEPARATE Claude Code mechanisms matter:
    #   1) Auto-backgrounding — a long foreground Bash command is auto-moved to a
    #      background task. In `-p` mode its completion NEVER re-invokes the model,
    #      so the agent blind-polls a frozen task-output file and wedges.
    #      BASH_MAX_TIMEOUT_MS does NOT control this (only the KILL timeout); the
    #      only reliable switch is CLAUDE_CODE_DISABLE_BACKGROUND_TASKS.
    #   2) Kill timeout — how long a foreground command may run before being killed.
    #      Raise both Bash timeouts to the whole `claude -p` budget so a long job
    #      runs to completion instead of being killed.
    # We want NO backgrounding at all (the PreToolUse hook also denies the explicit
    # `run_in_background` flag), so disable it outright.
    env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] = "1"
    # Uncapped run (timeout_s=None): the Bash-tool env still needs a number —
    # use 7 days, far beyond any real agent command.
    bash_timeout_ms = str(int(timeout_s if timeout_s is not None else 7 * 86400) * 1000)
    env["BASH_DEFAULT_TIMEOUT_MS"] = bash_timeout_ms
    env["BASH_MAX_TIMEOUT_MS"] = bash_timeout_ms

    # MCP startup: Claude Code kills stdio MCP servers that take >30s (default
    # MCP_TIMEOUT) to complete the handshake. The databricks UC server needs
    # ~20s of imports BEFORE it even starts enumerating UC functions over REST,
    # so it never connects and the agent sees zero mcp__* tools. Give startup
    # a generous fixed window; pin per-tool-call time to the run budget like
    # Bash (an MCP tool call can legitimately wait on a remote job).
    env["MCP_TIMEOUT"] = str(180 * 1000)
    env["MCP_TOOL_TIMEOUT"] = bash_timeout_ms

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

    # `runner.run` wraps this in `streaming.tee_to` for agent.log; the full
    # stream-json still lands in transcript.jsonl, which `runner.run` mines for
    # usage + commands then discards at teardown (not kept per run).
    # The ToolTimer rides the same live feed: events carry no timestamps, so
    # tool-call durations exist ONLY as arrival-time spans stamped here.
    timer = ToolTimer()
    printer = streaming.make_printer("agent")

    def _on_line(raw_line: str) -> None:
        timer.observe(raw_line)
        printer(raw_line)

    exit_code, wall, rl_wait = run_with_retry(
        cmd=cmd,
        cwd=run_dir,
        env=env,
        transcript_path=transcript_path,
        stderr_path=stderr_path,
        timeout_s=timeout_s,
        on_line=_on_line,
        log_prefix="banter",
    )

    # `run_with_retry` only writes stderr_path when there's actual stderr output,
    # so no empty-file cleanup is needed here.

    return ClaudeResult(
        exit_code=exit_code,
        wall_time_s=wall,
        transcript_path=transcript_path,
        stderr_path=stderr_path,
        rate_limit_wait_s=rl_wait,
        tool_spans=timer.finalize(),
    )


def run_with_retry(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    transcript_path: Path,
    stderr_path: Path,
    timeout_s: int | None = None,
    on_line: Callable[[str], None] | None = None,
    log_prefix: str = "banter",
) -> tuple[int, float, float]:
    """Run a `claude -p` subprocess with exponential rate-limit back-off.

    - Appends stdout to `transcript_path` (stream-json). Stderr is buffered to a
      temp file and PROMOTED to `stderr_path` only if non-empty, keeping the run
      dir clean when nothing went wrong.
    - When `on_line` is given, stdout is piped through and each line forwarded to
      the callback (live streaming display).
    - On rate-limit / 5xx detection in the last `result`, sleeps
      `min(2 * 2^(attempt-1), 3600)`s and retries until the 6h total budget is
      exhausted.

    Returns `(exit_code, total_wall_time_s, rate_limit_wait_s)`. Wall time
    includes sleep time; `rate_limit_wait_s` is the slept portion, so callers
    can report compute time as `wall - wait` (it feeds the CSV's
    `rate_limit_wait_s` column).
    """
    import tempfile

    start = time.monotonic()
    attempt = 0
    auth_attempts = 0
    rl_attempts = 0   # rate-limit retries only — drives the backoff exponent
    rl_wait_s = 0.0   # total back-off sleep — returned so wall − wait = compute
    exit_code = 1
    # Buffer stderr to a temp file outside the run dir; moved into place at the
    # END only if it has actual content.
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

        if exit_code == 0:
            break
        # Auth failure (e.g. OAuth token expired mid-session): re-pull a fresh
        # Keychain token and retry, up to a small bound. Checked before rate-limit
        # so a 401 doesn't fall into the 6h backoff loop.
        if _last_result_is_auth_error(transcript_path) and auth_attempts < AUTH_RETRY_MAX_ATTEMPTS:
            auth_attempts += 1
            # Only retry if a DIFFERENT (refreshed) token was obtained — retrying
            # the identical expired token would just fail again immediately.
            if not _refresh_oauth_token(env):
                print(
                    f"[{log_prefix}] auth error on attempt {attempt}; no fresh "
                    f"Keychain token available — giving up.",
                    flush=True,
                )
                break
            print(
                f"[{log_prefix}] auth error on attempt {attempt} "
                f"(auth-retry {auth_attempts}/{AUTH_RETRY_MAX_ATTEMPTS}); re-pulled a "
                f"fresh Keychain token, retrying.",
                flush=True,
            )
            continue
        if not _last_result_is_rate_limited(transcript_path):
            break
        rl_attempts += 1
        elapsed = time.monotonic() - start
        backoff = min(
            RATE_LIMIT_BASE_BACKOFF_S * (2 ** (rl_attempts - 1)),
            RATE_LIMIT_MAX_BACKOFF_S,
        )
        if elapsed + backoff > RATE_LIMIT_RETRY_WINDOW_S:
            print(
                f"[{log_prefix}] rate-limited after {rl_attempts} attempts "
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
        rl_wait_s += backoff

    # Promote the temp stderr buffer to `stderr_path` only if non-empty; otherwise
    # drop it so the run dir doesn't carry an empty log file.
    try:
        if err_tmp_path.exists() and err_tmp_path.stat().st_size > 0:
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            err_tmp_path.replace(stderr_path)
        else:
            err_tmp_path.unlink(missing_ok=True)
    except OSError:
        pass

    return exit_code, time.monotonic() - start, rl_wait_s


def _last_result_error_blob(transcript_path: Path) -> str | None:
    """Lowercased error text of the last `result` event when it is an error, else
    None. Shared by the rate-limit and auth-error classifiers."""
    if not transcript_path.exists():
        return None
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
        return None
    return " ".join(
        str(last_result.get(k) or "") for k in ("api_error_status", "subtype", "result")
    ).lower()


def _last_result_is_rate_limited(transcript_path: Path) -> bool:
    """The last `result` stopped on a 429/529-style / transient-5xx condition."""
    blob = _last_result_error_blob(transcript_path)
    return bool(blob) and any(t in blob for t in _RATE_LIMIT_TOKENS)


def _last_result_is_auth_error(transcript_path: Path) -> bool:
    """The last `result` stopped on a 401/authentication condition."""
    blob = _last_result_error_blob(transcript_path)
    return bool(blob) and any(t in blob for t in _AUTH_ERROR_TOKENS)


def _refresh_oauth_token(env: dict[str, str]) -> bool:
    """Re-pull the OAuth token from the Keychain (Claude Code refreshes it there)
    and update both the env and the BANTER_TOKEN_CACHE file in place. Returns True
    only if a DIFFERENT token was obtained — retrying with the identical (still-
    expired) token is pointless, so the caller stops then."""
    fresh = oauth_token_from_keychain()
    if not fresh or fresh == env.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return False
    env["CLAUDE_CODE_OAUTH_TOKEN"] = fresh
    cache = env.get(TOKEN_CACHE_ENV)
    if cache:
        try:
            p = Path(cache)
            p.write_text(fresh)
            p.chmod(0o600)
        except OSError:
            pass
    return True
