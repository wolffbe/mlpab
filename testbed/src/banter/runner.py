"""Orchestrates a single (task, platform, interface, skills) agent run.

Layout under the caller-supplied `runs_root`:
    <runs_root>/<category>/<task>/      # one agent run; venv torn down at end

Each task run folder contains:
    prompt.txt            # exact task prompt handed to claude -p
    venv/                 # per-run agent venv (deleted at teardown)
    data/                 # staged inputs of the generated eval instance
    submission/           # local deliverables (platform `none` baseline only)
    agent.log          # live formatted view (tee captures fd 1/2 of the run)
    commands.jsonl        # one line per tool call (cli/mcp/sdk/python/bash/skill/...)
    grading.json          # assertion-suite report (evals_provider.grade)
    .claude/settings.json # PreToolUse hook for claude -p
    .claude/skills/       # copied skill bundle (deleted at teardown)   [skills != none]
    .mcp.json             # MCP servers                                  [interface == mcp]

The generated instance (incl. the private answer key) lives in a SIBLING
`.<task>.private/` dir — outside the agent's sandbox boundary.
Login (`auth_command`) is verified per run.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from banter import (
    claude_runner,
    codex_runner,
    evals_provider,
    interfaces,
    preflight as preflight_mod,
    results,
    skills as skills_mod,
    streaming,
)


DEFAULT_MODEL = "claude-sonnet-4-6"
AUTH_MODES = ("api-key", "login")

# Testbed repo root.
TESTBED_ROOT = Path(__file__).resolve().parents[2]
# The managed base venv (testbed/.venv). `banter` itself runs outside any venv;
# each agent run gets its own venv nested on this one (sharing its libraries).
BASE_VENV = TESTBED_ROOT / ".venv"


@dataclass
class RunSpec:
    task: str                       # FTI sub-task (an evals_provider family)
    platform: str                   # e.g. "hopsworks", "none"
    interface: str                  # interface: cli/mcp/sdk/none
    skills: str = "none"            # platform skill bundle name or "none"
    category: str = "no_task"       # FTI category (stage) this task belongs to
    model: str = DEFAULT_MODEL
    auth: str = "api-key"
    timeout_s: int | None = 60 * 60   # None → NO per-run wall-clock cap
    runs_root: Path = Path("results")
    # Session tag: the treatment config's name, stored as the `run` column.
    run_id: str | None = None
    # Repeat-attempt number: run_dir gains a trailing /<attempt> segment so
    # repeats of one config nest side by side (…/<task>/1, /2, …). Also
    # feeds the instance seed, so every repeat gets a FRESH eval instance.
    attempt: int | None = None
    # The single global results CSV the row is appended to
    # (results/results.csv). None → derived from `runs_root` via the
    # nearest `results/` ancestor.
    results_csv: Path | None = None
    # Run the upfront preflight (platform install/login + skill access probe).
    # Batch runners set False after one shared preflight over the union.
    preflight: bool = True


def _results_root_from(runs_root: Path) -> Path:
    """Nearest `results/` ancestor of `runs_root` (…/results/<config>/…).
    Falls back to `runs_root` if none found."""
    runs_root = Path(runs_root)
    for p in (runs_root, *runs_root.parents):
        if p.name == "results":
            return p
    return runs_root


def _make_venv(target: Path) -> Path:
    """Create the agent's per-run venv FULLY MATERIALIZED inside the run dir.

    The base venv's (testbed/.venv) site-packages are copied entirely INSIDE
    the agent's run dir — no `.pth` indirection to an outside path — so
    the agent's sandbox is truly bounded to its run folder.

    On APFS (macOS default) `cp -Rc` does copy-on-write clones, so the 2-3 GB
    shared site-packages costs near-zero disk until the agent modifies a page
    (pip install of the interface). Falls back to a recursive copy where APFS
    clones aren't available (Linux, non-APFS volumes).
    """
    py = target / "bin" / "python"
    if py.exists():
        return py

    base_venv_py = BASE_VENV / "bin" / "python"
    base_py = base_venv_py if base_venv_py.exists() else Path(sys.executable)
    subprocess.run(
        [str(base_py), "-m", "venv", "--system-site-packages", str(target)],
        check=True,
    )
    # Materialize the base venv's site-packages into the agent venv: its
    # sandbox confines reads to the run dir, so any outside `.pth` reference
    # would resolve to a path Seatbelt denies.
    if base_venv_py.exists():
        eng_sp = next((target / "lib").glob("python*/site-packages"), None)
        base_sp = next((BASE_VENV / "lib").glob("python*/site-packages"), None)
        if eng_sp and base_sp:
            _clone_tree(base_sp, eng_sp)
    return py


def _clone_tree(src: Path, dst: Path) -> None:
    """APFS-clone (or copy) every entry in `src` into existing `dst`.

    `cp -Rc` for APFS clone (near-zero cost on macOS); falls back to `cp -R` where
    clones are unsupported. `dst` already exists (created by venv), so children
    are merged in without re-creating it.
    """
    for child in src.iterdir():
        target = dst / child.name
        if target.exists():
            continue
        try:
            subprocess.run(["cp", "-Rc", str(child), str(target)], check=True)
        except subprocess.CalledProcessError:
            subprocess.run(["cp", "-R", str(child), str(target)], check=True)


def _run_aux(steps: list[str], run_dir: Path, env: dict[str, str], timeout: int = 60) -> None:
    """Best-effort platform housekeeping (serve/teardown). Failures ignored and
    output discarded — not part of the agent's work.

    Setup/teardown scripts run with the per-run venv python, whose site-packages
    carry the request-logging shim — so the api-log env vars are stripped here.
    Their REST traffic (project create/delete, …) is platform plumbing and must
    never become api_calls.jsonl entries scored as endpoint coverage. The
    agent's wall_time_s never sees these phases either (it is timed around the
    claude subprocess only).
    """
    aux_env = {k: v for k, v in env.items()
               if k not in ("BANTER_API_LOG", "BANTER_IFACE_SDK")}
    for cmd in steps:
        try:
            subprocess.run(
                cmd, shell=True, cwd=str(run_dir), env=aux_env,
                timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


def _platform_env(run_dir: Path) -> dict[str, str]:
    """Env the platform setup step exported for the AGENT, then consume the file.

    Setup scripts run as throwaway subprocesses (`serve:` via `_run_aux`), so they
    cannot literally export env vars into the agent's process. Instead they may
    append KEY=VALUE lines to $BANTER_PLATFORM_ENV (<run_dir>/platform.env) for
    values only the platform side can know (e.g. the SageMaker execution-role ARN
    setup.py just created). The file is deleted after parsing so it never lingers
    in the agent's workdir; malformed lines are skipped (best-effort, like the
    step that wrote them). Keys declared in the manifest (resolved from .env) take
    precedence over these — see the merge at the call site.
    """
    path = run_dir / "platform.env"
    out: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip():
                # Same dotenv dialect as cli._load_dotenv: surrounding quotes
                # are syntax, not value — KEY="x" must mean the same thing in
                # platform.env as it does in .env.
                out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        return {}
    path.unlink(missing_ok=True)
    return out


TASK_PROMPT_PATH = TESTBED_ROOT / "prompts" / "agent.md"


def _total_ram_gb() -> float | None:
    """Total physical RAM in GiB, or None if undeterminable.

    Uses POSIX sysconf (macOS and Linux); no third-party deps.
    """
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3)
    except (ValueError, OSError, AttributeError):
        return None


def detect_environment() -> str:
    """Detect the agent's hardware/OS and render it as prompt bullet lines.

    Probed on the host — the same machine the per-run venv (and thus the agent)
    runs on — so the facts apply to the agent. Avoids importing torch (not yet
    installed at prompt-build time): accelerator availability is inferred from
    OS/arch and `nvidia-smi` presence. The `spawn`-deadlock warning is emitted only
    where the multiprocessing default is `spawn` (macOS/Windows), as the unguarded
    `num_workers>0` pattern hangs forever there.
    """
    import platform
    import shutil

    system = platform.system()                  # Darwin / Linux / Windows
    machine = platform.machine() or "?"         # arm64 / x86_64 / ...
    cpus = os.cpu_count() or 1
    ram = _total_ram_gb()
    ram_str = f", ~{ram:.0f} GB RAM" if ram else ""

    if shutil.which("nvidia-smi") is not None:
        accel = "An NVIDIA GPU is available — you may use CUDA."
    elif system == "Darwin" and machine in ("arm64", "aarch64"):
        accel = "Apple Silicon: no CUDA GPU. `torch` MPS is likely available; otherwise use CPU."
    else:
        accel = "No CUDA GPU detected — train on CPU."

    spawn = system in ("Darwin", "Windows")
    start_method = "spawn" if spawn else "fork"

    lines = [
        f"- You run on {system} ({machine}) with {cpus} CPU cores{ram_str}.",
        f"- {accel}",
        f"- Python multiprocessing uses the '{start_method}' start method.",
    ]
    if spawn:
        lines.append(
            "- Because the start method is 'spawn', a `DataLoader` with `num_workers>0` "
            "(or any multiprocessing) whose script lacks an `if __name__ == \"__main__\":` "
            "guard will DEADLOCK and hang forever — it never errors, it just stalls. "
            "Default to `num_workers=0`; only use workers if all top-level code is under "
            "that guard."
        )
    return "\n".join(lines)


# The "platform is under test" rule (agent default), baked into agent.md
# between these markers. Applies only when a platform is present; for none/none
# (agent builds its own model freely) the whole section is stripped.
_UNDER_TEST_RE = re.compile(
    r"<!--UNDER_TEST_START-->\n(.*?)\n<!--UNDER_TEST_END-->", re.DOTALL
)
# Inverse: content kept ONLY when NO interface is under test (none/none, agent
# trains its own model locally). Stripped when an interface is under test
# (everything must run remotely on the platform).
_LOCAL_ONLY_RE = re.compile(
    r"<!--LOCAL_ONLY_START-->\n(.*?)\n<!--LOCAL_ONLY_END-->", re.DOTALL
)


def _build_prompt(
    task_body: str,
    fragment: str,
    interface_under_test: bool = True,
) -> str:
    template = TASK_PROMPT_PATH.read_text()
    if interface_under_test:
        template = _UNDER_TEST_RE.sub(lambda m: m.group(1), template)
        template = _LOCAL_ONLY_RE.sub("", template)
    else:
        template = _UNDER_TEST_RE.sub("", template)
        template = _LOCAL_ONLY_RE.sub(lambda m: m.group(1), template)
    template = re.sub(r"\n{3,}", "\n\n", template)
    # The agent never receives platform docs — its only guide is the
    # interface's own self-description.
    return template.format(
        task_body=task_body.strip(),
        fragment=fragment,
        environment=detect_environment(),
    ).strip()


def run(spec: RunSpec) -> results.Row:
    if spec.auth not in AUTH_MODES:
        raise ValueError(f"Unknown auth mode {spec.auth!r}; expected one of {AUTH_MODES}")

    started = datetime.now(timezone.utc)

    # Fail fast: platform must be installed, login must work, and any chosen skill
    # bundle must be accessible to the agent. Batch runners
    # (treatments) preflight the union once and pass preflight=False
    # here to avoid re-probing per run.
    if spec.preflight:
        preflight_mod.check_run(
            platform=spec.platform,
            interface=spec.interface,
            skills=spec.skills,
            auth=spec.auth,
            model=spec.model,
        )

    # Fail fast: validate the platform/interface is known before doing work
    # (raises on unknown). The resolved ref/hash for the row come from
    # `interfaces.setup` below, not here.
    interfaces.variant_for(spec.platform, spec.interface)
    # Fail fast on unverified skill bundles — confirm the bundle is well-formed
    # before spending time on venv/data/prep.
    if spec.skills != "none":
        skills_mod.verify_installed(spec.platform, spec.skills)
    # Per-task output lives directly under runs_root:
    #   results/<config>/<model>/…/<category>/<task>/<n>/
    # The attempt folder holds exactly TWO dirs:
    #   task/      the agent's ENTIRE world — cwd + sandbox boundary: data,
    #              prompt, venv, submission, logs, .claude (everything below
    #              uses `run_dir` = this folder)
    #   solution/  the answer key + grading.json — a SIBLING outside the
    #              boundary, so the agent cannot see it
    # Platform/interface/skills/version are results.csv columns, not encoded in
    # the path (one platform per run, per the per-interface configs).
    attempt_dir = spec.runs_root / spec.category / spec.task
    if spec.attempt is not None:
        attempt_dir = attempt_dir / str(spec.attempt)
    # Re-runs overwrite the previous output.
    if attempt_dir.exists():
        shutil.rmtree(attempt_dir)
    run_dir = attempt_dir / "task"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Capture the ENTIRE terminal output of this run — venv build, instance
    # generation, pip installs, agent's streamed activity, grading — into
    # agent.log (FD-level tee). Echo to the real terminal unless quiet or
    # nested.
    passthrough = not streaming.nested() and not streaming.quiet()
    with streaming.tee_to(run_dir / "agent.log", passthrough=passthrough):
        # The per-run venv clones the base .venv. Ensure the base holds no copy of
        # the interface package before cloning, so this run installs the interface
        # wheel fresh + complete (console scripts, extras).
        interfaces.ensure_base_clean(spec.platform, spec.interface)
        venv_python = _make_venv(run_dir / "venv")
        (run_dir / "submission").mkdir(exist_ok=True)

        # Generate a FRESH eval instance (validity gates run inside the
        # generator) and stage its data/ — must succeed before spinning up
        # Claude. The seed is deterministic per (session, category, task,
        # attempt): reproducible rows, fresh instance every repeat.
        seed = evals_provider.seed_for(
            spec.run_id or "", spec.category, spec.task, spec.attempt or 1)
        task_body = evals_provider.prepare(spec.task, run_dir, seed)

        interface_setup = interfaces.setup(
            spec.platform, spec.interface, run_dir, venv_python,
        )
        skills_setup = skills_mod.apply(spec.platform, spec.skills, run_dir)
        if skills_setup.installed:
            print(
                f"[banter] skills ({skills_setup.hash}): "
                f"{','.join(skills_setup.installed)}",
                file=sys.stderr,
            )
        # The agent never receives platform docs. Its sole guide is the
        # interface's own self-description (names, `--help`, docstrings,
        # errors) — the interface's quality is what we measure, and agent-side
        # docs would confound that.

        # Start fresh: stop servers a previous (possibly crashed) run left running
        # and clear leaked state (incl. platform-side artifacts the agent created —
        # see the interface `teardown:` step), then start the servers THIS run
        # needs (e.g. the MCP HTTP server claude connects to at launch).
        # `BANTER_PLATFORM_DIR` points teardown/serve steps at the COMMITTED
        # configs/platforms/<platform>/ dir (e.g. a teardown.py cleanup script)
        # — cleanup logic is fixed infrastructure, immune to run-dir state.
        base_keys_env = {
            **os.environ,
            **interface_setup.keys,
            "BANTER_PLATFORM_DIR": str(interfaces.CONFIGS_DIR / spec.platform),
        }
        _run_aux(interface_setup.teardown, run_dir, base_keys_env)
        if interface_setup.serve:
            serve_env = dict(base_keys_env)
            serve_env["PATH"] = f"{run_dir / 'venv' / 'bin'}{os.pathsep}{serve_env.get('PATH', '')}"
            serve_env["VIRTUAL_ENV"] = str(run_dir / "venv")
            serve_env["BANTER_PLATFORM_ENV"] = str(run_dir / "platform.env")
            # Generous timeout: hopsworks setup.py retries project creation for
            # up to ~3 min while the backend finishes deleting the previous
            # run's namespace (teardown is async server-side).
            _run_aux(interface_setup.serve, run_dir, serve_env, timeout=300)
        # Env the setup step exported for the agent (e.g. a role ARN it just
        # created) — declared keys (.env) win on conflict, so an explicit .env
        # value always overrides what setup derived.
        agent_keys = {**_platform_env(run_dir), **interface_setup.keys}

        # Log in + verify auth for THIS run in its own venv. Build/test ran
        # once at treatment preflight; login is re-checked every run (catches
        # expired creds mid-session; each run authenticates in its own venv).
        login = interfaces.login_status(
            spec.platform, spec.interface, venv_python=venv_python, keys=agent_keys,
        )
        if not login.ok:
            raise preflight_mod.PreflightError(login.message)

        prompt = _build_prompt(task_body, interface_setup.prompt_fragment,
                               interface_under_test=interface_setup.interface != "none")
        (run_dir / "prompt.txt").write_text(prompt)

        # The agent is confined to its own run dir (the generated
        # instance's private answer key lives in a sibling dir it cannot read).
        # Engine dispatch: OpenAI models (`gpt-*` / `*codex*`) run on the Codex
        # CLI, everything else on Claude Code. Same signature + result shape;
        # codex runs skip the PreToolUse enforcement hook (no codex equivalent)
        # but keep full post-hoc accounting via the normalized transcript.
        engine = codex_runner if codex_runner.is_codex_model(spec.model) else claude_runner
        cr = engine.run(
            prompt=prompt,
            run_dir=run_dir,
            auth=spec.auth,
            model=spec.model,
            cli_binary=interface_setup.cli_binary,
            cli_subcommand=interface_setup.cli_subcommand,
            sdk_module=interface_setup.sdk_module,
            mcp_servers=interface_setup.mcp_servers,
            command_log=run_dir / "commands.jsonl",
            timeout_s=spec.timeout_s,
            extra_env=agent_keys,
            allowed_domains=interface_setup.allowed_domains,
            interface=interface_setup.interface,
            instance_allowlist=interface_setup.instance_allowlist,
            sandbox_excluded_commands=interface_setup.sandbox_excluded_commands,
            platform=spec.platform,
        )

        if cr.exit_code != 0:
            print(f"[banter] claude exit={cr.exit_code}", file=sys.stderr)

        # Grade by replaying the instance's assertion suite against the
        # platform's read paths (or the local deliverable for platform `none`).
        # Runs BEFORE venv teardown: the grader uses the run venv's python,
        # whose platform client was installed from the committed pinned wheel.
        try:
            grading = evals_provider.grade(
                spec.task, run_dir, spec.platform, venv_python)
        except Exception as e:
            grading = {"success": False, "asserts_passed": 0, "asserts_total": 0,
                       "asserts": [], "error": str(e)}
        # grading.json lives grader-side, next to the answer key.
        (attempt_dir / "solution" / "grading.json").write_text(json.dumps(grading, indent=2))

        usage = results.parse_transcript_usage(cr.transcript_path)
        # Counting classifies by the ACTIVE interface only: a `hops`/`import
        # hopsworks` call is `cli_calls`/`sdk_calls` ONLY when that interface is
        # under test. (interface_setup markers are resolved for ALL interfaces to
        # feed the cross-interface enforcement HOOK, but counting must not relabel
        # an off-interface call as on-interface.)
        count_cli = interface_setup.cli_binary if interface_setup.interface == "cli" else None
        count_cli_sub = interface_setup.cli_subcommand if interface_setup.interface == "cli" else None
        count_sdk = interface_setup.sdk_module if interface_setup.interface == "sdk" else None
        counts = results.aggregate_commands(
            cr.transcript_path,
            cli_binary=count_cli,
            sdk_module=count_sdk,
            run_dir=run_dir,
            cli_subcommand=count_cli_sub,
            # The hook's own log (pre-rebuild): its `denied: true` records are
            # the structured source for denied_calls.
            commands_log=run_dir / "commands.jsonl",
        )
        # Rebuild commands.jsonl from the transcript so it's always populated even
        # if the PreToolUse hook silently fails.
        results.write_commands_log(
            cr.transcript_path,
            run_dir / "commands.jsonl",
            cli_binary=count_cli,
            sdk_module=count_sdk,
            run_dir=run_dir,
            cli_subcommand=count_cli_sub,
        )
        # Surface the interface CLIENT's runtime logs + crashes into a per-run
        # <platform>_client.logs (cli/mcp/sdk). For mcp this rescues the stdio
        # server's stderr/connection status — a startup crash there otherwise only
        # reaches the agent as "No such tool available", reading like an empty
        # interface. The agent's HOME (where Claude buries the MCP logs) is the
        # run dir.
        if interface_setup.interface in ("cli", "mcp", "sdk"):
            client_log = results.collect_client_logs(
                run_dir=run_dir,
                boundary=run_dir,
                interface=interface_setup.interface,
                platform=spec.platform,
                mcp_servers=interface_setup.mcp_servers,
                transcript_path=cr.transcript_path,
            )
            if client_log.get("crashed"):
                print(
                    f"[banter] WARNING: {spec.platform} {interface_setup.interface} "
                    f"client crashed — see {client_log['path']} "
                    f"(markers: {', '.join(client_log['markers'])})",
                    file=sys.stderr,
                )
        # REST-endpoint coverage: no per-config endpoint policy anymore — the
        # whitelist/blacklist mechanism stays available but unconfigured.
        _wl, _bl = None, None
        # Attribute whitelist coverage to the interface under test: only calls made
        # THROUGH cli/mcp/sdk count — not hand-rolled `requests`, not server-side
        # Job calls (which never reach the venv shim). A none/none baseline has no
        # interface (and no endpoints), so it stays unattributed.
        _iface = interface_setup.interface if interface_setup.interface in ("cli", "mcp", "sdk") else None
        endpoint_cov = results.endpoint_coverage(run_dir / "api_calls.jsonl", _wl, _bl, interface=_iface)
        endpoint_counts = {
            "whitelist_hits": endpoint_cov["whitelist_hits"],
            "blacklist_hits": endpoint_cov["blacklist_hits"],
        }
        # Per-run coverage breakdown (covered/missed target endpoints) showing
        # WHICH lifecycle steps the agent reached — a missed
        # endpoint is a capability the interface must expose. Embeds the
        # `whitelist`/`blacklist` patterns it was scored against, so it stays
        # self-contained once the raw api_calls.jsonl is discarded below. Only when
        # configured.
        if _wl or _bl:
            (run_dir / "endpoint_coverage.json").write_text(json.dumps(endpoint_cov, indent=2))
        # Assertion-suite report → Row fields: the passed/total tallies
        # (all green = task solved). `deliverable_exists` = the deliverable
        # exists on the platform (first assert passed), separating "wrong" from
        # "absent". The full report (asserts, diagnostic) lives in grading.json.
        _asserts = grading.get("asserts") or []
        deliverable_exists = bool(_asserts) and bool(_asserts[0].get("passed"))
        slim_grading = {
            "asserts_passed": grading.get("asserts_passed", 0),
            "asserts_total": grading.get("asserts_total", 0),
        }
        if grading.get("asserts_total"):
            print(f"[banter] graded: {grading.get('asserts_passed', 0)}/"
                  f"{grading['asserts_total']} asserts"
                  + (f" — {grading['diagnostic']}" if grading.get("diagnostic") else ""),
                  flush=True)
        # Map the agent's `claude -p` usage into the Row's metric columns.
        # Wall metrics count COMPUTE time only: rate-limit back-off sleeps are
        # excluded (waiting != computing) and recorded separately in
        # `rate_limit_wait_s`.
        rl_wait = round(cr.rate_limit_wait_s, 2)
        wall = round(cr.wall_time_s - cr.rate_limit_wait_s, 2)
        if rl_wait:
            print(f"[banter] rate-limit waits excluded from wall time: "
                  f"{rl_wait:.0f}s", flush=True)
        # Exact execution split of wall: platform = seconds inside interface
        # (cli/mcp/sdk) tool calls — remote execution against the platform;
        # local = the rest (local tools + LLM generation). Clamped so the
        # wall = platform + local identity survives rounding.
        platform_time = min(
            results.platform_tool_time(
                cr.tool_spans,
                cli_binary=count_cli,
                sdk_module=count_sdk,
                run_dir=run_dir,
                cli_subcommand=count_cli_sub,
            ),
            wall,
        )
        local_time = round(wall - platform_time, 2)
        usage_cols = {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "cost_usd": usage.get("cost_usd", 0.0),
        }
        row = results.Row(
            started_at=started.isoformat(),
            run=spec.run_id or "",
            version="",   # raw Row field; the CSV `version` column carries interface_ref
            platform=spec.platform,
            interface=spec.interface,
            skills=spec.skills,
            category=spec.category,
            task=spec.task,
            interface_ref=interface_setup.ref,
            model=spec.model,
            wall_time_s=wall,
            rate_limit_wait_s=rl_wait,
            platform_time_s=platform_time,
            local_time_s=local_time,
            **usage_cols,
            llm_calls=usage.get("llm_calls", 0),
            run_dir=str(attempt_dir),
            **counts,
            **endpoint_counts,
            **slim_grading,
        )

        # Annotate failures. A missing deliverable keeps its metrics — a clean
        # give-up with a capability report is meaningful data (tokens,
        # friction, denied calls). Metrics are zeroed ONLY when the agent
        # process itself died (crash/timeout), where partial counts would
        # mislead.
        if not deliverable_exists:
            row.error = str(grading.get("error")
                            or grading.get("diagnostic")
                            or "deliverable not found on the platform")[:1000]
            if cr.exit_code != 0:
                for _f in ("wall_time_s", "rate_limit_wait_s",
                           "platform_time_s", "local_time_s",
                           "input_tokens", "output_tokens",
                           "total_tokens", "cost_usd", "llm_calls", "cli_calls",
                           "mcp_calls", "sdk_calls", "python_calls", "bash_calls",
                           "skill_calls", "read_calls", "write_calls", "edit_calls",
                           "glob_calls", "grep_calls", "todo_calls",
                           "failed_commands", "denied_calls",
                           "whitelist_hits", "blacklist_hits"):
                    setattr(row, _f, 0)
                print(
                    f"[banter] DEAD run (agent exited {cr.exit_code}, no "
                    f"deliverable) for {run_dir} — recording a zeroed "
                    f"row with error; artifacts kept on disk.",
                    file=sys.stderr,
                )

        # ONE global CSV for all configs, appended right here
        # after every run so the table is always fresh. `append` numbers the
        # row's repeat counter `n` against the rows already in the global table.
        global_csv = spec.results_csv or (
            _results_root_from(spec.runs_root) / "results.csv")
        results.append(global_csv, row, fields=results.RESULTS_FIELDS)

        # Teardown: run done — stop the platform's background servers, then remove
        # its standalone venv (platform package + everything the agent
        # pip-installed) and the copied skill bundle, so nothing persists into
        # other runs. Results/artifacts (agent.log, submission, grading,
        # commands.jsonl, endpoint_coverage.json) stay.
        _run_aux(interface_setup.teardown, run_dir, {
            **os.environ,
            **interface_setup.keys,
            "BANTER_PLATFORM_DIR": str(interfaces.CONFIGS_DIR / spec.platform),
        })
        shutil.rmtree(run_dir / "venv", ignore_errors=True)
        shutil.rmtree(run_dir / ".claude" / "skills", ignore_errors=True)
        # Transient processing inputs, not kept per run: the raw stream-json
        # transcript (already mined for usage + commands above) and the API log
        # (already folded into endpoint_coverage.json). Drop both so each run
        # dir keeps only distilled artifacts.
        (run_dir / "transcript.jsonl").unlink(missing_ok=True)
        (run_dir / "api_calls.jsonl").unlink(missing_ok=True)
        # Claude Code's built-in sandbox is per-session config (the `sandbox` block
        # in <run_dir>/.claude/settings.json); no teardown.
    return row
