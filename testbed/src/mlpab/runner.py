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
import secrets
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from mlpab import (
    claude_runner,
    codex_runner,
    evals_provider,
    interfaces,
    mistral_runner,
)
from mlpab import preflight as preflight_mod
from mlpab import (
    redact,
    results,
)
from mlpab import skills as skills_mod
from mlpab import (
    streaming,
)

DEFAULT_MODEL = "claude-sonnet-4-6"
AUTH_MODES = ("api-key", "login")

# Testbed repo root.
TESTBED_ROOT = Path(__file__).resolve().parents[2]
# The managed base venv (testbed/.venv). `mlpab` itself runs outside any venv;
# each agent run gets its own venv nested on this one (sharing its libraries).
BASE_VENV = TESTBED_ROOT / ".venv"


@dataclass
class RunSpec:
    task: str  # FTI sub-task (an evals_provider family)
    platform: str  # e.g. "hopsworks", "none"
    interface: str  # interface: cli/mcp/sdk/none
    skills: str = "none"  # platform skill bundle name or "none"
    category: str = "no_task"  # FTI category (stage) this task belongs to
    model: str = DEFAULT_MODEL
    auth: str = "api-key"
    timeout_s: int | None = 60 * 60  # None → NO per-run wall-clock cap
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


def _deliverable_exists(asserts: list[dict]) -> bool:
    """Whether a gradeable deliverable was produced — the `valid` column,
    separating "wrong answer" from "no deliverable".

    It is the verdict of the FIRST GRADED (non-skip) assert: skipped checks
    (not reached, or not applicable on this platform — e.g. the `none`
    baseline's table-exists check) are stepped over, so the probe lands on the
    first assert actually evaluated rather than positionally `asserts[0]` (which
    can now be a skip carrying passed=False)."""
    graded = [a for a in asserts if a.get("status") != "skip"]
    return bool(graded) and bool(graded[0].get("passed"))


def _results_root_from(runs_root: Path) -> Path:
    """Nearest `results/` ancestor of `runs_root` (…/results/<config>/…).
    Falls back to `runs_root` if none found."""
    runs_root = Path(runs_root)
    for p in (runs_root, *runs_root.parents):
        if p.name == "results":
            return p
    return runs_root


def _make_venv(target: Path, prepared: Path | None = None) -> Path:
    """Create the agent's per-run venv FULLY MATERIALIZED inside the run dir.

    When `prepared` is given (an interface's prepared venv — see
    interfaces.prepare), the run venv is a CLONE of it: base libs AND the
    interface are already installed, so the run does no pip work and mutates no
    shared state. Otherwise (no prepared venv) the venv is built from the base
    venv and the interface is installed per run by interfaces.setup (legacy path).

    Either way the site-packages are copied entirely INSIDE the agent's run dir —
    no `.pth` indirection to an outside path — so the agent's sandbox is truly
    bounded to its run folder.

    On APFS (macOS default) `cp -Rc` does copy-on-write clones, so the 2-3 GB
    shared tree costs near-zero disk until the agent modifies a page. Falls back
    to a recursive copy where APFS clones aren't available (Linux, non-APFS).
    """
    py = target / "bin" / "python"
    if py.exists():
        return py

    if prepared is not None and (prepared / "bin" / "python").exists():
        _clone_prepared_venv(prepared, target)
        return py

    base_venv_py = BASE_VENV / "bin" / "python"
    base_py = base_venv_py if base_venv_py.exists() else Path(sys.executable)
    subprocess.run(
        [str(base_py), "-m", "venv", "--system-site-packages", str(target)],
        check=True,
        timeout=300,
    )
    # Materialize the base venv's site-packages into the agent venv: its
    # sandbox confines reads to the run dir, so any outside `.pth` reference
    # would resolve to a path Seatbelt denies.
    if base_venv_py.exists():
        eng_sp = interfaces.venv_site_packages(target)
        base_sp = interfaces.venv_site_packages(BASE_VENV)
        if eng_sp and base_sp:
            _clone_tree(base_sp, eng_sp)
    return py


def _clone_prepared_venv(src: Path, dst: Path) -> None:
    """Clone a prepared venv into the run dir (APFS CoW where available) and
    RELOCATE it: a venv hardcodes its own absolute path into bin/ scripts
    (shebangs, `activate`) and `pyvenv.cfg`, so the copy would still point at the
    prepared venv (outside the sandbox). Rewrite those paths to the run dir."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["cp", "-Rc", str(src), str(dst)], check=True, timeout=600)
    except subprocess.CalledProcessError:
        subprocess.run(["cp", "-R", str(src), str(dst)], check=True, timeout=600)
    _relocate_venv(src, dst)


def _relocate_venv(old: Path, new: Path) -> None:
    """Rewrite a venv's self-referential absolute path (`old` → `new`) in its
    bin/ text scripts and pyvenv.cfg. Console-script shebangs (`#!<venv>/bin/
    python`) and the `activate` family carry the venv path; bin/python itself is
    a symlink to the base interpreter (left as-is, like a freshly-created venv).
    """
    old_s, new_s = str(old), str(new)
    bin_dir = new / "bin"
    if bin_dir.is_dir():
        for entry in bin_dir.iterdir():
            if entry.is_symlink() or not entry.is_file():
                continue
            try:
                data = entry.read_bytes()
            except OSError:
                continue
            if b"\0" in data[:2048]:  # a real binary, not a text script
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if old_s in text:
                entry.write_text(text.replace(old_s, new_s))
    cfg = new / "pyvenv.cfg"
    if cfg.is_file():
        text = cfg.read_text()
        if old_s in text:
            cfg.write_text(text.replace(old_s, new_s))


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
            subprocess.run(["cp", "-Rc", str(child), str(target)], check=True, timeout=600)
        except subprocess.CalledProcessError:
            subprocess.run(["cp", "-R", str(child), str(target)], check=True, timeout=600)


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
    aux_env = {k: v for k, v in env.items() if k not in ("MLPAB_API_LOG", "MLPAB_IFACE_SDK")}
    for cmd in steps:
        try:
            subprocess.run(
                cmd,
                shell=True,
                cwd=str(run_dir),
                env=aux_env,
                timeout=timeout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


def _verify_platform(
    phase: str,
    platform: str,
    run_dir: Path,
    env: dict[str, str],
    timeout: int = 120,
) -> tuple[bool, str]:
    """Run a platform's `<phase>.py verify` (phase: 'setup' | 'teardown') if the
    script exists, capturing its output. Returns (ok, output); (True, "") when
    there's no such script — nothing to verify.

    The twofold check (per the platform's setup.py/teardown.py `verify` mode):
    confirm CONNECTION, then that the resources setup creates are PRESENT (setup)
    / that the agent's resources are GONE (teardown). Read-only — it mutates
    nothing. Runs with the platform's PLUMBING venv python when one exists (it
    holds the platform's python client regardless of the interface under test —
    a CLI run venv may lack it), else the per-run venv python. The api-logging
    shim is stripped (its reads are plumbing, not the agent's endpoint
    coverage)."""
    script = interfaces.CONFIGS_DIR / platform / f"{phase}.py"
    if not script.exists():
        return True, ""
    # Prefer the plumbing venv (has the python client on every interface); fall
    # back to the run venv. Absolute paths: the subprocess runs with cwd=run_dir,
    # so a venv-python path relative to the CURRENT cwd would resolve against
    # run_dir and miss (the single-task form passes a relative run_dir).
    # .absolute() does NOT follow the bin/python symlink out of the venv (unlike
    # .resolve()).
    plumbing_py = interfaces.plumbing_python(platform)
    venv_py = (run_dir / "venv" / "bin" / "python").absolute()
    if plumbing_py is not None:
        py = str(plumbing_py.absolute())
    elif venv_py.exists():
        py = str(venv_py)
    else:
        py = sys.executable
    verify_env = {k: v for k, v in env.items() if k not in ("MLPAB_API_LOG", "MLPAB_IFACE_SDK")}
    try:
        proc = subprocess.run(
            [py, str(script.absolute()), "verify"],
            cwd=str(run_dir),
            env=verify_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except Exception as e:
        return False, f"{phase} verify did not run: {e}"
    # Scrub credentials before this output reaches an exception message / the log.
    # Scan THIS verify env (not just os.environ) so a manifest-inlined key value
    # that isn't in the process environment is still redacted.
    return proc.returncode == 0, redact.redact(
        proc.stdout.decode("utf-8", "replace"), env=verify_env
    )


def _platform_env(run_dir: Path) -> dict[str, str]:
    """Env the platform setup step exported for the AGENT, then consume the file.

    Setup scripts run as throwaway subprocesses (`serve:` via `_run_aux`), so they
    cannot literally export env vars into the agent's process. Instead they may
    append KEY=VALUE lines to $MLPAB_PLATFORM_ENV (<run_dir>/platform.env) for
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


# Credential env vars whose VALUE is a path to a key/cert FILE (not the secret
# itself). The agent runs sandboxed to its run dir, so a file outside that
# boundary (e.g. testbed/.azure/azure-sp.pem, testbed/.gcp/adc.json) is
# unreadable to it. azure/gcp need these because their tenants/orgs block inline
# secrets/keys (cert / ADC only). Value-based creds (DATABRICKS_TOKEN, AWS_*)
# need no localization — they ride in the env directly.
_FILE_CREDENTIAL_KEYS = ("AZURE_CLIENT_CERTIFICATE_PATH", "GOOGLE_APPLICATION_CREDENTIALS")


def _localize_file_credentials(keys: dict[str, str], run_dir: Path) -> dict[str, str]:
    """Copy any file-path credential into the agent's run dir (sandbox-readable)
    and repoint its env var at the in-sandbox copy. Returns a NEW dict — the
    grader keeps the original host path (it runs unsandboxed from .env). The
    copies live under <run_dir>/.creds and vanish with the run dir at teardown."""
    out = dict(keys)
    for key in _FILE_CREDENTIAL_KEYS:
        src = out.get(key)
        if not src or not Path(src).is_file():
            continue
        creds_dir = run_dir / ".creds"
        creds_dir.mkdir(parents=True, exist_ok=True)
        dst = (creds_dir / Path(src).name).absolute()
        shutil.copy(src, dst)
        os.chmod(dst, 0o600)
        out[key] = str(dst)
    return out


TASK_PROMPT_PATH = TESTBED_ROOT / "prompts" / "agent.md"


def _total_ram_gb() -> float | None:
    """Total physical RAM in GiB, or None if undeterminable.

    Uses POSIX sysconf (macOS and Linux); no third-party deps.
    """
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024**3)
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

    system = platform.system()  # Darwin / Linux / Windows
    machine = platform.machine() or "?"  # arm64 / x86_64 / ...
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
            '(or any multiprocessing) whose script lacks an `if __name__ == "__main__":` '
            "guard will DEADLOCK and hang forever — it never errors, it just stalls. "
            "Default to `num_workers=0`; only use workers if all top-level code is under "
            "that guard."
        )
    return "\n".join(lines)


# The "platform is under test" rule (agent default), baked into agent.md
# between these markers. Applies only when a platform is present; for none/none
# (agent builds its own model freely) the whole section is stripped.
_UNDER_TEST_RE = re.compile(r"<!--UNDER_TEST_START-->\n(.*?)\n<!--UNDER_TEST_END-->", re.DOTALL)
# Inverse: content kept ONLY when NO interface is under test (none/none, agent
# trains its own model locally). Stripped when an interface is under test
# (everything must run remotely on the platform).
_LOCAL_ONLY_RE = re.compile(r"<!--LOCAL_ONLY_START-->\n(.*?)\n<!--LOCAL_ONLY_END-->", re.DOTALL)


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
        # Prefer cloning the interface's PREPARED venv (base libs + interface,
        # built once at preflight): the run does no pip work and mutates no shared
        # state. Only when none exists do we fall back to the legacy path — clone
        # the base .venv (kept free of the interface) and install per run below.
        prepared = interfaces.prepared_venv_dir(spec.platform, spec.interface)
        use_prepared = (prepared / "bin" / "python").exists()
        if not use_prepared:
            interfaces.ensure_base_clean(spec.platform, spec.interface)
        venv_python = _make_venv(run_dir / "venv", prepared if use_prepared else None)
        (run_dir / "submission").mkdir(exist_ok=True)

        # Generate a FRESH eval instance (validity gates run inside the
        # generator) and stage its data/ — must succeed before spinning up
        # Claude. The seed is deterministic per (session, category, task,
        # attempt): reproducible rows, fresh instance every repeat.
        seed = evals_provider.seed_for(
            spec.run_id or "", spec.category, spec.task, spec.attempt or 1
        )
        task_body = evals_provider.prepare(spec.task, run_dir, seed)

        interface_setup = interfaces.setup(
            spec.platform,
            spec.interface,
            run_dir,
            venv_python,
            # A prepared-venv clone already has the interface installed — skip the
            # per-run install (it would re-pip into the run venv for nothing).
            run_install=not use_prepared,
        )
        skills_setup = skills_mod.apply(spec.platform, spec.skills, run_dir)
        if skills_setup.installed:
            print(
                f"[mlpab] skills ({skills_setup.hash}): {','.join(skills_setup.installed)}",
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
        # `MLPAB_PLATFORM_DIR` points teardown/serve steps at the COMMITTED
        # configs/platforms/<platform>/ dir (e.g. a teardown.py cleanup script)
        # — cleanup logic is fixed infrastructure, immune to run-dir state.
        # `MLPAB_PLUMBING_PY` is the interpreter those setup/teardown scripts run
        # under: the platform's plumbing venv (has the python client on EVERY
        # interface — a CLI run venv may lack it), falling back to the run venv
        # python when the platform needs none. Built once (idempotent) — cheap
        # per run; covers the single-task form too (no preflight there).
        try:
            plumbing_py = interfaces.prepare_plumbing(spec.platform)
        except Exception as e:
            print(f"[mlpab] plumbing venv prep failed for {spec.platform}: {e}", file=sys.stderr)
            plumbing_py = interfaces.plumbing_python(spec.platform)
        plumbing_py = plumbing_py or (run_dir / "venv" / "bin" / "python")
        # Per-run Hopsworks project name, the single source of truth shared by
        # every phase of this run: setup creates it, the agent's bare
        # `hopsworks.login()` auto-selects it via HOPSWORKS_PROJECT, the grader's
        # adapter reads back through it, and teardown deletes ONLY it. Scoping the
        # name per run is what lets two hopsworks runs coexist on one cluster
        # without deleting each other's projects or cross-attaching at login.
        # Set on os.environ (not just a local dict) because every downstream
        # consumer derives from it — base_keys_env below, each engine's
        # os.environ.copy(), and the grader subprocess which inherits the
        # environment unchanged. A fresh name every run (matching the old
        # setup.py behavior) also sidesteps the backend's async namespace/Kafka
        # delete race documented in setup.py.
        if spec.platform == "hopsworks":
            os.environ["HOPSWORKS_PROJECT"] = f"mlpab{secrets.token_hex(3)}"
        base_keys_env = {
            **os.environ,
            **interface_setup.keys,
            "MLPAB_PLATFORM_DIR": str(interfaces.CONFIGS_DIR / spec.platform),
            "MLPAB_PLUMBING_PY": str(Path(plumbing_py).absolute()),
        }
        _run_aux(interface_setup.teardown, run_dir, base_keys_env)
        if interface_setup.serve:
            serve_env = dict(base_keys_env)
            serve_env["PATH"] = f"{run_dir / 'venv' / 'bin'}{os.pathsep}{serve_env.get('PATH', '')}"
            serve_env["VIRTUAL_ENV"] = str(run_dir / "venv")
            serve_env["MLPAB_PLATFORM_ENV"] = str(run_dir / "platform.env")
            # Generous timeout: hopsworks setup.py retries project creation for
            # up to ~3 min while the backend finishes deleting the previous
            # run's namespace (teardown is async server-side).
            _run_aux(interface_setup.serve, run_dir, serve_env, timeout=300)
        # Twofold check: confirm the platform connects AND that setup actually
        # established what the agent needs, BEFORE spending a (timed, billed)
        # agent run against it. A failure here means the run could not be valid
        # — fail it now with a clear reason instead of letting the agent flail
        # against a half-set-up platform (and the grader later report a missing
        # deliverable it never had a chance to create).
        ok, detail = _verify_platform("setup", spec.platform, run_dir, base_keys_env)
        if not ok:
            # PlatformNotReadyError aborts the whole config (the treatment loop
            # re-raises it) — every later run would hit the same broken platform.
            raise preflight_mod.PlatformNotReadyError(
                f"platform setup verification failed for {spec.platform!r} — the "
                f"platform is not ready, so this run would not be valid:\n"
                f"{detail.strip()[-1000:]}"
            )
        # Env the setup step exported for the agent (e.g. a role ARN it just
        # created) — declared keys (.env) win on conflict, so an explicit .env
        # value always overrides what setup derived.
        agent_keys = {**_platform_env(run_dir), **interface_setup.keys}

        # Log in + verify auth for THIS run in its own venv. Build/test ran
        # once at treatment preflight; login is re-checked every run (catches
        # expired creds mid-session; each run authenticates in its own venv).
        login = interfaces.login_status(
            spec.platform,
            spec.interface,
            venv_python=venv_python,
            keys=agent_keys,
        )
        if not login.ok:
            raise preflight_mod.PreflightError(login.message)

        prompt = _build_prompt(
            task_body,
            interface_setup.prompt_fragment,
            interface_under_test=interface_setup.interface != "none",
        )
        (run_dir / "prompt.txt").write_text(prompt)

        # The agent is confined to its own run dir (the generated
        # instance's private answer key lives in a sibling dir it cannot read).
        # Engine dispatch by model id: OpenAI (`gpt-*`/`*codex*`) → Codex CLI;
        # Mistral (`mistral-*`) → Mistral Vibe CLI; everything else →
        # Claude Code. Same signature + result shape; the codex/mistral engines
        # skip the PreToolUse enforcement hook (no equivalent) but keep full
        # post-hoc accounting via the normalized transcript.
        if codex_runner.is_codex_model(spec.model):
            engine = codex_runner
        elif mistral_runner.is_mistral_model(spec.model):
            engine = mistral_runner
        else:
            engine = claude_runner
        # No agent may touch the testbed git repo during a run (it must not
        # commit/push its run artifacts — see the stray agent commits this
        # guards against). Point git's repo search ceiling at the run dir so a
        # `git` invocation cannot ascend to discover the repo's .git: every
        # command then reports "not a git repository". Engine-agnostic — all
        # engines merge extra_env into the agent subprocess (claude additionally
        # denies the `git` binary via its PreToolUse allowlist).
        agent_env = _localize_file_credentials(agent_keys, run_dir)
        agent_env["GIT_CEILING_DIRECTORIES"] = str(run_dir)
        cr = engine.run(
            prompt=prompt,
            run_dir=run_dir,
            auth=spec.auth,
            model=spec.model,
            cli_binary=interface_setup.cli_binary,
            cli_subcommand=interface_setup.cli_subcommand,
            cli_aux_commands=interface_setup.cli_aux_commands,
            sdk_module=interface_setup.sdk_module,
            mcp_servers=interface_setup.mcp_servers,
            command_log=run_dir / "commands.jsonl",
            timeout_s=spec.timeout_s,
            # sandboxed agent reads cert/ADC from inside its run dir
            extra_env=agent_env,
            allowed_domains=interface_setup.allowed_domains,
            interface=interface_setup.interface,
            instance_allowlist=interface_setup.instance_allowlist,
            sandbox_excluded_commands=interface_setup.sandbox_excluded_commands,
            platform=spec.platform,
        )

        if cr.exit_code != 0:
            print(f"[mlpab] claude exit={cr.exit_code}", file=sys.stderr)

        # Grade by replaying the instance's assertion suite against the
        # platform's read paths (or the local deliverable for platform `none`).
        # Runs BEFORE venv teardown, with the run venv's python.
        #
        # The checker adapter reads the deliverable back through the platform's
        # PYTHON client (databricks-sdk, boto3, …). An SDK-interface run venv
        # already has it; a CLI/MCP venv does NOT, so install it now — after the
        # agent has finished, so its interface purity stands, and the venv is torn
        # down right after. Idempotent (a no-op when already present).
        if spec.platform in evals_provider.ADAPTERS and spec.interface != "sdk":
            try:
                interfaces.install_for_grader(spec.platform, run_dir, venv_python)
            except Exception as e:
                print(
                    f"[mlpab] grader-deps install failed for {spec.platform}: {e}", file=sys.stderr
                )
        try:
            grading = evals_provider.grade(spec.task, run_dir, spec.platform, venv_python)
        except Exception as e:
            grading = {
                "success": False,
                "asserts_passed": 0,
                "asserts_failed": 0,
                "asserts_skipped": 0,
                "total_asserts": 0,
                "asserts": [],
                "error": str(e),
            }
        # grading.json lives grader-side, next to the answer key.
        (attempt_dir / "solution" / "grading.json").write_text(json.dumps(grading, indent=2))

        usage = results.parse_transcript_usage(cr.transcript_path, model=spec.model)
        # Counting classifies by the ACTIVE interface only: a `hops`/`import
        # hopsworks` call is `cli_calls`/`sdk_calls` ONLY when that interface is
        # under test. (interface_setup markers are resolved for ALL interfaces to
        # feed the cross-interface enforcement HOOK, but counting must not relabel
        # an off-interface call as on-interface.)
        count_cli = interface_setup.cli_binary if interface_setup.interface == "cli" else None
        count_cli_sub = (
            interface_setup.cli_subcommand if interface_setup.interface == "cli" else None
        )
        count_cli_aux = (
            interface_setup.cli_aux_commands if interface_setup.interface == "cli" else None
        )
        count_sdk = interface_setup.sdk_module if interface_setup.interface == "sdk" else None
        counts = results.aggregate_commands(
            cr.transcript_path,
            cli_binary=count_cli,
            sdk_module=count_sdk,
            run_dir=run_dir,
            cli_subcommand=count_cli_sub,
            cli_aux=count_cli_aux,
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
            cli_aux=count_cli_aux,
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
                    f"[mlpab] WARNING: {spec.platform} {interface_setup.interface} "
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
        _iface = (
            interface_setup.interface
            if interface_setup.interface in ("cli", "mcp", "sdk")
            else None
        )
        endpoint_cov = results.endpoint_coverage(
            run_dir / "api_calls.jsonl", _wl, _bl, interface=_iface
        )
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
        # (all green = task solved). The full report lives in grading.json.
        deliverable_exists = _deliverable_exists(grading.get("asserts") or [])
        slim_grading = {
            # valid = a gradeable deliverable was produced (first assert passed);
            # success = the task was solved correctly (no failed asserts).
            "valid": deliverable_exists,
            "success": bool(grading.get("success", False)),
            "asserts_passed": grading.get("asserts_passed", 0),
            "asserts_failed": grading.get("asserts_failed", 0),
            "asserts_skipped": grading.get("asserts_skipped", 0),
            "total_asserts": grading.get("total_asserts", 0),
        }
        if grading.get("total_asserts"):
            print(
                f"[mlpab] graded: {grading.get('asserts_passed', 0)} passed / "
                f"{grading.get('asserts_failed', 0)} failed / "
                f"{grading.get('asserts_skipped', 0)} skipped "
                f"of {grading['total_asserts']} asserts"
                + (f" — {grading['diagnostic']}" if grading.get("diagnostic") else ""),
                flush=True,
            )
        # Map the agent's `claude -p` usage into the Row's metric columns.
        # Wall metrics count COMPUTE time only: rate-limit back-off sleeps are
        # excluded (waiting != computing) and recorded separately in
        # `rate_limit_wait_s`.
        rl_wait = round(cr.rate_limit_wait_s, 2)
        wall = round(cr.wall_time_s - cr.rate_limit_wait_s, 2)
        if rl_wait:
            print(f"[mlpab] rate-limit waits excluded from wall time: {rl_wait:.0f}s", flush=True)
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
                cli_aux=count_cli_aux,
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
            version="",  # raw Row field; the CSV `version` column carries interface_ref
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
            row.error = str(
                grading.get("error")
                or grading.get("diagnostic")
                or "deliverable not found on the platform"
            )[:1000]
            if cr.exit_code != 0:
                # The agent itself DIED — that's the root cause; "no deliverable"
                # (and any grader read error it triggers) is just a symptom.
                # Report the agent failure, with its stderr tail, INSTEAD of the
                # downstream grader message — otherwise a dead engine (e.g. a
                # model the runner can't reach) masquerades as a grading bug.
                tail = ""
                try:
                    if cr.stderr_path and Path(cr.stderr_path).exists():
                        tail = Path(cr.stderr_path).read_text(errors="replace").strip()[-400:]
                except Exception:
                    pass
                row.error = (
                    f"agent exited {cr.exit_code} (no deliverable produced)"
                    + (f": …{tail}" if tail else "")
                )[:1000]
                for _f in (
                    "wall_time_s",
                    "rate_limit_wait_s",
                    "platform_time_s",
                    "local_time_s",
                    "sleep_calls",
                    "sleep_time_s",
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "cost_usd",
                    "llm_calls",
                    "cli_calls",
                    "mcp_calls",
                    "sdk_calls",
                    "python_calls",
                    "bash_calls",
                    "skill_calls",
                    "read_calls",
                    "write_calls",
                    "edit_calls",
                    "glob_calls",
                    "grep_calls",
                    "todo_calls",
                    "failed_commands",
                    "denied_calls",
                    "whitelist_hits",
                    "blacklist_hits",
                ):
                    setattr(row, _f, 0)
                print(
                    f"[mlpab] DEAD run (agent exited {cr.exit_code}, no "
                    f"deliverable) for {run_dir} — recording a zeroed "
                    f"row with error; artifacts kept on disk.",
                    file=sys.stderr,
                )

        # Scrub any credential value that leaked into the error text (e.g. an
        # agent stderr tail or a platform error echoing a token) before it's
        # persisted to the CSV. extra = this interface's declared key values,
        # which may be manifest-embedded and not in the environment.
        row.error = redact.redact(row.error, extra=interface_setup.keys.values())

        # ONE global CSV for all configs, appended right here
        # after every run so the table is always fresh. `append` numbers the
        # row's repeat counter `n` against the rows already in the global table.
        global_csv = spec.results_csv or (_results_root_from(spec.runs_root) / "results.csv")
        results.append(global_csv, row, fields=results.RESULTS_FIELDS)

        # Teardown: run done — stop the platform's background servers, then remove
        # its standalone venv (platform package + everything the agent
        # pip-installed) and the copied skill bundle, so nothing persists into
        # other runs. Results/artifacts (agent.log, submission, grading,
        # commands.jsonl, endpoint_coverage.json) stay.
        teardown_env = {
            **os.environ,
            **interface_setup.keys,
            "MLPAB_PLATFORM_DIR": str(interfaces.CONFIGS_DIR / spec.platform),
            "MLPAB_PLUMBING_PY": str(Path(plumbing_py).absolute()),
        }
        _run_aux(interface_setup.teardown, run_dir, teardown_env)
        # Twofold check (other half): confirm teardown actually swept the agent's
        # resources. A leak is a cost + cross-run-contamination risk, but the run
        # already happened — so WARN (don't fail), and the next run's start
        # teardown sweeps again. Runs before the venv is removed (an SDK-based
        # verify needs it).
        ok, detail = _verify_platform("teardown", spec.platform, run_dir, teardown_env)
        if not ok:
            print(
                f"[mlpab] WARNING: teardown verification for {spec.platform!r} found "
                f"un-swept resources (possible leak / cost):\n{detail.strip()[-1000:]}",
                file=sys.stderr,
                flush=True,
            )
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
