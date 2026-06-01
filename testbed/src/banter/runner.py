"""Orchestrates a single (challenge, interface, mode, skills) engineer run.

Layout produced under the caller-supplied `runs_root`:
    <runs_root>/<task>/<challenge>/      # one engineer run; venv torn down at end
    <runs_root>/results.csv              # one detailed row per challenge

Callers set `runs_root` to put the runs where they want them:
    benchmark      results/benchmark/<run>/
    autoresearch   results/autoresearch/<run>/v<N>/       (per-version)

Each challenge run folder contains:
    prompt.txt            # exact task prompt handed to claude -p
    venv/                 # per-run engineer venv (deleted at teardown)
    data/                 # symlink to prepared MLE-bench data
    submission/           # where the engineer writes submission.csv
    transcript.jsonl      # full claude -p stream-json (tokens + cost in `result`)
    stream.log            # live formatted view (tee captures fd 1/2 of the run)
    commands.jsonl        # one line per tool call (cli/mcp/sdk/python/bash/skill/...)
    grading.json          # MLE-bench grader output
    .claude/settings.json # PreToolUse hook for claude -p
    .claude/skills/       # copied skill bundle (deleted at teardown)   [skills != none]
    .mcp.json             # MCP servers                                  [mode == mcp]

When `spec.interface_dir` is set (autoresearch's per-increment copy), this run's
interface home is pointed at that copy via `interfaces.set_interface_home`; any
stale wheel is dropped and `interfaces.preflight` force-rebuilds the researcher's
edits before the engineer uses it. Login (`auth_command`) is verified per run.
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
    docs as docs_mod,
    interfaces,
    mlebench_wrapper,
    preflight as preflight_mod,
    results,
    skills as skills_mod,
    streaming,
)


DEFAULT_MODEL = "claude-sonnet-4-6"
AUTH_MODES = ("api-key", "login")

# Testbed repo root — we keep the mle-bench data cache inside the repo so
# it travels with the testbed and doesn't depend on the user's $HOME.
TESTBED_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = TESTBED_ROOT / "cache" / "mle-bench"
# The managed researcher venv. `banter` itself runs outside any venv; the
# researcher operates in this venv, and every engineer run gets its own venv
# nested on it (sharing its libraries).
RESEARCHER_VENV = TESTBED_ROOT / ".venv"


@dataclass
class RunSpec:
    challenge_id: str
    interface: str                  # e.g. "hopsworks", "none"
    mode: str                       # interface type: cli/mcp/sdk/none
    skills: str = "none"            # bundle name under testbed/skills/ or "none"
    docs: str = "none"              # docs bundle under interfaces/<project>/docs/ or "none"
    task: str = "no_task"           # ML task / challenge-group this challenge belongs to
    model: str = DEFAULT_MODEL
    auth: str = "api-key"
    timeout_s: int = 60 * 60
    runs_root: Path = Path("results")
    data_root: Path = DEFAULT_DATA_ROOT
    interface_version: int | None = None  # None/0 → base manifest; >0 → session version
    skills_version: int | None = None     # None → latest version
    # Where session-local interface versions live (an autoresearch session dir).
    # Required to resolve interface_version > 0.
    version_root: Path | None = None
    # Per-increment interface copy (autoresearch): the engineer builds + uses THIS
    # interface home instead of the committed one. Built fresh here (the copy
    # ships source, no binary). None → committed interfaces/<name>/<type>/.
    interface_dir: Path | None = None
    # Autoresearch context (None for benchmark). `run_id` → `run` column.
    # `prev_run` / `prev_version` come from the autoresearch config
    # (continuation hints) and are surfaced as columns so results.csv can be
    # queried for "all runs continuing from run X version v2".
    run_id: str | None = None
    version: str | None = None       # "v<N>" or just N (int as string)
    prev_run: str | None = None
    prev_version: str | None = None  # "v<N>" or just N
    # Run the upfront preflight (interface install/login + skill access probe).
    # Batch runners set this False after doing one shared preflight over the union.
    preflight: bool = True


def _make_venv(target: Path) -> Path:
    """Create the engineer's per-run venv FULLY MATERIALIZED inside the challenge.

    `banter` runs outside any venv. The researcher operates in the managed
    researcher venv (testbed/.venv); each engineer venv gets a complete copy
    of the researcher venv's site-packages INSIDE the engineer's challenge
    dir — no `.pth` indirection to an outside path. This lets the engineer's
    sandbox be truly bounded to its challenge folder.

    On APFS (default on macOS) `cp -Rc` does copy-on-write clones, so the
    2-3 GB shared site-packages costs near-zero disk space until the engineer
    modifies a page (pip install of the interface). Falls back to a regular
    recursive copy if APFS clones aren't available (Linux, non-APFS volumes).
    """
    py = target / "bin" / "python"
    if py.exists():
        return py

    researcher_py = RESEARCHER_VENV / "bin" / "python"
    base_py = researcher_py if researcher_py.exists() else Path(sys.executable)
    subprocess.run(
        [str(base_py), "-m", "venv", "--system-site-packages", str(target)],
        check=True,
    )
    # Materialize the researcher venv's site-packages into the engineer venv.
    # The engineer's sandbox confines reads to its challenge dir; any `.pth`
    # reference outside would resolve to a path Seatbelt denies.
    if researcher_py.exists():
        eng_sp = next((target / "lib").glob("python*/site-packages"), None)
        res_sp = next((RESEARCHER_VENV / "lib").glob("python*/site-packages"), None)
        if eng_sp and res_sp:
            _clone_tree(res_sp, eng_sp)
    return py


def _clone_tree(src: Path, dst: Path) -> None:
    """APFS-clone (or copy) every entry in `src` into existing `dst`.

    Uses `cp -Rc` for APFS clone (near-zero cost on macOS); falls back to
    `cp -R` on filesystems that don't support clones. `dst` already exists
    (created by venv); we merge children into it without re-creating dst.
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
    """Best-effort interface housekeeping steps (serve/teardown). Failures are
    ignored and output discarded — these are not part of the engineer's work.
    """
    for cmd in steps:
        try:
            subprocess.run(
                cmd, shell=True, cwd=str(run_dir), env=env,
                timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


TASK_PROMPT_PATH = TESTBED_ROOT / "prompts" / "engineer.md"


def _total_ram_gb() -> float | None:
    """Total physical RAM in GiB, or None if it can't be determined.

    Uses POSIX sysconf (works on both macOS and Linux); no third-party deps.
    """
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3)
    except (ValueError, OSError, AttributeError):
        return None


def detect_environment() -> str:
    """Detect the engineer's hardware/OS and render it as prompt bullet lines.

    Probed on the host, which is the same machine the per-run venv (and thus the
    engineer) runs on, so the facts apply to the engineer. Avoids importing
    torch (not yet installed at prompt-build time) — accelerator availability is
    inferred from OS/arch and `nvidia-smi` presence. The `spawn`-deadlock warning
    is only emitted on platforms whose multiprocessing default is `spawn`
    (macOS/Windows), where the unguarded `num_workers>0` pattern hangs forever.
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


# The "interface is under test" rule is the engineer default — baked into
# engineer.md between these markers. It only applies when an interface is
# present; for none/none (the engineer builds its own model freely) the whole
# section is stripped.
_UNDER_TEST_RE = re.compile(
    r"<!--UNDER_TEST_START-->\n(.*?)\n<!--UNDER_TEST_END-->", re.DOTALL
)


def _build_prompt(
    challenge_id: str,
    fragment: str,
    docs_name: str = "none",
    interface_under_test: bool = True,
) -> str:
    template = TASK_PROMPT_PATH.read_text()
    if interface_under_test:
        template = _UNDER_TEST_RE.sub(lambda m: m.group(1), template)
    else:
        template = _UNDER_TEST_RE.sub("", template)
    template = re.sub(r"\n{3,}", "\n\n", template)
    if docs_name and docs_name != "none":
        docs_block = (
            f"\n\n## Reference docs\n\n"
            f"A docs bundle (`{docs_name}`) is available at `./docs/` — browse "
            f"these files when you need to look up how the interface works "
            f"(API surfaces, expected inputs/outputs, examples). Treat them as "
            f"read-only reference. They will NOT make API calls themselves."
        )
    else:
        docs_block = ""
    return template.format(
        challenge_id=challenge_id,
        fragment=fragment,
        environment=detect_environment(),
    ).strip() + docs_block


def run(spec: RunSpec) -> results.Row:
    if spec.auth not in AUTH_MODES:
        raise ValueError(f"Unknown auth mode {spec.auth!r}; expected one of {AUTH_MODES}")

    started = datetime.now(timezone.utc)

    # Per-version interface copy (autoresearch): point this interface's home
    # at the copy so config/build/$INTERFACE_DIR/install all resolve there.
    # ALWAYS rebuild — the researcher may have edited source between
    # `prepare-version` and this run, so any wheel in the copy is potentially
    # stale. Drop it; preflight will rebuild + test.
    if spec.interface_dir is not None:
        interfaces.set_interface_home(spec.interface, spec.mode, spec.interface_dir)
        for w in list(Path(spec.interface_dir).glob("*.whl")):
            try:
                w.unlink()
            except OSError:
                pass
        st = interfaces.preflight(spec.interface, spec.mode, check_login=False, timeout_s=spec.timeout_s)
        if not st.ok:
            raise preflight_mod.PreflightError(st.message)

    # Fail fast, upfront: the interface must be installed + login must work, and
    # any chosen skill bundle must be accessible to the engineer in a run.
    # Batch runners (benchmark/autoresearch) preflight the union once and pass
    # preflight=False here to avoid re-probing per run.
    if spec.preflight:
        preflight_mod.check_run(
            interface=spec.interface,
            mode=spec.mode,
            interface_version=spec.interface_version,
            version_root=spec.version_root,
            skills=spec.skills,
            skills_version=spec.skills_version,
            auth=spec.auth,
            model=spec.model,
        )

    version, variant_hash = interfaces.variant_for(
        spec.interface, spec.mode, spec.interface_version, spec.version_root
    )
    interface_dir = (
        str(spec.interface_dir) if spec.interface_dir is not None
        else f"interfaces/{spec.interface}/{spec.mode}/config.yaml"
    )
    if version and spec.version_root is not None:
        prompt_file = (
            f"{interfaces.version_dir(spec.version_root, spec.interface, spec.mode, version)}"
            "/version.yaml#prompt"
        )
    else:
        prompt_file = f"{interface_dir}#prompt"
    prompt_version = version
    prompt_hash = interfaces.prompt_hash_for(
        spec.interface, spec.mode, version, spec.version_root
    )
    # Fail fast on unverified skill bundles — before spending time on
    # venv/data/prep we want to know the bundle is well-formed.
    if spec.skills == "none":
        skills_version, skills_hash = 0, ""
        skills_dir = ""
    else:
        skills_version, skills_hash, _ = skills_mod.verify_installed(
            spec.interface, spec.skills, spec.skills_version, spec.version_root
        )
        skills_dir = (
            f"interfaces/{spec.interface}/skills/{spec.skills}"
            if not skills_version
            else f"{spec.version_root}/skills/{spec.skills}/v{skills_version}"
        )
    # Per-challenge output lives directly under the caller-supplied runs_root:
    #   benchmark      results/benchmark/<run>/<task>/<challenge>/
    #   autoresearch   results/autoresearch/<run>/<increment>/<task>/<challenge>/
    # Interface / type / skills / version are recorded as results.csv columns, not
    # encoded in the path (one interface per run, per the per-type configs).
    run_dir = spec.runs_root / spec.task / spec.challenge_id
    # Re-runs overwrite the previous output.
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Capture the ENTIRE terminal output of this run — venv build, mle-bench data
    # prep, pip installs, the engineer's streamed activity, grading — into the
    # run's stream.log (FD-level tee). Echo to the real terminal unless quiet or
    # nested (autoresearch, where the researcher captures this subprocess's
    # stdout and the FileTailer surfaces stream.log live instead).
    passthrough = not streaming.nested() and not streaming.quiet()
    with streaming.tee_to(run_dir / "stream.log", passthrough=passthrough):
        venv_python = _make_venv(run_dir / "venv")
        (run_dir / "submission").mkdir(exist_ok=True)

        # mle-bench data prep must succeed before we spin up Claude — without
        # data/ the engineer has nothing to score against. Don't swallow.
        mlebench_wrapper.prepare(spec.challenge_id, run_dir, spec.data_root)

        interface_setup = interfaces.setup(
            spec.interface, spec.mode, run_dir, venv_python,
            spec.interface_version, spec.version_root,
        )
        skills_setup = skills_mod.apply(
            spec.interface, spec.skills, run_dir, spec.skills_version, spec.version_root
        )
        if skills_setup.installed:
            print(
                f"[banter] skills v{skills_setup.version} ({skills_setup.hash}): "
                f"{','.join(skills_setup.installed)}",
                file=sys.stderr,
            )
        # Docs: reference material cloned from a git URL (or "none"). In
        # autoresearch the bundle is cloned once into `<run>/docs/` by the
        # session bootstrap; here we APFS-clone from there to avoid re-fetching
        # per challenge. In benchmark / standalone runs there's no upstream
        # copy and we fetch directly into the challenge dir.
        share = spec.runs_root.parent if (spec.runs_root.parent / "docs").is_dir() else None
        docs_setup = docs_mod.apply(spec.docs, run_dir, share_from=share)
        if docs_setup.files:
            print(
                f"[banter] docs {docs_setup.spec!r}: {len(docs_setup.files)} files "
                f"at {run_dir / 'docs'}",
                file=sys.stderr,
            )

        # Start fresh: stop any servers a previous (possibly crashed) run left
        # running and clear leaked state, then start the servers THIS run needs
        # (e.g. the MCP HTTP server claude connects to at launch) in the run venv.
        base_keys_env = {**os.environ, **interface_setup.keys}
        _run_aux(interface_setup.teardown, run_dir, base_keys_env)
        if interface_setup.serve:
            serve_env = dict(base_keys_env)
            serve_env["PATH"] = f"{run_dir / 'venv' / 'bin'}{os.pathsep}{serve_env.get('PATH', '')}"
            serve_env["VIRTUAL_ENV"] = str(run_dir / "venv")
            _run_aux(interface_setup.serve, run_dir, serve_env)

        # Log in + verify auth for THIS challenge in its own venv. Build/test ran
        # once at session preflight; login is re-checked on every run (catches
        # expired creds mid-session; each run authenticates in its own venv).
        login = interfaces.login_status(
            spec.interface, spec.mode, venv_python=venv_python, keys=interface_setup.keys,
        )
        if not login.ok:
            raise preflight_mod.PreflightError(login.message)

        prompt = _build_prompt(spec.challenge_id, interface_setup.prompt_fragment,
                               docs_name=spec.docs,
                               interface_under_test=interface_setup.type != "none")
        (run_dir / "prompt.txt").write_text(prompt)

        # When in autoresearch (`spec.version` is set), confine the engineer's
        # sandbox to the entire version dir (`<run>/v<N>`) — runs_root IS the
        # version dir. For benchmark, leave it None so the engineer is confined
        # to its own challenge dir.
        version_dir = spec.runs_root if spec.version else None
        cr = claude_runner.run(
            prompt=prompt,
            run_dir=run_dir,
            auth=spec.auth,
            model=spec.model,
            cli_binary=interface_setup.cli_binary,
            sdk_module=interface_setup.sdk_module,
            mcp_servers=interface_setup.mcp_servers,
            command_log=run_dir / "commands.jsonl",
            timeout_s=spec.timeout_s,
            extra_env=interface_setup.keys,
            version_dir=version_dir,
            allowed_domains=interface_setup.allowed_domains,
        )

        if cr.exit_code != 0:
            print(f"[banter] claude exit={cr.exit_code}", file=sys.stderr)

        submission = run_dir / "submission" / "submission.csv"
        try:
            grading = mlebench_wrapper.grade(spec.challenge_id, submission, spec.data_root)
        except Exception as e:
            grading = {"medal": None, "score": None, "error": str(e)}
        (run_dir / "grading.json").write_text(json.dumps(grading, indent=2))

        usage = results.parse_transcript_usage(cr.transcript_path)
        counts = results.aggregate_commands(
            cr.transcript_path,
            cli_binary=interface_setup.cli_binary,
            sdk_module=interface_setup.sdk_module,
            run_dir=run_dir,
        )
        # Rebuild commands.jsonl from the transcript so it's always populated even
        # when the PreToolUse hook silently fails.
        results.write_commands_log(
            cr.transcript_path,
            run_dir / "commands.jsonl",
            cli_binary=interface_setup.cli_binary,
            sdk_module=interface_setup.sdk_module,
            run_dir=run_dir,
        )
        # mle-bench grading report → Row fields. Booleans stored 0/1 so they
        # average into rates at rollup; thresholds kept as-is (may be None).
        _b = lambda x: int(bool(x))  # noqa: E731
        grading_fields = dict(
            medal=grading.get("medal"),
            score=grading.get("score"),
            valid_submission=_b(grading.get("valid_submission")),
            above_median=_b(grading.get("above_median")),
            any_medal=_b(grading.get("any_medal")),
            gold_medal=_b(grading.get("gold_medal")),
            silver_medal=_b(grading.get("silver_medal")),
            bronze_medal=_b(grading.get("bronze_medal")),
            gold_threshold=grading.get("gold_threshold"),
            silver_threshold=grading.get("silver_threshold"),
            bronze_threshold=grading.get("bronze_threshold"),
            median_threshold=grading.get("median_threshold"),
            is_lower_better=_b(grading.get("is_lower_better")),
        )
        # `version` is stored as `v<N>` (e.g. "v2"). Accept either "v2" or
        # the bare integer "2" on input — normalise to the prefixed form.
        def _to_v(v: str | None) -> str:
            if not v:
                return ""
            tail = str(v).rsplit("_", 1)[-1]
            if tail.startswith("v") and tail[1:].isdigit():
                return tail
            if tail.isdigit():
                return f"v{tail}"
            return ""
        slim_grading = {k: grading_fields.get(k) for k in ("valid_submission", "score", "medal")}
        # Map engineer's `claude -p` usage into eng_* columns. Researcher
        # values stay at 0 here; autoresearch end-of-session backfills them
        # and computes the `total_*` aggregates.
        eng_wall = round(cr.wall_time_s, 2)
        eng_usage = {
            "eng_input_tokens": usage.get("input_tokens", 0),
            "eng_output_tokens": usage.get("output_tokens", 0),
            "eng_total_tokens": usage.get("total_tokens", 0),
            "eng_cost_usd": usage.get("cost_usd", 0.0),
        }
        row = results.Row(
            started_at=started.isoformat(),
            run=spec.run_id or "",
            version=_to_v(spec.version),
            interface=spec.interface,
            type=spec.mode,
            skills=spec.skills,
            prev_run=spec.prev_run or "",
            prev_version=_to_v(spec.prev_version),
            task=spec.task,
            challenge=spec.challenge_id,
            eng_wall_time_s=eng_wall,
            **eng_usage,
            total_wall_time_s=eng_wall,
            total_tokens=eng_usage["eng_total_tokens"],
            total_cost=eng_usage["eng_cost_usd"],
            llm_calls=usage.get("llm_calls", 0),
            run_dir=str(run_dir),
            **counts,
            **slim_grading,
        )

        # One results.csv per session, inside that session's run dir. For
        # autoresearch, `runs_root` is `<run>/incr_N`, so the CSV lives one
        # level up at `<run>/results.csv` and accumulates rows from every
        # increment in the session. Benchmark leaves `increment` None, so
        # the CSV stays at `<runs_root>/results.csv`.
        master_csv_dir = spec.runs_root.parent if spec.version else spec.runs_root
        results.append(
            master_csv_dir / "results.csv",
            row,
            # Benchmark uses the slimmer column set; autoresearch uses the full FIELDS.
            fields=None if spec.version else results.BENCHMARK_FIELDS,
        )

        # Notebook regeneration happens once at end-of-autoresearch (which
        # has the goals in-process); the runner just appends rows to the CSV.

        # Teardown: the run is done — stop the interface's background servers,
        # then remove its standalone venv (the interface package + everything the
        # engineer pip-installed) and the copied skill bundle, so nothing persists
        # into other runs. Results/artifacts (transcript, stream.log, submission,
        # grading) stay.
        _run_aux(interface_setup.teardown, run_dir, {**os.environ, **interface_setup.keys})
        shutil.rmtree(run_dir / "venv", ignore_errors=True)
        shutil.rmtree(run_dir / ".claude" / "skills", ignore_errors=True)
        # Claude Code's built-in sandbox is per-session config (the
        # `sandbox` block in <run_dir>/.claude/settings.json); no teardown.
    return row
