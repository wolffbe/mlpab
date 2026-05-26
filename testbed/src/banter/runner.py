"""Orchestrates a single (challenge, interface, mode, skills) run.

Layout produced under (within the caller-supplied runs_root):
    <interface>/<mode>/<skills|no_skills>/v<version>/<challenge>/

Each challenge run folder contains:
    prompt.txt            # exact task prompt handed to claude -p
    venv/                 # fresh per-run venv (memory: project_run_isolation)
    data/                 # symlink to prepared MLE-bench data
    submission/           # where the agent writes submission.csv
    transcript.jsonl      # full claude -p stream-json output (tokens + cost in `result` event)
    commands.jsonl        # one line per tool call (cli/mcp/sdk/python/bash/...)
    grading.json          # MLE-bench grader output
    .claude/settings.json # PreToolUse hook config for claude -p
    .claude/skills/       # copied skill bundle                    [skills != none]
    .mcp.json             # MCP servers                            [mode == mcp]

A row is appended to runs_root/results.csv for each run. Autoresearch and
benchmark sessions each supply their own isolated runs_root so every session
has its own results.csv.
"""
from __future__ import annotations

import json
import sys
import venv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from banter import claude_runner, interfaces, mlebench_wrapper, results, skills as skills_mod


DEFAULT_MODEL = "claude-sonnet-4-6"
AUTH_MODES = ("api-key", "login")

# Testbed repo root — we keep the mle-bench data cache inside the repo so
# it travels with the testbed and doesn't depend on the user's $HOME.
TESTBED_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = TESTBED_ROOT / "cache" / "mle-bench"

# Global aggregate CSV — every run from every session (autoresearch,
# benchmark, ad-hoc `banter run`) appends here so you have a single
# cross-session table for analysis.
GLOBAL_RESULTS_CSV = TESTBED_ROOT / "results" / "results.csv"


@dataclass
class RunSpec:
    challenge_id: str
    interface: str                  # e.g. "hopsworks", "none"
    mode: str                       # interface type: cli/mcp/sdk/none
    skills: str = "none"            # bundle name under testbed/skills/ or "none"
    model: str = DEFAULT_MODEL
    auth: str = "api-key"
    timeout_s: int = 60 * 60
    runs_root: Path = Path("results")
    data_root: Path = DEFAULT_DATA_ROOT
    interface_version: int | None = None  # None → latest version
    skills_version: int | None = None     # None → latest version


def _make_venv(target: Path) -> Path:
    if not (target / "bin" / "python").exists():
        venv.EnvBuilder(with_pip=True, clear=False).create(target)
    return target / "bin" / "python"


TASK_PROMPT_PATH = TESTBED_ROOT / "prompts" / "task.md"


def _build_prompt(challenge_id: str, fragment: str) -> str:
    template = TASK_PROMPT_PATH.read_text()
    return template.format(challenge_id=challenge_id, fragment=fragment).strip()


def run(spec: RunSpec) -> results.Row:
    if spec.auth not in AUTH_MODES:
        raise ValueError(f"Unknown auth mode {spec.auth!r}; expected one of {AUTH_MODES}")

    started = datetime.now(timezone.utc)
    version, variant_hash = interfaces.variant_for(spec.interface, spec.mode, spec.interface_version)
    interface_dir = f"configs/interfaces/{spec.interface}/{spec.mode}.yaml"
    prompt_file = f"{interface_dir}#versions.{version}.prompt"
    prompt_version = version
    prompt_hash = interfaces.prompt_hash_for(spec.interface, spec.mode, version)
    # Fail fast on unverified skill bundles — before spending time on
    # venv/data/prep we want to know the bundle is well-formed.
    if spec.skills == "none":
        skills_version, skills_hash = 0, ""
        skills_dir = ""
        skills_label = "no_skills"
    else:
        skills_version, skills_hash, _ = skills_mod.verify_installed(spec.skills, spec.skills_version)
        skills_dir = f"configs/skills/{spec.skills}/{skills_version}"
        skills_label = spec.skills
    run_dir = (
        spec.runs_root
        / spec.interface
        / spec.mode
        / skills_label
        / f"v{version}"
        / spec.challenge_id
    )
    # Same combo re-runs overwrite the previous output.
    if run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    venv_python = _make_venv(run_dir / "venv")
    (run_dir / "submission").mkdir(exist_ok=True)

    # mle-bench data prep must succeed before we spin up Claude — without
    # data/ the solver has nothing to score against. Don't swallow.
    mlebench_wrapper.prepare(spec.challenge_id, run_dir, spec.data_root)

    interface_setup = interfaces.setup(spec.interface, spec.mode, run_dir, venv_python, spec.interface_version)
    skills_setup = skills_mod.apply(spec.skills, run_dir, spec.skills_version)
    if skills_setup.installed:
        print(
            f"[banter] skills v{skills_setup.version} ({skills_setup.hash}): "
            f"{','.join(skills_setup.installed)}",
            file=sys.stderr,
        )

    prompt = _build_prompt(spec.challenge_id, interface_setup.prompt_fragment)
    (run_dir / "prompt.txt").write_text(prompt)

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
    results.write_readable_transcript(cr.transcript_path, run_dir / "transcript.log")

    row = results.Row(
        started_at=started.isoformat(),
        challenge_id=spec.challenge_id,
        interface=spec.interface,
        mode=spec.mode,
        skills=spec.skills,
        skills_version=skills_version,
        skills_hash=skills_hash,
        skills_dir=skills_dir,
        version=version,
        hash=variant_hash,
        interface_dir=interface_dir,
        prompt_version=prompt_version,
        prompt_hash=prompt_hash,
        prompt_file=prompt_file,
        auth=spec.auth,
        model=spec.model,
        wall_time_s=round(cr.wall_time_s, 2),
        medal=grading.get("medal"),
        score=grading.get("score"),
        run_dir=str(run_dir),
        **usage,
        **counts,
    )

    # Per-session CSV (autoresearch/benchmark session dir, or the standalone
    # `banter run` root) + global cross-session aggregate. When they resolve
    # to the same path (standalone runs into results/), we only write once
    # — append() dedupes by run_dir anyway, but the comparison saves I/O.
    session_csv = spec.runs_root / "results.csv"
    results.append(GLOBAL_RESULTS_CSV, row)
    if session_csv.resolve() != GLOBAL_RESULTS_CSV.resolve():
        results.append(session_csv, row)
    return row
