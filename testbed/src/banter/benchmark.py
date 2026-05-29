"""One-shot benchmark runner.

Runs a list of (challenge × interface × skills) entries once each, prints a
summary, and writes detailed results.

Layout (under the caller-supplied `runs_root`):
    results/benchmark/<id>__<iface>__<mode>__<skills>__no_prev_run__no_prev_v/
        <task>/<challenge>/        # the engineer run + its artifacts
        results.csv                # one row per challenge (incl. `task`)
    results/benchmark/results.csv  # global rollup: one row per run

The session-folder name mirrors autoresearch (minus the `v<N>` level since
benchmark has no versions); `no_prev_*` is fixed because benchmark doesn't
continue prior work.

Config format (YAML):

    engineer_model: claude-sonnet-4-6   # model for the engineer (legacy: model)
    engineer_auth: api-key              # engineer auth (legacy: auth)

    # Option A — explicit run list
    runs:
      - challenge: aerial-cactus-identification
        interface: none
        mode: none
        skills: none

    # Option B — Cartesian matrix
    # challenges: [aerial-cactus-identification]
    # interfaces: [{name: mlkit, mode: cli}]
    # skills: [none]

    max_seconds: 3600  # per-engineer-run wall-clock cap in SECONDS (legacy: `max_min`, `timeout_s`)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from banter import claude_runner, interfaces, preflight as preflight_mod, results, runner


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RunEntry:
    challenge: str
    interface: str
    mode: str
    skills: str = "none"
    docs: str = "none"
    interface_version: int | None = None
    skills_version: int | None = None
    # Autoresearch session that produced the pinned interface version (needed
    # to resolve interface_version > 0, which lives inside that session).
    session: str | None = None
    task: str = "no_task"           # ML task / challenge group (a folder in the run path)


@dataclass
class BenchmarkConfig:
    runs: list[RunEntry]
    engineer_model: str = runner.DEFAULT_MODEL
    engineer_auth: str = "api-key"
    # Per-engineer-run wall-clock cap, in seconds (matches autoresearch's budget.max_seconds).
    max_seconds: float = 3600.0


def _parse_runs(data: dict[str, Any]) -> list[RunEntry]:
    """Return the run list from the config dict.

    Supports an explicit `runs:` list and a Cartesian matrix via `interfaces`,
    `skills`, and either a flat `challenges:` list (task = "no_task") or a
    `tasks:` mapping (task name → [challenges]) so runs are grouped by task.
    """
    if "runs" in data:
        entries = []
        for r in data["runs"]:
            if r.get("config"):
                name, mode = interfaces.name_type_from_config(r["config"])
            else:
                name, mode = r["interface"], r.get("mode", "none")
            entries.append(
                RunEntry(
                    challenge=r["challenge"],
                    interface=name,
                    mode=mode,
                    skills=r.get("skills", "none"),
                    docs=r.get("docs", "none"),
                    interface_version=r.get("interface_version", r.get("version")),
                    skills_version=r.get("skills_version"),
                    session=r.get("session"),
                    task=r.get("task", "no_task"),
                )
            )
        return entries

    # Cartesian matrix fallback. Each interface entry references its config
    # (`config:`) or gives name+mode, and may pin a `version:` (+ `session:`).
    if data.get("tasks"):
        challenge_task = [(c, str(t)) for t, cs in data["tasks"].items() for c in cs]
    else:
        challenge_task = [(c, "no_task") for c in (data.get("challenges") or [])]
    iface_entries = data.get("interfaces") or []
    skills_list = data.get("skills") or ["none"]
    entries = []
    for ch, task in challenge_task:
        for iface in iface_entries:
            if not isinstance(iface, dict):
                raise ValueError(
                    f"benchmark `interfaces` entries must be mappings, got {iface!r}"
                )
            if iface.get("config"):
                name, mode = interfaces.name_type_from_config(iface["config"])
            else:
                name, mode = iface["name"], iface.get("mode", "none")
            for sk in skills_list:
                if isinstance(sk, str):
                    sk_name, sk_version = sk, None
                elif isinstance(sk, dict):
                    sk_name = sk.get("name", "none")
                    sk_version = sk.get("version")
                else:
                    raise ValueError(f"benchmark `skills` entry must be str or mapping, got {sk!r}")
                entries.append(
                    RunEntry(
                        challenge=ch,
                        interface=name,
                        mode=mode,
                        skills=sk_name,
                        docs=data.get("docs", "none"),
                        interface_version=iface.get("version"),
                        skills_version=sk_version,
                        session=iface.get("session"),
                        task=task,
                    )
                )
    return entries


def load_config(path: Path) -> BenchmarkConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    # `max_seconds` preferred; legacy `max_min` (× 60) and `timeout_s` accepted.
    if "max_seconds" in data:
        max_seconds = float(data["max_seconds"])
    elif "max_min" in data:
        max_seconds = float(data["max_min"]) * 60.0
    elif "timeout_s" in data:
        max_seconds = float(data["timeout_s"])
    else:
        max_seconds = 3600.0
    return BenchmarkConfig(
        runs=_parse_runs(data),
        # `engineer_model` is a research knob set in the config. Auth, however,
        # is a machine/setup concern: it defaults to BANTER_AUTH (what `make
        # setup` chose) unless the config explicitly overrides it.
        engineer_model=data.get("engineer_model", data.get("model", runner.DEFAULT_MODEL)),
        engineer_auth=data.get(
            "engineer_auth", data.get("auth", os.environ.get("BANTER_AUTH", "api-key"))
        ),
        max_seconds=max_seconds,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _version_root_for(session: str | None) -> Path | None:
    """Session dir holding pinned interface versions, or None for the base."""
    if not session:
        return None
    return runner.TESTBED_ROOT / "results" / "autoresearch" / session


def run_benchmark(config: BenchmarkConfig, runs_root: Path) -> None:
    total = len(config.runs)
    if total == 0:
        print("[benchmark] No runs configured.", file=sys.stderr)
        return

    # Fail fast, once, over the union of requirements before doing any work.
    reqs = [
        preflight_mod.Requirement(
            interface=e.interface,
            mode=e.mode,
            interface_version=e.interface_version,
            version_root=_version_root_for(e.session),
            skills=e.skills,
            skills_version=e.skills_version,
        )
        for e in config.runs
    ]
    try:
        # Build + test the interfaces once at session start (login is checked per
        # challenge by the runner, in each run's own venv).
        preflight_mod.preflight(
            reqs, auth=config.engineer_auth, model=config.engineer_model, check_login=False,
        )
    except preflight_mod.PreflightError as e:
        print(f"\n[benchmark] preflight failed:\n{e}", file=sys.stderr)
        raise

    # Pre-download every challenge's dataset NOW, while we're still in the
    # unsandboxed parent. Each engineer's claude -p subprocess won't be able
    # to write to <testbed>/cache once its sandbox is active. Idempotent.
    from banter import mlebench_wrapper
    challenges = sorted({e.challenge for e in config.runs})
    for n, comp in enumerate(challenges, 1):
        print(f"[benchmark] preparing data {n}/{len(challenges)}: {comp}", flush=True)
        mlebench_wrapper.download_competition(comp, runner.DEFAULT_DATA_ROOT)

    parent = runs_root / "benchmark"
    parent.mkdir(parents=True, exist_ok=True)
    run_id = results.next_session_id(parent)
    # Folder name mirrors autoresearch (minus the v<N> level since benchmark
    # has no versions): <id>__<iface>__<mode>__<skills>__no_prev_run__no_prev_v
    first = config.runs[0] if config.runs else None
    iface_seg = (first.interface if first and first.interface != "none" else "no_interface")
    type_seg = (first.mode if first and first.mode != "none" else "no_type")
    skills_seg = (first.skills if first and first.skills != "none" else "no_skills")
    run_dir_name = f"{run_id}__{iface_seg}__{type_seg}__{skills_seg}__no_prev_run__no_prev_v"
    # The runner writes <run>/<task>/<challenge>/ and the per-session results.csv
    # at <run>/results.csv (one row per challenge, incl. `task`).
    run_path = parent / run_dir_name
    run_path.mkdir(parents=True, exist_ok=True)

    # OAuth token side-channel, mirroring autoresearch: write the Keychain JWT
    # to `<run>/.claude-oauth` (mode 0600, gitignored, removed with the run) and
    # point BANTER_TOKEN_CACHE at it. claude_runner.resolve_oauth_token() reads
    # the cache when the engineer's redirected HOME has no on-disk creds, so the
    # token never has to be committed or left at the repo root.
    token = claude_runner.oauth_token_from_keychain()
    if token:
        cache_path = claude_runner.write_token_cache(token, run_path.resolve())
        os.environ[claude_runner.TOKEN_CACHE_ENV] = str(cache_path)

    # Pre-create empty results.csv (header only) and a placeholder analysis
    # notebook so they exist from the moment the run dir is created. The
    # runner appends rows; we replace the notebook at end of run.
    empty_csv = run_path / "results.csv"
    if not empty_csv.exists():
        import csv as _csv
        with empty_csv.open("w", newline="") as f:
            _csv.DictWriter(f, fieldnames=results.BENCHMARK_FIELDS).writeheader()
    empty_nb = run_path / "analysis.ipynb"
    if not empty_nb.exists():
        from nbformat.v4 import new_notebook, new_markdown_cell
        import nbformat as _nbf
        nb = new_notebook()
        nb.cells = [new_markdown_cell(
            f"# Benchmark run `{run_id}` — analysis\n\n"
            f"_(No results yet — this notebook regenerates at end of run.)_\n"
        )]
        with empty_nb.open("w") as f:
            _nbf.write(nb, f)

    print(f"[benchmark] run={run_id}  runs={total}  dir={run_path}")

    completed: list[results.Row] = []
    failed: list[str] = []

    for n, entry in enumerate(config.runs, 1):
        version_tag = f" v{entry.interface_version}" if entry.interface_version is not None else ""
        skills_tag = entry.skills
        if entry.skills_version is not None:
            skills_tag += f" v{entry.skills_version}"
        print(
            f"\n[benchmark {n}/{total}] {entry.challenge} | "
            f"{entry.interface}/{entry.mode}{version_tag} | skills={skills_tag}"
        )
        try:
            spec = runner.RunSpec(
                challenge_id=entry.challenge,
                interface=entry.interface,
                mode=entry.mode,
                skills=entry.skills,
                docs=entry.docs,
                model=config.engineer_model,
                auth=config.engineer_auth,
                timeout_s=int(config.max_seconds),
                runs_root=run_path,
                run_id=run_id,        # tagged into row.session in the master CSV
                interface_version=entry.interface_version,
                skills_version=entry.skills_version,
                task=entry.task,
                version_root=_version_root_for(entry.session),
                preflight=False,  # union already verified upfront
            )
            row = runner.run(spec)
            completed.append(row)
            print(
                f"[benchmark] score={row.score}  tokens={row.total_tokens}  "
                f"wall={row.total_wall_time_s:.1f}s  cost=${row.total_cost:.4f}"
            )
        except Exception as exc:
            label = f"{entry.challenge}/{entry.interface}/{entry.mode}"
            print(f"[benchmark] FAILED {label}: {exc}", file=sys.stderr)
            failed.append(f"{label}: {exc}")

    # Each benchmark session keeps its results.csv local to its run dir.
    # No global rollup — sessions are independent and shouldn't be averaged.
    _print_summary(completed, failed, run_path / "results.csv")

    # Generate analysis.ipynb with one bar chart per TRACKED_METRICS metric,
    # x = (task, challenge), y = metric value. Replaces the placeholder
    # created at run start.
    try:
        from banter import notebook as notebook_mod
        nb_path = notebook_mod.build_benchmark_notebook(run_path, run_id)
        if nb_path is not None:
            print(f"[benchmark] wrote analysis notebook: {nb_path}", flush=True)
    except Exception as e:
        print(f"[benchmark] notebook generation skipped: {e}", flush=True)


def _print_summary(rows: list[results.Row], failed: list[str], rollup_csv: Path) -> None:
    w = 110
    print("\n" + "=" * w)
    print("BENCHMARK SUMMARY")
    print("=" * w)
    if rows:
        print(
            f"{'challenge':<35} {'iface/type':<20} {'skills':<18} "
            f"{'score':<8} {'tokens':<8} {'wall_s':<7} cost"
        )
        print("-" * w)
        for r in rows:
            print(
                f"{r.challenge:<35} {r.interface}/{r.type:<18} {r.skills:<18} "
                f"{str(r.score or ''):<8} {r.total_tokens:<8} {r.total_wall_time_s:<7.1f} ${r.total_cost:.4f}"
            )
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")
    print(f"\nResults CSV: {rollup_csv}")
    print("=" * w)
