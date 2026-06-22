"""One-shot treatment runner.

A "treatment" is one config: a platform × model × skills arm run over the eval
tasks. Runs every (task × interface × skills) combo `repeats` times,
prints a summary, and writes detailed results.

Layout (under the caller-supplied `runs_root`):
    results/<config>/                                # the config yaml's stem
        <model>/<platform>/<interface>/<version>/<skills|no-skills>/
            <category>/<task>/<n>/   # one attempt; repeats accumulate as n+1
    results/results.csv     # the ONE CSV: one row per execution, all
                            # configs, appended after EVERY run
    results/results.ipynb   # GLOBAL: raw list + averages + charts,
                            # regenerated after EVERY run

Per-leaf results.csv files (next to the attempt folders) are deprecated — any
legacy ones found are merged into the global CSV at run start, deduped by
run_dir. Re-running a config never overwrites: each rerun appends the next
/<n> attempt folder and gets the next repeat number n in the global CSV.

Config format (YAML):

    model: claude-sonnet-4-6            # agent model
    auth: api-key                       # agent auth

    # Option A — explicit run list
    runs:
      - task: training_data
        category: feature
        platform: none
        interface: none
        skills: none

    # Option B — Cartesian matrix: `tasks:` maps <category>: [<task>, …]
    # tasks: {feature: [training_data]}
    # interfaces: [{platform: hopsworks, interface: cli}]
    # skills: [none]

    max_seconds: 1200  # OPTIONAL per-agent-run wall-clock cap in SECONDS
                       # (legacy: `max_min`, `timeout_s`). Absent/null → NO cap.

    concurrency: 3     # OPTIONAL — run this many of the config's runs at once,
                       # each in its OWN process and its OWN Hopsworks project
                       # (alias: `parallel`). Default 1 = sequential. Orthogonal
                       # to launching several `mlpab start` in separate terminals
                       # (each of those is already its own process + project).
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from mlpab import claude_runner, interfaces
from mlpab import preflight as preflight_mod
from mlpab import results, runner, skills

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RunEntry:
    task: str  # FTI sub-task (an evals family)
    platform: str
    interface: str
    skills: str = "none"
    category: str = "no_task"  # FTI category (a folder in the run path)
    model: str = ""  # agent model; "" → filled from the config's model list


@dataclass
class TreatmentConfig:
    runs: list[RunEntry]
    model: str = runner.DEFAULT_MODEL
    auth: str = "api-key"
    # Manual per-million-token prices keyed by model id, from the config's
    # `prices:` block — e.g. {"mistral-medium-3-5": {"input": 1.5, "output": 7.5}}.
    # Used to cost models litellm can't price; absent → litellm → 0.
    prices: dict = field(default_factory=dict)
    # Per-agent-run wall-clock cap, in seconds. None = NO cap (the default):
    # runs go to completion; only Ctrl-C or the agent finishing ends them.
    max_seconds: float | None = None
    # How many times to run EVERY combo (config key `n:` or `repeats:`). Each
    # repeat lands in its own …/<task>/<n> attempt folder and gets its own
    # results.csv row (n = 1..repeats).
    repeats: int = 1
    # How many runs to execute CONCURRENTLY within this one `mlpab run`
    # (config key `concurrency:` / `parallel:`). 1 = sequential (the default,
    # unchanged behavior). >1 fans the runs out across that many worker
    # PROCESSES — each worker is its own process, so each run mints its OWN
    # HOPSWORKS_PROJECT (runner.py) and setup/teardown stay scoped to it: N runs
    # share one cluster + API key, one project per run, never sweeping each
    # other. Builds/prepared venvs are materialized once upfront (preflight), so
    # workers only clone read-only — no build barrier.
    concurrency: int = 1


def _parse_runs(data: dict[str, Any]) -> list[RunEntry]:
    """Return the run list from the config dict.

    Supports an explicit `runs:` list and a Cartesian matrix via `interfaces`,
    `skills`, and a `tasks:` mapping (`<category>: [<task>, …]`) so runs are
    grouped by FTI category.
    """
    if "runs" in data:
        entries = []
        for r in data["runs"]:
            if r.get("config"):
                platform, interface = interfaces.platform_interface_from_config(r["config"])
            else:
                platform, interface = r["platform"], r.get("interface", "none")
            entries.append(
                RunEntry(
                    task=r["task"],
                    platform=platform,
                    interface=interface,
                    skills=r.get("skills", "none"),
                    category=r.get("category", "no_task"),
                    model=r.get("model", ""),  # per-run override; else config model list
                )
            )
        return entries

    # Cartesian matrix fallback. Each platform entry references its config
    # (`config:`) or gives platform+interface.
    #
    # `skills` entries mirror `interfaces` entries: reference the platform's
    # skills manifest by config path —
    #     skills:
    #       - none
    #       - {config: configs/platforms/hopsworks/skills.yaml}            # single bundle
    #       - {config: configs/platforms/hopsworks/skills.yaml, bundle: official} # explicit
    # The OWNING platform is inferred from the path (its folder), so a bundle
    # applies only to that platform's runs; every other platform runs the combo
    # with skills=none (duplicate combos are deduped). Plain-string entries
    # (bare bundle names) apply to all platforms unchanged.
    # The `tasks:` mapping key is the FTI category; its values are tasks.
    task_category = [(t, str(cat)) for cat, ts in (data.get("tasks") or {}).items() for t in ts]
    iface_entries = data.get("interfaces") or []
    skills_list = data.get("skills") or ["none"]
    entries = []
    for tk, category in task_category:
        for iface in iface_entries:
            if not isinstance(iface, dict):
                raise ValueError(f"treatment `interfaces` entries must be mappings, got {iface!r}")
            if iface.get("config"):
                platform, interface = interfaces.platform_interface_from_config(iface["config"])
            else:
                platform, interface = iface["platform"], iface.get("interface", "none")
            for sk in skills_list:
                sk_platform = None  # None → bundle applies to every platform
                if isinstance(sk, str):
                    sk_name = sk
                elif isinstance(sk, dict) and sk.get("config"):
                    sk_platform, sk_name = _skills_from_config(sk)
                elif isinstance(sk, dict):
                    sk_name = sk.get("name", "none")
                else:
                    raise ValueError(f"treatment `skills` entry must be str or mapping, got {sk!r}")
                if sk_platform is not None and sk_platform != platform:
                    # Foreign-platform bundle: this combo runs without skills
                    # (deduped against an explicit `none` entry below).
                    sk_name = "none"
                entries.append(
                    RunEntry(
                        task=tk,
                        platform=platform,
                        interface=interface,
                        skills=sk_name,
                        category=category,
                    )
                )
    # Dedup: a foreign-platform bundle degrades to skills=none and would
    # duplicate the explicit `none` combo for that interface × task.
    seen: set[tuple] = set()
    unique: list[RunEntry] = []
    for e in entries:
        key = (e.task, e.platform, e.interface, e.skills, e.category)
        if key in seen:
            continue
        seen.add(key)
        unique.append(e)
    return unique


def _skills_from_config(sk: dict[str, Any]) -> tuple[str, str]:
    """Resolve a `{config: configs/platforms/<p>/skills.yaml}` skills entry to
    (owning_platform, bundle_name). The platform is the config's folder name
    (mirroring `platform_interface_from_config`); the bundle is `bundle:` when
    given, else the platform's single bundle. Resolution goes through
    `skills.bundle_names` (testbed-root-anchored manifest + default dir), not
    the raw config path — which would be CWD-relative and silently empty when
    mlpab runs from outside the testbed root."""
    platform = Path(sk["config"]).parent.name
    bundle = sk.get("bundle")
    if not bundle:
        names = skills.bundle_names(platform)
        if len(names) == 1:
            bundle = names[0]
        else:
            raise ValueError(
                f"skills config {sk['config']!r} declares {len(names)} bundles "
                f"({', '.join(names) or 'none'}); pick one with `bundle:`."
            )
    return platform, bundle


def load_config(path: Path) -> TreatmentConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    # `max_seconds` preferred; legacy `max_min` (× 60) and `timeout_s` accepted.
    # Absent (or explicit null) → NO per-run wall-clock cap ("no budget key =
    # unlimited").
    if data.get("max_seconds") is not None:
        max_seconds = float(data["max_seconds"])
    elif data.get("max_min") is not None:
        max_seconds = float(data["max_min"]) * 60.0
    elif data.get("timeout_s") is not None:
        max_seconds = float(data["timeout_s"])
    else:
        max_seconds = None
    # `model` is a research knob; it may be a single id OR a list (the matrix
    # then expands over every model). A per-run `model:` in an explicit `runs:`
    # entry wins over the list.
    raw_model = data.get("model", runner.DEFAULT_MODEL)
    model_list = [raw_model] if isinstance(raw_model, str) else [str(m) for m in (raw_model or [])]
    model_list = model_list or [runner.DEFAULT_MODEL]
    runs, seen = [], set()
    for e in _parse_runs(data):
        for m in [e.model] if e.model else model_list:
            key = (m, e.task, e.platform, e.interface, e.skills, e.category)
            if key in seen:
                continue
            seen.add(key)
            runs.append(replace(e, model=m))
    return TreatmentConfig(
        runs=runs,
        # Auth is a machine/setup concern: defaults to MLPAB_AUTH (what `make
        # setup` chose) unless the config explicitly overrides it.
        model=model_list[0],  # representative; per-run model lives on the RunEntry
        auth=data.get("auth", os.environ.get("MLPAB_AUTH", "api-key")),
        prices=dict(data.get("prices") or {}),
        max_seconds=max_seconds,
        repeats=max(1, int(data.get("n", data.get("repeats", 1)) or 1)),
        concurrency=max(1, int(data.get("concurrency", data.get("parallel", 1)) or 1)),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def check_readiness(config: "TreatmentConfig") -> tuple[bool, list[tuple[str, bool, str]]]:
    """Is every agent engine the config needs usable (CLI on PATH + auth)? Returns
    (all_ready, [(model, ok, detail), …]). Dispatches per model id to the right
    engine: gpt-*/codex → Codex, mistral-* → Vibe, else Claude."""
    from mlpab import codex_runner, mistral_runner

    report: list[tuple[str, bool, str]] = []
    for m in sorted({e.model for e in config.runs}):
        if codex_runner.is_codex_model(m):
            ok, detail = codex_runner.engine_ready()
        elif mistral_runner.is_mistral_model(m):
            ok, detail = mistral_runner.engine_ready()
        else:
            ok, detail = claude_runner.engine_ready(config.auth)
        report.append((m, ok, detail))
    return all(ok for _, ok, _ in report), report


def print_readiness(
    report: list[tuple[str, bool, str]], title: str = "agent-engine readiness"
) -> None:
    print(f"\n[mlpab] {title}:")
    for model, ok, detail in report:
        mark = "✓" if ok else "✗"
        print(f"  {mark} {model:20} {detail}")


def check_llm_live(
    config: "TreatmentConfig",
    timeout_s: int = 60,
) -> tuple[bool, list[tuple[str, bool, str]]]:
    """Live LLM responsiveness: send each unique model a one-word prompt through
    its agent engine and confirm it answers. Returns (all_ok, [(model, ok, detail)]).
    Dispatches per model id like check_readiness."""
    from mlpab import codex_runner, mistral_runner

    report: list[tuple[str, bool, str]] = []
    for m in sorted({e.model for e in config.runs}):
        if codex_runner.is_codex_model(m):
            ok, detail = codex_runner.engine_live(m, timeout_s=timeout_s)
        elif mistral_runner.is_mistral_model(m):
            ok, detail = mistral_runner.engine_live(m, timeout_s=timeout_s)
        else:
            ok, detail = claude_runner.engine_live(m, auth=config.auth, timeout_s=timeout_s)
        report.append((m, ok, detail))
    return all(ok for _, ok, _ in report), report


def _pool_worker_init() -> None:
    """Initializer for the concurrency ProcessPool workers.

    Force quiet so concurrent runs don't interleave their live agent streams on
    the shared terminal — each run's full output is still teed to its own
    task/agent.log. Workers inherit the parent's env (creds, OAuth token cache)
    at spawn; we only add the quiet flag here.
    """
    os.environ["MLPAB_QUIET"] = "1"


def run_treatments(
    config: TreatmentConfig,
    runs_root: Path,
    config_name: str | None = None,
    assume_yes: bool = False,  # kept for CLI compat; runs ACCUMULATE, never prompt
    skip_existing: bool = True,
    retry: bool = False,
) -> None:
    total = len(config.runs) * config.repeats
    if total == 0:
        print("[mlpab] No runs configured.", file=sys.stderr)
        return

    # Fail fast, once, over the union of requirements before doing any work.
    reqs = [
        preflight_mod.Requirement(
            platform=e.platform,
            interface=e.interface,
            skills=e.skills,
        )
        for e in config.runs
    ]
    # Cheapest gate first: a no-build, no-network availability check that every
    # platform/interface/skill is even usable (config present, creds set, skill
    # bundle well-formed). Catches a forgotten credential or a missing bundle in
    # ~a second, instead of after minutes of building. The live connection +
    # interface build/test happen in the preflight below.
    preflight_mod.check_availability(reqs)
    # Fail fast if the agent ENGINE any model needs isn't ready (CLI + auth) —
    # before building platforms or launching a single run.
    ready, report = check_readiness(config)
    print_readiness(report)
    if not ready:
        raise preflight_mod.PreflightError(
            "agent engine(s) not ready (see above) — run `mlpab setup`, or fix the "
            "missing CLI/credential, before running this config."
        )
    # Fail fast on unimplemented eval families BEFORE building anything.
    from mlpab import evals_provider

    for tk in sorted({e.task for e in config.runs}):
        evals_provider._family(tk)  # raises ValueError with the implemented list
    # Live per-model probe: confirm each model actually ANSWERS a one-word prompt.
    # A model id the account can't reach passes the static readiness check above
    # but then produces zero-activity runs (see docs/code_review.md §12.1) — catch
    # it here, before building platforms or launching a single run.
    live_ok, live_report = check_llm_live(config)
    print_readiness(live_report, title="agent model liveness (live probe)")
    if not live_ok:
        raise preflight_mod.PreflightError(
            "agent model(s) did not respond to a liveness probe (see above) — a "
            "model this config names cannot be reached; check the model id / "
            "account access before running."
        )
    # Build + test the platforms once at session start, AND materialize each
    # interface's PREPARED venv (interfaces.prepare, inside preflight). This is
    # the whole setup phase: every run below just CLONES its prepared venv
    # read-only, so no run builds or mutates shared state — parallel sessions
    # need no build barrier and never wait on one another. (Login is checked per
    # run by the runner, in each run's own venv.) The build artifacts + prepared
    # venvs MUST persist here for the runs that clone them, so we do NOT
    # cleanup_build.
    try:
        preflight_mod.preflight(
            reqs,
            auth=config.auth,
            model=config.model,
            check_login=False,
        )
    except preflight_mod.PreflightError as e:
        print(f"\n[mlpab] preflight failed:\n{e}", file=sys.stderr)
        raise

    parent = runs_root
    parent.mkdir(parents=True, exist_ok=True)
    # The config FILENAME stem names the results folder; else auto-increment.
    run_id = config_name or results.next_session_id(parent)
    # results/<config-name>/. Each entry nests under
    # <model>/<platform>/<interface>/<version>/<skills|no-skills>/<category>/<task>/<n>
    # — mirroring the results.csv identity columns. Re-running the same config
    # ACCUMULATES: each repeat lands in the next /<n> attempt folder (and gets
    # n+1 in results.csv) instead of overwriting.
    run_path = parent / run_id
    if run_path.exists():
        print(f"[mlpab] {run_path} exists — accumulating repeat attempts (n+1).", flush=True)
    run_path.mkdir(parents=True, exist_ok=True)

    # OAuth token side-channel: write the Keychain JWT to `<run>/.claude-oauth`
    # (mode 0600, gitignored, removed with the run) and point
    # MLPAB_TOKEN_CACHE at it. claude_runner.resolve_oauth_token() reads
    # the cache when the agent's redirected HOME has no on-disk creds, so the
    # token never has to be committed or left at the repo root.
    token = claude_runner.oauth_token_from_keychain()
    if token:
        cache_path = claude_runner.write_token_cache(token, run_path.resolve())
        os.environ[claude_runner.TOKEN_CACHE_ENV] = str(cache_path)

    # The GLOBAL results.csv at the results root is the ONLY csv: the runner
    # appends each row straight into it after every run (per-leaf results.csv
    # files are deprecated). Refreshing it here folds any legacy leaf rows
    # (sessions run before the deprecation, or a crashed session's leftovers)
    # into the global table — and pre-creates it (header only) on first use.
    global_csv = parent / "results.csv"
    try:
        results.roll_up_results(parent, global_csv)
    except Exception as e:
        print(f"[mlpab] legacy leaf-CSV merge skipped: {e}", flush=True)
    global_nb = parent / "results.ipynb"
    if not global_nb.exists():
        import nbformat as _nbf
        from nbformat.v4 import new_markdown_cell, new_notebook

        nb = new_notebook()
        nb.cells = [
            new_markdown_cell(
                "# Results — global analysis\n\n"
                "_(No results yet — this notebook regenerates at end of every "
                "treatment run.)_\n"
            )
        ]
        with global_nb.open("w") as f:
            _nbf.write(nb, f)

    print(f"[mlpab] run={run_id}  runs={total}  dir={run_path}")

    completed: list[results.Row] = []
    failed: list[str] = []

    # Build every run's spec UP FRONT, assigning each its own /<n> attempt
    # folder. Attempt numbers are computed here (not inside the run) so the
    # repeats of one combo get DISTINCT, collision-free attempts even when the
    # runs execute concurrently — the old "read the disk max inside the loop"
    # scheme raced once two runs of the same combo ran at once. We seed each
    # leaf's counter from the on-disk max so re-running a config still
    # accumulates (n+1) across separate invocations.
    # Classify this config's existing rows by combo identity (everything but the
    # repeat `n`), scoped to THIS config_name so another config's rows never mask
    # a combo:
    #   * done — has at least one COMPLETED (valid=True) attempt.
    #   * failed — has rows but NONE valid=True (agent died, timed out, or never
    #     produced a deliverable).
    valid_by_combo: dict[tuple[str, ...], bool] = {}
    for r in results._read_runs(global_csv):
        if r.get("config") != run_id:
            continue
        key = results.combo_key(r)
        is_valid = str(r.get("valid", "")).strip().lower() == "true"
        valid_by_combo[key] = valid_by_combo.get(key, False) or is_valid
    done_combos = {k for k, ok in valid_by_combo.items() if ok}
    failed_combos = {k for k, ok in valid_by_combo.items() if not ok}

    # --retry: a failed combo carries no usable result, so purge its stale CSV
    # rows (its on-disk attempt dirs are removed in the plan loop below) and
    # re-run it clean. WITHOUT --retry the default --skip leaves failed combos
    # exactly as-is (skipped like any other existing row); only a missing combo
    # runs. So `retry_combos` (the set we purge + re-run) is the failed set under
    # --retry and empty otherwise.
    retry_combos = failed_combos if retry else set()
    if retry:
        pruned = results.prune_runs(global_csv, run_id, retry_combos)
        if pruned:
            print(f"[mlpab] --retry: removed {pruned} failed run row(s) — re-running them.", flush=True)

    expanded = [e for e in config.runs for _ in range(config.repeats)]
    next_attempt: dict[Path, int] = {}
    cleaned_dead: set[Path] = set()
    plan: list[tuple[runner.RunSpec, str]] = []
    skipped = 0
    for entry in expanded:
        # Nest by model/platform/interface/version/skills — the results.csv identity
        # columns — so no two combos share a <category>/<task> dir. The version
        # segment is the manifest's pinned version/ref; the local baseline has
        # no interface (and no manifest), so it reads "none"; skills "none"
        # reads as the literal "no-skills".
        if entry.platform == "none" and entry.interface == "none":
            version_seg = "none"
        else:
            version_seg = interfaces.interface_ref(
                interfaces.load_manifest(entry.platform, entry.interface)
            )
        combo_key = (
            entry.model,
            entry.platform,
            entry.interface,
            version_seg,
            entry.skills or "none",
            entry.category,
            entry.task,
        )
        # Default --skip leaves EVERY existing combo alone (done or failed); only
        # --retry pulls failed combos back out for a re-run, so they fall through
        # this skip into the plan below.
        skip_combos = done_combos if retry else (done_combos | failed_combos)
        if skip_existing and combo_key in skip_combos:
            skipped += 1
            continue
        skills_seg = entry.skills if entry.skills and entry.skills != "none" else "no-skills"
        leaf_root = (
            run_path / entry.model / entry.platform / entry.interface / version_seg / skills_seg
        )
        task_dir = leaf_root / entry.category / entry.task
        # A retried combo's stale CSV rows were already pruned; clear its on-disk
        # attempt dirs too (once) so the re-run starts at attempt 1 and fully
        # replaces the corpse instead of accumulating beside it.
        if combo_key in retry_combos and task_dir not in cleaned_dead:
            cleaned_dead.add(task_dir)
            if task_dir.is_dir():
                shutil.rmtree(task_dir)
        if task_dir not in next_attempt:
            existing = (
                [int(d.name) for d in task_dir.iterdir() if d.is_dir() and d.name.isdigit()]
                if task_dir.is_dir()
                else []
            )
            next_attempt[task_dir] = max(existing, default=0) + 1
        attempt = next_attempt[task_dir]
        next_attempt[task_dir] += 1
        spec = runner.RunSpec(
            task=entry.task,
            platform=entry.platform,
            interface=entry.interface,
            skills=entry.skills,
            model=entry.model,
            price=config.prices.get(entry.model),
            auth=config.auth,
            timeout_s=int(config.max_seconds) if config.max_seconds is not None else None,
            runs_root=leaf_root,
            run_id=run_id,  # tagged into the `config` column of the master CSV
            category=entry.category,
            attempt=attempt,
            preflight=False,  # union already verified upfront
            results_csv=global_csv,  # the ONE results CSV, appended per run
        )
        plan.append((spec, f"{entry.model} | {entry.task} | {entry.platform}/{entry.interface}"))

    # --skip dropped already-completed combos: the real workload is the plan, so
    # progress counters and concurrency size off it (not the pre-skip total).
    if skipped:
        print(
            f"[mlpab] --skip: {skipped}/{total} combo(s) already completed "
            f"(valid=True) — running the remaining {len(plan)}.",
            flush=True,
        )
    total = len(plan)
    if total == 0:
        print("[mlpab] nothing to run — every combo already completed (use --no-skip to rerun).")
        _print_summary(completed, failed, global_csv)
        return

    concurrency = min(max(1, config.concurrency), total)

    def _record(idx: int, spec: runner.RunSpec, row: results.Row) -> None:
        completed.append(row)
        print(
            f"[mlpab] [{idx}/{total}] done: {spec.task} {spec.platform}/{spec.interface}  "
            f"asserts={row.asserts_passed}/{row.total_asserts}  "
            f"tokens={row.total_tokens}  wall={row.wall_time_s:.1f}s  cost=${row.cost_usd:.4f}"
        )
        # The runner appended the row to the global results.csv; refresh the
        # global results.ipynb to match, so both stay current after EVERY run.
        _refresh_notebook(parent)

    if concurrency == 1:
        # Sequential path — unchanged behavior. A failed platform setup aborts
        # the whole config (every later run would hit the same broken platform).
        for idx, (spec, label) in enumerate(plan, 1):
            print(f"\n[treatment {idx}/{total}] {label} | skills={spec.skills}")
            try:
                _record(idx, spec, runner.run(spec))
            except preflight_mod.PlatformNotReadyError as exc:
                print(
                    f"\n[mlpab] ABORTING config {config_name!r} — platform "
                    f"{spec.platform!r} is not ready:\n{exc}",
                    file=sys.stderr,
                )
                print(f"[mlpab] results: {global_csv}", flush=True)
                _print_summary(completed, failed, global_csv)
                raise
            except Exception as exc:
                label = f"{spec.task}/{spec.platform}/{spec.interface}"
                print(f"[mlpab] FAILED {label}: {exc}", file=sys.stderr)
                failed.append(f"{label}: {exc}")
    else:
        # Concurrent path — fan out across worker PROCESSES. A process per run
        # means each run keeps its OWN os.environ (so the per-run HOPSWORKS_PROJECT
        # set in runner.run never clobbers another run's), and the results.csv
        # append is already flock-serialized across processes. Workers run quiet
        # (per-run agent.log still captures everything) so their live output does
        # not interleave on the terminal.
        from concurrent.futures import ProcessPoolExecutor, as_completed

        print(
            f"[mlpab] running {total} runs at concurrency={concurrency} "
            f"(one Hopsworks project per run; live output → each run's agent.log)"
        )
        with ProcessPoolExecutor(
            max_workers=concurrency, initializer=_pool_worker_init
        ) as pool:
            futures = {
                pool.submit(runner.run, spec): (i, spec) for i, (spec, _label) in enumerate(plan, 1)
            }
            for fut in as_completed(futures):
                idx, spec = futures[fut]
                try:
                    _record(idx, spec, fut.result())
                except Exception as exc:
                    # PlatformNotReadyError included: with runs already in flight
                    # we cannot cleanly abort the others, so record it as a
                    # failure rather than tearing the pool down mid-run.
                    label = f"{spec.task}/{spec.platform}/{spec.interface}"
                    print(f"[mlpab] FAILED {label}: {exc}", file=sys.stderr)
                    failed.append(f"{label}: {exc}")

    print(f"[mlpab] results: {global_csv}", flush=True)
    _print_summary(completed, failed, global_csv)


def _refresh_notebook(parent: Path) -> None:
    """Regenerate the GLOBAL results.ipynb at the results root from the
    global results.csv: per config → per model → one bar chart per
    TRACKED_METRICS metric, one bar per (platform, interface, version,
    skills), averaged across categories, tasks, and repeats. Replaces the
    placeholder created at run start. Failures never abort the session."""
    try:
        from mlpab import notebook as notebook_mod

        nb_path = notebook_mod.build_results_notebook(parent)
        if nb_path is not None:
            print(f"[mlpab] refreshed analysis notebook: {nb_path}", flush=True)
    except Exception as e:
        print(f"[mlpab] notebook refresh skipped: {e}", flush=True)


def _print_summary(rows: list[results.Row], failed: list[str], rollup_csv: Path) -> None:
    w = 110
    print("\n" + "=" * w)
    print("RESULTS SUMMARY")
    print("=" * w)
    if rows:
        print(
            f"{'task':<35} {'platform/interface':<20} {'skills':<18} "
            f"{'asserts':<8} {'tokens':<8} {'wall_s':<7} cost"
        )
        print("-" * w)
        for r in rows:
            print(
                f"{r.task:<35} {r.platform}/{r.interface:<18} {r.skills:<18} "
                f"{f'{r.asserts_passed}/{r.total_asserts}':<8} "
                f"{r.total_tokens:<8} {r.wall_time_s:<7.1f} ${r.cost_usd:.4f}"
            )
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")
    print(f"\nResults CSV: {rollup_csv}")
    print("=" * w)
