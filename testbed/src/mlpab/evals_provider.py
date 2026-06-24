"""Task provider: generated FTI evals.

The treatment config's `tasks:` mapping is `<category>: [<task>, …]`
(e.g. `feature: [training_data]`); each task names ONE eval implemented
by a generator family under `evals/`. The runner depends on exactly:

    prepare(task, run_dir, seed)   -> stage data/ + return the task body
    grade(task, run_dir, platform, venv_python) -> report dict

Layout per run (everything lives IN the attempt folder):
    <run_dir>/data/        inputs the agent sees
    <run_dir>/prompt.txt   assembled by the runner from the task body
    <run_dir>/task/        the task spec (instance.json + the generator prompt)
    <run_dir>/solution/    the ANSWER KEY (truth + known-wrong variants) —
                           DENIED to the agent by the PreToolUse hook (path
                           deny on <boundary>/solution for file tools, bash,
                           and search tools), while the grader reads it freely

Grading reads the deliverable back THROUGH the platform via a checker adapter
(evals/adapters/), executed with the run venv's python — in treatment runs its
interface package is installed from the committed pinned wheel, so the client
is trusted. Platform `none` (local baseline) grades the local file the prompt
names instead.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import zlib
from pathlib import Path
from typing import Any

# Task name -> (module under evals/, deliverable kind).
# One eval per FTI task. Entries are added as their generators land; an
# unimplemented name fails fast at prepare().
# Deliverable kinds:
#   dataset — a versioned training dataset ON the platform; graded through a
#             checker adapter (platform `none` grades submission/<name>.csv)
#   table   — a feature table ON the platform; graded by reading the table
#             back through a checker adapter (platform `none` grades
#             submission/<table_name>.csv)
#   answers — a local submission/answers.json verdict (works on any platform;
#             the investigation may use the platform, the answer is a file)
#   platform — the grade module decides what to read (tables, answers, AND
#             platform state like models/jobs/endpoints via the adapter's
#             state reads); it always receives `--adapter <platform|none>`
FAMILIES: dict[str, tuple[str, str]] = {
    # F — feature pipelines (evals/feature/)
    "ingest": ("feature.ingest", "table"),  # full load w/ overlap + mixed ts
    "backfill": ("feature.backfill", "table"),  # out-of-order corrections
    "mit": ("feature.mit", "table"),  # model-independent transforms
    "validate": ("feature.validate", "table"),  # data contract + quarantine report
    "incremental_load": ("feature.incremental_load", "platform"),  # increments + recurring job
    "full_reload": ("feature.full_reload", "table"),  # schema-breaking re-create (v2)
    "training_data": ("feature.pit", "dataset"),  # PIT-correct training dataset
    "leakage": ("feature.leakage", "answers"),  # leaky-feature detection
    # T — training pipelines (evals/training/)
    "train": ("training.train", "platform"),  # provided script as a platform job
    "mdt": ("training.mdt", "table"),  # model-dependent transforms (train-split stats)
    "register": ("training.register", "platform"),  # model registry entry + metrics
    "llm_finetuning": (
        "training.llm_finetuning",
        "platform",
    ),  # LLM fine-tune as a platform job + registry entry
    # I — inference pipelines (evals/inference/)
    "batch": ("inference.batch", "table"),  # batch scoring as-of T
    "online": ("inference.online", "platform"),  # online lookups recorded + verified
    "odt": ("inference.odt", "table"),  # on-demand request-time feature
    "skew": ("inference.skew", "answers"),  # training/serving skew detection
    "llm_serving": ("inference.llm_serving", "platform"),  # deploy + invoke a real-time scorer
    "recsys": ("inference.recsys", "table"),  # top-k retrieval excl. seen
    "vector_search": ("inference.vector_search", "platform"),  # native ANN/KNN top-k by L2
    # Ops — observability & operations (evals/ops/)
    "drift": ("ops.drift", "answers"),  # data drift detection
    "prediction_monitoring": ("ops.prediction_monitoring", "answers"),  # prediction drift onset
    "scheduled_jobs": ("ops.scheduled_jobs", "platform"),  # recurring heartbeat job
    "alerting": ("ops.alerting", "platform"),  # failure alert on a failing job
    "lineage": ("ops.lineage", "platform"),  # derivation chain + lineage answer
    # Capstone — end-to-end FTI modelling tasks (evals/capstone/); hybrid grade:
    # held-out predictive metric + on-platform FTI artifacts (feature group,
    # training dataset, registered model) read back through the adapter.
    "ccfraud": ("capstone.ccfraud", "platform"),  # classification (fraud) — ROC AUC
    "airquality": ("capstone.airquality", "platform"),  # regression (PM2.5) — RMSE
}

# Platforms with a checker adapter (grading reads through the platform).
ADAPTERS = ("hopsworks", "databricks", "aws", "azure", "gcp")

# Upper bound on deterministic reseeds when a generator's validity gate rejects
# a seed (see prepare()). Gate rejections are rare, so a clean seed is normally
# found in the first one or two tries; this is just a guard against a generator
# whose gate can never pass.
_MAX_RESEEDS = 50


def seed_for(run_id: str, category: str, task: str, attempt: int) -> int:
    """Deterministic, distinct seed per (session, category, task, repeat):
    reproducible rows, fresh instance every repeat."""
    key = f"{run_id}:{category}:{task}:{attempt}"
    return zlib.crc32(key.encode()) & 0x7FFFFFFF


def _family(task: str) -> tuple[str, str]:
    fam = FAMILIES.get(task)
    if fam is None:
        raise ValueError(
            f"no eval generator implemented for task {task!r} — "
            f"implemented: {', '.join(sorted(FAMILIES))}"
        )
    return fam


def prepare(task: str, run_dir: Path, seed: int) -> str:
    """Generate a fresh instance (validity gates run inside the generator).

    `run_dir` is the agent's workspace — the `task/` child of the attempt
    folder, and the agent's sandbox boundary. The instance's data/ goes in
    there; the ANSWER KEY (plus the task spec: instance.json + the
    generator's prompt, which embeds the seed) goes to the SIBLING
    `<attempt>/solution/` — outside the boundary, invisible to the agent.
    Returns the task body for the prompt."""
    import importlib

    fam, _kind = _family(task)
    run_dir = Path(run_dir).absolute()
    attempt = run_dir.parent
    staging = attempt / ".staging"

    gen = importlib.import_module(f"evals.{fam}.generate")
    # A generator's validity gates (e.g. cross-checking the answer key against
    # an independent implementation) can reject a seed: the random instance it
    # drew is one the gate can't certify (for `mit`, two transactions of one
    # account share an event_time, which the pandas rolling cross-check counts
    # differently from the row scan). Because the seed is deterministic per
    # (config, category, task, attempt), such a seed would kill that combo on
    # EVERY run. So on a gate rejection, deterministically reseed and retry —
    # a clean instance for the same combo, still fully reproducible. A seed that
    # passes the gate (the common case) succeeds on the first iteration, so
    # already-valid instances are byte-for-byte unchanged.
    gate_err = getattr(gen, "GateError", ())
    cur_seed = seed
    last_gate: Exception | None = None
    for _ in range(_MAX_RESEEDS):
        if staging.exists():
            shutil.rmtree(staging)
        try:
            gen.generate(cur_seed, staging)
            break
        except gate_err as e:  # type: ignore[misc]  # () catches nothing when no gate
            last_gate = e
            cur_seed = zlib.crc32(f"{cur_seed}:reseed".encode()) & 0x7FFFFFFF
    else:
        raise RuntimeError(
            f"could not generate a gate-valid {task!r} instance after "
            f"{_MAX_RESEEDS} reseeds (seed {seed}): {last_gate}"
        )

    for target in (run_dir / "data", attempt / "solution"):
        if target.exists():
            shutil.rmtree(target)
    shutil.move(str(staging / "data"), str(run_dir / "data"))
    shutil.move(str(staging / "solution"), str(attempt / "solution"))
    body = (staging / "prompt.txt").read_text()
    for name in ("instance.json", "prompt.txt"):
        if (staging / name).exists():
            shutil.move(str(staging / name), str(attempt / "solution" / name))
    shutil.rmtree(staging, ignore_errors=True)
    return body


def _plumbing_error(msg: str) -> dict[str, Any]:
    """A grading report for a failure BEFORE the grader could enumerate its
    suite (no adapter, subprocess died, no parseable report). The full suite
    size is unknowable here, so the tally fields are all zero."""
    return {
        "success": False,
        "asserts_passed": 0,
        "asserts_failed": 0,
        "asserts_skipped": 0,
        "total_asserts": 0,
        "asserts": [],
        "error": msg,
    }


def grade(task: str, run_dir: Path, platform: str, venv_python: Path) -> dict[str, Any]:
    """Grade the run. Returns the report dict (always contains `success`,
    `asserts_passed`, `asserts_failed`, `asserts_skipped`, `total_asserts`;
    `error` on plumbing failures).

    `dataset` deliverables: platform with an adapter → read the deliverable
    back through the platform, using the run venv's python (its evals package +
    platform client come from the committed base install); platform `none` →
    grade the local file the prompt named (`submission/<dataset>.csv`); other
    platforms → explicit "no checker adapter" failure, so the row is honest
    about why it scored 0. `answers` deliverables: grade the local
    submission/answers.json on every platform.
    """
    fam, kind = _family(task)
    # The grader subprocess runs with cwd = the agent workspace (run_dir =
    # <attempt>/task), so every path it receives must be ABSOLUTE (.absolute(),
    # not .resolve() — the venv python is a symlink that must not be followed
    # out of the venv). The instance root is the ATTEMPT folder: its solution/
    # sibling holds the answer key the graders read.
    run_dir = Path(run_dir).absolute()
    inst = run_dir.parent
    cmd = [str(Path(venv_python).absolute()), "-m", f"evals.{fam}.grade", "--instance", str(inst)]

    if kind == "answers":
        cmd += ["--answers", str(Path(run_dir) / "submission" / "answers.json")]
    elif kind == "platform":
        cmd += ["--adapter", platform if platform in ADAPTERS else "none"]
    elif platform in ADAPTERS:
        cmd += ["--adapter", platform]
    elif platform == "none":
        meta = json.loads((inst / "solution" / "truth.json").read_text())
        deliverable = meta.get("dataset_name") or meta.get("table_name")
        local = Path(run_dir) / "submission" / f"{deliverable}.csv"
        # Pass --csv even when the local deliverable is missing — the grader reads
        # a missing/empty file as an empty deliverable and still enumerates its
        # full assert suite (deliverable check fails, dependents skip).
        cmd += ["--csv", str(local)]
    else:
        return _plumbing_error(
            f"no checker adapter for platform {platform!r} — "
            f"available: {', '.join(ADAPTERS)} (+ `none` for local)"
        )

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(run_dir),
            # PYTHONSAFEPATH stops Python prepending cwd (the agent workspace) to
            # sys.path under `-m`, so agent-authored files (e.g. a bisect.py) can't
            # shadow stdlib/dependency modules the grader itself imports.
            env={**os.environ, "PYTHONSAFEPATH": "1"},
        )
    except Exception as e:
        return _plumbing_error(f"grader failed to run: {e}")
    report = _extract_report(proc.stdout)
    if report is not None:
        return report
    return _plumbing_error(
        f"grader produced no report (exit {proc.returncode}): {(proc.stderr or proc.stdout)[-500:]}"
    )


def _extract_report(stdout: str) -> dict[str, Any] | None:
    """The report JSON from the grader's stdout, tolerating surrounding noise.

    Platform clients log to STDOUT around the report (the Hopsworks SDK prints
    login/engine/closing lines before AND after it), so a bare json.loads on
    the full stream fails. Scan for the last balanced JSON object that looks
    like a report (has `success`)."""
    dec = json.JSONDecoder()
    starts = [i for i, ch in enumerate(stdout) if ch == "{"]
    for i in reversed(starts):
        try:
            obj, _ = dec.raw_decode(stdout[i:])
        except ValueError:
            continue
        if isinstance(obj, dict) and "success" in obj:
            return obj
    return None
