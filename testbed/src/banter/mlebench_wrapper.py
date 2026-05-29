"""Thin wrapper around the `mlebench` CLI.

Prepare downloads + stages the competition data into a workspace; grade
evaluates a submission.csv and returns the medal level + score.

The exact subcommand surface of mle-bench evolves; we shell out so changes
upstream don't require code edits here. The runner only depends on:

    prepare(competition_id, workspace_dir, data_dir)  -> workspace path with
        Kaggle-style data files Claude can read.
    grade(competition_id, submission_path)            -> dict with medal info.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


# Anchor mlebench subprocess CWDs at the testbed root. Importing
# `mlebench.data` creates a `./cache/cache.db` diskcache relative to CWD
# (see mlebench/data.py:29). Running banter from outside the repo would
# scatter that file into the caller's directory, so we always pin cwd
# here.
_TESTBED_ROOT = Path(__file__).resolve().parents[2]

# pip installs the leaderboard CSVs as Git LFS pointer files (~130 bytes
# starting with this magic). We detect that and force a Kaggle-API download
# during prepare, otherwise grade-sample raises "Leaderboard must have a
# `score` column".
_LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"


def _mlebench() -> str:
    """Locate the mlebench CLI inside banter's venv, then on PATH."""
    candidate = Path(sys.executable).parent / "mlebench"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("mlebench")
    if found:
        return found
    raise RuntimeError(
        "`mlebench` CLI not found. Install with "
        "`pip install git+https://github.com/openai/mle-bench.git`."
    )


def _leaderboard_path(competition_id: str) -> Path:
    """Locate the shipped leaderboard.csv inside the installed mlebench pkg."""
    import mlebench
    return Path(mlebench.__file__).parent / "competitions" / competition_id / "leaderboard.csv"


def _ensure_real_leaderboard(competition_id: str) -> None:
    """If the shipped leaderboard is still an LFS pointer, fetch it via Kaggle."""
    lb = _leaderboard_path(competition_id)
    if lb.exists() and not lb.read_text(errors="ignore").startswith(_LFS_POINTER_PREFIX):
        return
    cli = _mlebench()
    subprocess.run(
        [cli, "dev", "download-leaderboard", "-c", competition_id, "--force"],
        check=True,
        cwd=_TESTBED_ROOT,
    )


def download_competition(competition_id: str, data_dir: Path) -> Path:
    """Download + prepare a competition into `data_dir/<comp>/prepared/public`.

    Idempotent — skips the `mlebench prepare` invocation when the prepared
    dir already exists. Call this from the UNSANDBOXED parent process at
    session start so the cache is ready before any sandboxed work begins.
    Returns the prepared path.
    """
    prepared = data_dir / competition_id / "prepared" / "public"
    if prepared.is_dir() and any(prepared.iterdir()):
        return prepared
    cli = _mlebench()
    data_dir.mkdir(parents=True, exist_ok=True)
    _ensure_real_leaderboard(competition_id)
    subprocess.run(
        [
            cli,
            "prepare",
            "--competition-id",
            competition_id,
            "--data-dir",
            str(data_dir),
        ],
        check=True,
        cwd=_TESTBED_ROOT,
    )
    return prepared


def prepare(competition_id: str, workspace_dir: Path, data_dir: Path) -> Path:
    """Stage competition data under workspace_dir and return its path.

    Calls `download_competition` (idempotent — no-op if cache is warm) then
    materializes `workspace_dir/data` as an APFS clone of the prepared dir.
    We can't symlink here because Seatbelt resolves symlinks to their target
    before checking allowRead — the engineer's sandbox would deny reads of
    the cache target. APFS `cp -Rc` clones share blocks until modified, so
    even multi-GB datasets cost near-zero disk space.
    """
    workspace_dir.mkdir(parents=True, exist_ok=True)
    prepared = download_competition(competition_id, data_dir)
    target = workspace_dir / "data"
    if not target.exists():
        try:
            subprocess.run(["cp", "-Rc", str(prepared), str(target)], check=True)
        except subprocess.CalledProcessError:
            subprocess.run(["cp", "-R", str(prepared), str(target)], check=True)
    return target


# `mlebench grade-sample` logs the report via Python's logger (stderr) framed
# by a "Competition report:" line followed by a pretty-printed JSON block.
# We extract the JSON object that follows that marker.
_REPORT_RE = re.compile(r"Competition report:\s*\n(.*)", re.DOTALL)


def _parse_report(stderr_text: str) -> dict[str, Any] | None:
    m = _REPORT_RE.search(stderr_text)
    if not m:
        return None
    # Strip the "[timestamp] [cli.py:NNN] " log prefix from each line so the
    # remaining text is a valid JSON object.
    cleaned = "\n".join(
        re.sub(r"^\[[^\]]+\]\s+\[[^\]]+\]\s+", "", line)
        for line in m.group(1).splitlines()
    )
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def grade(competition_id: str, submission_path: Path, data_dir: Path) -> dict[str, Any]:
    """Grade a submission and return the parsed JSON report.

    Falls back to a stub result if the submission file does not exist.
    """
    if not submission_path.exists():
        return {"medal": None, "score": None, "error": "no submission.csv produced"}
    cli = _mlebench()

    proc = subprocess.run(
        [
            cli,
            "grade-sample",
            str(submission_path),
            competition_id,
            "--data-dir",
            str(data_dir),
        ],
        capture_output=True,
        text=True,
        cwd=_TESTBED_ROOT,
    )
    report = _parse_report(proc.stderr) or _parse_report(proc.stdout)
    if report is not None:
        # Flatten the gold/silver/bronze booleans into a single medal label so
        # the runner can write it to results.csv unchanged.
        if report.get("gold_medal"):
            report["medal"] = "gold"
        elif report.get("silver_medal"):
            report["medal"] = "silver"
        elif report.get("bronze_medal"):
            report["medal"] = "bronze"
        else:
            report["medal"] = None
        return report
    return {
        "medal": None,
        "score": None,
        "error": f"grade-sample exited {proc.returncode}: {proc.stderr[-500:]}",
    }
