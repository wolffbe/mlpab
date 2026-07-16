"""Scheduled-jobs task (FTI sub-category: ops/scheduled_jobs) — generator.

Usage:
    python -m evals.ops.scheduled_jobs.generate --seed 7 --out /tmp/sched-7
    python -m evals.ops.scheduled_jobs.generate --selftest

World: a trivial provided script (`data/heartbeat.py` — prints one seeded
heartbeat line and exits 0) plus instructions. The agent must create a
RECURRING (scheduled) job named `heartbeat<sfx>` (per-instance suffix) on the
platform that runs the script periodically (any reasonable schedule),
let/trigger one run to completion, and write `submission/answers.json`:

    {"job_name": "heartbeat<sfx>"}

Grading reads the job back through the platform's state read (existence +
scheduled); on adapter `none` the platform asserts are skipped-pass, so the
generation-time gates exercise the answers asserts directly: the reference
answers pass the grade function, a wrong/missing job name fails.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from evals.common import instance_suffix

KIND = "platform"  # deliverable kind: table | dataset | answers | platform
SUMMARY = (
    "Measures whether the agent can set up a recurring scheduled job on the platform from a "
    "provided script and prove it ran."
)

JOB_BASE = "heartbeat"  # per-instance: f"{JOB_BASE}{instance_suffix(seed)}"

HEARTBEAT_TEMPLATE = '''"""Heartbeat — a trivial periodic task. Prints one line and exits 0."""
import datetime

TOKEN = "{token}"

if __name__ == "__main__":
    print(f"heartbeat {{TOKEN}} alive at "
          f"{{datetime.datetime.now(datetime.timezone.utc).isoformat()}}")
'''


class GateError(RuntimeError):
    pass


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    job = JOB_BASE + instance_suffix(seed)
    token = f"HB-{int(rng.integers(10**7, 10**8)):08d}"

    # --- write instance --------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    (out / "data" / "heartbeat.py").write_text(HEARTBEAT_TEMPLATE.format(token=token))
    (out / "data" / "instructions.md").write_text(
        "# Heartbeat job\n\n"
        "`heartbeat.py` is a trivial task: it prints one heartbeat line "
        f"(token `{token}`) and exits 0. It must run PERIODICALLY on the "
        "platform — set up a recurring schedule (any reasonable interval, e.g. "
        "hourly or daily) rather than a one-off run.\n"
    )
    (out / "prompt.txt").write_text(
        "The directory data/ contains a trivial script (data/heartbeat.py — "
        "prints one heartbeat line and exits 0) and data/instructions.md.\n"
        f"Create a RECURRING (scheduled) job named `{job}` on the platform that "
        "runs the provided script periodically — any reasonable schedule (e.g. "
        "hourly or daily) is fine, but it must be a recurring job, not a "
        "one-off run.\n"
        "Trigger (or let) one run complete, then write "
        "submission/answers.json as:\n"
        f'    {{"job_name": "{job}"}}\n'
    )
    truth = {"family": "scheduled_jobs", "seed": seed, "job_name": job, "token": token}
    (out / "solution" / "truth.json").write_text(json.dumps(truth, indent=2))
    (out / "instance.json").write_text(
        json.dumps({"family": "scheduled_jobs", "seed": seed}, indent=2)
    )

    # --- gates: answers asserts via the grade function (adapter none) ----------
    from evals.ops.scheduled_jobs.grade import grade

    def run(answers: dict | None) -> bool:
        with tempfile.TemporaryDirectory(prefix="mlpab-sched-gate-") as td:
            run_dir = Path(td)
            (run_dir / "submission").mkdir()
            if answers is not None:
                (run_dir / "submission" / "answers.json").write_text(json.dumps(answers))
            return grade(out, "none", run_dir)["success"]

    if not run({"job_name": job}):
        raise GateError(f"reference answers fail the grade function (seed={seed})")
    if run({"job_name": "some_other_job"}):
        raise GateError(f"wrong job name passes the grade function (seed={seed})")
    if run(None):
        raise GateError(f"missing answers.json passes the grade function (seed={seed})")
    return truth


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/mlpab-sched-selftest/{seed}"))
            print(f"[scheduled_jobs] seed={seed} token={meta['token']} gates=OK")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
