"""Alerting task (FTI sub-category: ops/alerting) — generator.

Usage:
    python -m evals.ops.alerting.generate --seed 7 --out /tmp/alert-7
    python -m evals.ops.alerting.generate --selftest

World: a provided script that ALWAYS fails (`data/failing_job.py` — raises a
clear seeded error and exits 1) plus instructions. The agent must create a
job named `flaky<sfx>` (per-instance suffix) on the platform that runs the
script, configure an ALERT/notification for its failures (the platform's
mechanism: job-failure notifications, alert rules, or a metric alarm — name
or hint `flaky<sfx>` in the alert), let it run once (it WILL fail — that is
expected), and write `submission/answers.json`:

    {"job_name": "flaky<sfx>", "alert": "<name or short description of the alert>"}

Grading reads the job and the alert back through the platform's state reads;
on adapter `none` those asserts are skipped-pass, so the generation-time
gates exercise the answers-shape asserts directly: the reference answers pass
the grade function; a missing/empty `alert` key and a wrong job name fail.
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

JOB_BASE = "flaky"  # per-instance: f"{JOB_BASE}{instance_suffix(seed)}"

FAILING_TEMPLATE = '''"""Flaky job — this task ALWAYS fails (that is the point)."""
import sys

ERROR_CODE = "{code}"

if __name__ == "__main__":
    print(f"{job} starting (error code {{ERROR_CODE}})", file=sys.stderr)
    raise RuntimeError(
        f"seeded failure {{ERROR_CODE}}: upstream source unavailable — "
        "this job is EXPECTED to fail"
    )
'''


class GateError(RuntimeError):
    pass


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    job = JOB_BASE + instance_suffix(seed)
    code = f"ERR-{int(rng.integers(10**5, 10**6)):06d}"

    # --- write instance --------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    (out / "data" / "failing_job.py").write_text(
        FAILING_TEMPLATE.format(code=code, job=job))
    (out / "data" / "instructions.md").write_text(
        "# Flaky job + failure alert\n\n"
        f"`failing_job.py` ALWAYS exits 1 (it raises a seeded error, {code}). "
        "That is intentional: the goal is to verify that a FAILURE of the job "
        "raises an alert. Configure the platform's alerting mechanism for the "
        f"job's failures — name or mention `{job}` in the alert so it is "
        "identifiable.\n"
    )
    (out / "prompt.txt").write_text(
        "The directory data/ contains a script that ALWAYS fails "
        "(data/failing_job.py — exits 1 with a clear error; that is expected) "
        "and data/instructions.md.\n"
        f"Create a job named `{job}` on the platform that runs the provided "
        "script, and configure an ALERT/notification for its FAILURES using "
        "the platform's mechanism (job-failure notifications, alert rules, or "
        f"a metric alarm) — name or mention `{job}` in the alert so it is "
        "identifiable.\n"
        "Let the job run once (it will fail — that is expected and is what the "
        "alert is for), then write submission/answers.json as:\n"
        f'    {{"job_name": "{job}", "alert": "<name or short description '
        'of the alert you configured>"}\n'
    )
    truth = {"family": "alerting", "seed": seed,
             "job_name": job, "error_code": code}
    (out / "solution" / "truth.json").write_text(json.dumps(truth, indent=2))
    (out / "instance.json").write_text(json.dumps(
        {"family": "alerting", "seed": seed}, indent=2))

    # --- gates: answers-shape asserts via the grade function (adapter none) ----
    from evals.ops.alerting.grade import grade

    def run(answers: dict | None) -> bool:
        with tempfile.TemporaryDirectory(prefix="banter-alert-gate-") as td:
            run_dir = Path(td)
            (run_dir / "submission").mkdir()
            if answers is not None:
                (run_dir / "submission" / "answers.json").write_text(json.dumps(answers))
            return grade(out, "none", run_dir)["success"]

    if not run({"job_name": job, "alert": f"{job}-failure-alert"}):
        raise GateError(f"reference answers fail the grade function (seed={seed})")
    if run({"job_name": job}):
        raise GateError(f"missing 'alert' key passes the grade function (seed={seed})")
    if run({"job_name": job, "alert": "  "}):
        raise GateError(f"empty 'alert' passes the grade function (seed={seed})")
    if run({"job_name": "other_job", "alert": "x"}):
        raise GateError(f"wrong job name passes the grade function (seed={seed})")
    return truth


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/banter-alerting-selftest/{seed}"))
            print(f"[alerting] seed={seed} error_code={meta['error_code']} gates=OK")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
