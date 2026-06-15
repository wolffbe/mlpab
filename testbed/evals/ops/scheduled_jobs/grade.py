"""Scheduled-jobs task — grader (platform kind).

Asserts:
    A1_answers_present — submission/answers.json names the job
        (truth.json `job_name`);
    A2_job_exists      — the job exists on the platform (state read;
        skipped-pass on adapter `none`);
    A3_scheduled       — the platform reports the job as scheduled/recurring
        (skipped-pass on adapter `none`).

Usage:
    python -m evals.ops.scheduled_jobs.grade --instance <dir> --adapter <hopsworks|databricks|sagemaker|none>
(cwd must be the run dir — the provider runs graders there.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from evals.common import Suite, grade_platform_main, load_answers, state_checker


def grade(instance_dir: Path, adapter: str, run_dir: Path) -> dict:
    truth = json.loads((Path(instance_dir) / "solution" / "truth.json").read_text())
    job = truth["job_name"]
    s = Suite()
    check = s.check

    answers = load_answers(Path(run_dir) / "submission" / "answers.json")
    a1_ok = isinstance(answers, dict) and str(answers.get("job_name", "")).strip() == job
    check(
        "A1_answers_present",
        a1_ok,
        "" if a1_ok else f"answers.json must contain job_name = {job!r}",
    )

    if adapter == "none":
        s.skip("A2_job_exists", "no checker adapter (platform none) — skipped")
        s.skip("A3_scheduled", "no checker adapter (platform none) — skipped")
    else:
        st = state_checker(adapter).get_job(job)
        check(
            "A2_job_exists",
            st.get("exists"),
            "" if st.get("exists") else f"job {job!r} not found: {st}",
        )
        check(
            "A3_scheduled",
            st.get("exists") and st.get("scheduled"),
            "" if st.get("scheduled") else f"job {job!r} is not scheduled/recurring: {st}",
        )

    return s.report(family="scheduled_jobs", seed=truth["seed"])


def main(argv=None) -> int:
    return grade_platform_main("scheduled_jobs", grade, argv)


if __name__ == "__main__":
    sys.exit(main())
