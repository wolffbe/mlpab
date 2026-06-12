"""Alerting task — grader (platform kind).

Asserts:
    A1_answers_present — submission/answers.json names the job
        (truth.json `job_name`)
        and carries a non-empty `alert` description;
    A2_job_exists      — the job exists on the platform (state read;
        skipped-pass on adapter `none`);
    A3_alert_exists    — an alert naming/hinting the job name exists on the
        platform (state read; skipped-pass on adapter `none`).

Usage:
    python -m evals.ops.alerting.grade --instance <dir> --adapter <hopsworks|databricks|sagemaker|none>
(cwd must be the run dir — the provider runs graders there.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from evals.common import grade_platform_main, load_answers, state_checker


def grade(instance_dir: Path, adapter: str, run_dir: Path) -> dict:
    truth = json.loads((Path(instance_dir) / "solution" / "truth.json").read_text())
    job = truth["job_name"]
    asserts: list[dict] = []

    def check(name, ok, detail=""):
        asserts.append({"name": name, "passed": bool(ok),
                        **({"detail": detail} if detail else {})})
        return bool(ok)

    answers = load_answers(Path(run_dir) / "submission" / "answers.json")
    a1_ok = (isinstance(answers, dict)
             and str(answers.get("job_name", "")).strip() == job
             and bool(str(answers.get("alert", "") or "").strip()))
    a1 = check("A1_answers_present", a1_ok,
               "" if a1_ok else
               f"answers.json must contain job_name = {job!r} and a non-empty 'alert'")

    if adapter == "none":
        a2 = check("A2_job_exists", True,
                   "no checker adapter (platform none) — skipped")
        a3 = check("A3_alert_exists", True,
                   "no checker adapter (platform none) — skipped")
    else:
        checker = state_checker(adapter)
        st = checker.get_job(job)
        a2 = check("A2_job_exists", st.get("exists"),
                   "" if st.get("exists") else f"job {job!r} not found: {st}")
        al = checker.get_alert(job)
        a3 = check("A3_alert_exists", al.get("exists"),
                   "" if al.get("exists") else
                   f"no alert naming/hinting {job!r} found: {al}")

    success = a1 and a2 and a3
    return {"family": "alerting", "seed": truth["seed"], "success": success,
            "asserts_passed": sum(a["passed"] for a in asserts),
            "asserts_total": len(asserts), "asserts": asserts}


def main(argv=None) -> int:
    return grade_platform_main("alerting", grade, argv)


if __name__ == "__main__":
    sys.exit(main())
