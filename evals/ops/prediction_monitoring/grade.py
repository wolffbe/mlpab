"""Prediction-monitoring task — grader: check the reported onset against the
seeded shift (±tolerance).

Usage:
    python -m evals.ops.prediction_monitoring.grade --instance <dir> --answers submission/answers.json
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from evals.common import Suite, grade_answers_main


def grade(instance_dir: Path, answers: dict | None) -> dict:
    truth = json.loads((Path(instance_dir) / "solution" / "truth.json").read_text())
    s = Suite()

    a1 = s.check(
        "A1_answers_present",
        isinstance(answers, dict) and "onset" in answers,
        "answers.json must contain 'onset'",
    )
    if a1:
        try:
            got = date.fromisoformat(str(answers["onset"]).strip()[:10])
            want = date.fromisoformat(truth["onset"])
            delta = abs((got - want).days)
            s.check(
                "A2_onset",
                delta <= truth["tolerance_days"],
                f"onset off by {delta} days (tolerance {truth['tolerance_days']})",
            )
        except ValueError as e:
            s.check("A2_onset", False, f"unparseable onset date: {e}")
    else:
        s.skip("A2_onset", "skipped: no answers.json with 'onset'")

    return s.report(family="prediction_monitoring", seed=truth["seed"])


def main(argv: list[str] | None = None) -> int:
    return grade_answers_main("prediction_monitoring", grade, argv)


if __name__ == "__main__":
    sys.exit(main())
