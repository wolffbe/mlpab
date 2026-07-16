"""Drift task — grader: check the reported (feature, onset) against the seed.

Usage:
    python -m evals.ops.drift.grade --instance <dir> --answers submission/answers.json
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from evals.common import Suite, grade_answers_main


def grade(instance_dir: Path, answers: dict | None) -> dict:
    truth = json.loads((instance_dir / "solution" / "truth.json").read_text())
    s = Suite()

    a1 = s.check(
        "A1_answers_present",
        isinstance(answers, dict) and "feature" in answers and "onset" in answers,
        "answers.json must contain 'feature' and 'onset'",
    )
    if a1:
        s.check(
            "A2_feature",
            str(answers["feature"]).strip() == truth["feature"],
            f"got {answers['feature']!r}, drifted feature differs",
        )
        try:
            got = date.fromisoformat(str(answers["onset"]).strip()[:10])
            want = date.fromisoformat(truth["onset"])
            delta = abs((got - want).days)
            s.check(
                "A3_onset",
                delta <= truth["tolerance_days"],
                f"onset off by {delta} days (tolerance {truth['tolerance_days']})",
            )
        except ValueError as e:
            s.check("A3_onset", False, f"unparseable onset date: {e}")
    else:
        s.skip("A2_feature", "skipped: no answers.json with 'feature' and 'onset'")
        s.skip("A3_onset", "skipped: no answers.json with 'feature' and 'onset'")

    return s.report(family="drift", seed=truth["seed"])


def main(argv: list[str] | None = None) -> int:
    return grade_answers_main("drift", grade, argv)


if __name__ == "__main__":
    sys.exit(main())
