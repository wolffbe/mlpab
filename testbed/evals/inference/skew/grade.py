"""Skew task — grader: check the reported diverging feature against the seed.

Usage:
    python -m evals.inference.skew.grade --instance <dir> --answers submission/answers.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from evals.common import Suite, grade_answers_main


def grade(instance_dir: Path, answers: dict | None) -> dict:
    truth = json.loads((instance_dir / "solution" / "truth.json").read_text())
    s = Suite()

    a1 = s.check(
        "A1_answers_present",
        isinstance(answers, dict) and "feature" in answers,
        "answers.json must contain 'feature'",
    )
    if a1:
        s.check(
            "A2_feature",
            str(answers["feature"]).strip() == truth["feature"],
            f"got {answers['feature']!r}, skewed feature differs",
        )
    else:
        s.skip("A2_feature", "skipped: no answers.json with 'feature'")

    return s.report(family="skew", seed=truth["seed"])


def main(argv: list[str] | None = None) -> int:
    return grade_answers_main("skew", grade, argv)


if __name__ == "__main__":
    sys.exit(main())
