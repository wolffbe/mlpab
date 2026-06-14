"""Leakage task — grader: check the reported leaky feature against the seed.

Usage:
    python -m evals.feature.leakage.grade --instance <dir> --answers submission/answers.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from evals.common import grade_answers_main


def grade(instance_dir: Path, answers: dict) -> dict:
    truth = json.loads((instance_dir / "solution" / "truth.json").read_text())
    asserts: list[dict] = []

    def check(name, ok, detail=""):
        asserts.append(
            {"name": name, "passed": bool(ok), **({"detail": detail} if detail and not ok else {})}
        )
        return bool(ok)

    a1 = check(
        "A1_answers_present",
        isinstance(answers, dict) and "feature" in answers,
        "answers.json must contain 'feature'",
    )
    a2 = False
    if a1:
        a2 = check(
            "A2_feature",
            str(answers["feature"]).strip() == truth["feature"],
            f"got {answers['feature']!r}, leaky feature differs",
        )
    return {
        "family": "leakage",
        "seed": truth["seed"],
        "success": a1 and a2,
        "asserts_passed": sum(a["passed"] for a in asserts),
        "asserts_total": len(asserts),
        "asserts": asserts,
    }


def main(argv=None) -> int:
    return grade_answers_main("leakage", grade, argv)


if __name__ == "__main__":
    sys.exit(main())
