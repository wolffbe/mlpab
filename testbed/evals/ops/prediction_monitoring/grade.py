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

from evals.common import grade_answers_main


def grade(instance_dir: Path, answers: dict) -> dict:
    truth = json.loads((Path(instance_dir) / "solution" / "truth.json").read_text())
    asserts: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        asserts.append({"name": name, "passed": bool(ok),
                        **({"detail": detail} if detail and not ok else {})})
        return bool(ok)

    a1 = check("A1_answers_present",
               isinstance(answers, dict) and "onset" in answers,
               "answers.json must contain 'onset'")
    a2 = False
    if a1:
        try:
            got = date.fromisoformat(str(answers["onset"]).strip()[:10])
            want = date.fromisoformat(truth["onset"])
            delta = abs((got - want).days)
            a2 = check("A2_onset", delta <= truth["tolerance_days"],
                       f"onset off by {delta} days (tolerance {truth['tolerance_days']})")
        except ValueError as e:
            a2 = check("A2_onset", False, f"unparseable onset date: {e}")

    return {
        "family": "prediction_monitoring", "seed": truth["seed"],
        "success": a1 and a2,
        "asserts_passed": sum(a["passed"] for a in asserts),
        "asserts_total": len(asserts),
        "asserts": asserts,
    }


def main(argv: list[str] | None = None) -> int:
    return grade_answers_main("prediction_monitoring", grade, argv)


if __name__ == "__main__":
    sys.exit(main())
