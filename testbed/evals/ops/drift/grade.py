"""Drift task — grader: check the reported (feature, onset) against the seed.

Usage:
    python -m evals.ops.drift.grade --instance <dir> --answers submission/answers.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path


def grade(instance_dir: Path, answers: dict) -> dict:
    truth = json.loads((instance_dir / "solution" / "truth.json").read_text())
    asserts: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        asserts.append({"name": name, "passed": bool(ok),
                        **({"detail": detail} if detail and not ok else {})})
        return bool(ok)

    a1 = check("A1_answers_present",
               isinstance(answers, dict) and "feature" in answers and "onset" in answers,
               "answers.json must contain 'feature' and 'onset'")
    a2 = a3 = False
    if a1:
        a2 = check("A2_feature", str(answers["feature"]).strip() == truth["feature"],
                   f"got {answers['feature']!r}, drifted feature differs")
        try:
            got = date.fromisoformat(str(answers["onset"]).strip()[:10])
            want = date.fromisoformat(truth["onset"])
            delta = abs((got - want).days)
            a3 = check("A3_onset", delta <= truth["tolerance_days"],
                       f"onset off by {delta} days (tolerance {truth['tolerance_days']})")
        except ValueError as e:
            a3 = check("A3_onset", False, f"unparseable onset date: {e}")

    return {
        "family": "drift", "seed": truth["seed"],
        "success": a1 and a2 and a3,
        "asserts_passed": sum(a["passed"] for a in asserts),
        "asserts_total": len(asserts),
        "asserts": asserts,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instance", type=Path, required=True)
    ap.add_argument("--answers", type=Path, required=True)
    args = ap.parse_args(argv)

    if not args.answers.exists():
        report = {"family": "drift", "success": False, "asserts_passed": 0,
                  "asserts_total": 1,
                  "asserts": [{"name": "A0_deliverable_exists", "passed": False,
                               "detail": f"no answers file at {args.answers}"}]}
    else:
        try:
            answers = json.loads(args.answers.read_text())
        except json.JSONDecodeError as e:
            answers = {"_parse_error": str(e)}
        report = grade(args.instance, answers)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
