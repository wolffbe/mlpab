"""Validate task — grader: the standard table suite over the clean-rows table
PLUS A4_rejected, the rejected-id set from submission/answers.json (read from
the run dir — the provider runs graders with cwd = the run dir).

Usage:
    python -m evals.feature.validate.grade --instance <dir> (--csv <file> | --adapter <name>)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from evals.common import grade_table_content, grade_table_main, load_answers, tally


def grade(instance_dir, produced, adapter=None) -> dict:
    truth = json.loads((Path(instance_dir) / "solution" / "truth.json").read_text())
    report = grade_table_content("validate", instance_dir, produced)

    answers = load_answers(Path.cwd() / "submission" / "answers.json")
    if answers is None:
        a4_ok, a4_detail = False, "no submission/answers.json with the rejected row ids"
    elif "_parse_error" in answers:
        a4_ok, a4_detail = False, f"answers.json is not valid JSON: {answers['_parse_error']}"
    elif not isinstance(answers.get("rejected"), list):
        a4_ok, a4_detail = False, "answers.json must contain a 'rejected' list of row ids"
    else:
        got = {str(r).strip() for r in answers["rejected"]}
        want = set(truth["rejected_ids"])
        a4_ok = got == want
        a4_detail = (
            ""
            if a4_ok
            else f"{len(want - got)} violating ids missing, {len(got - want)} clean ids wrongly rejected"
        )
    a4 = {
        "name": "A4_rejected",
        "status": "pass" if a4_ok else "fail",
        "passed": a4_ok,
        **({"detail": a4_detail} if a4_detail and not a4_ok else {}),
    }

    report["asserts"].append(a4)
    # Re-tally over the full A1–A4 suite (grade_table_content gave A1–A3).
    report.update(tally(report["asserts"]))
    return report


def main(argv=None) -> int:
    return grade_table_main("validate", grade, argv)


if __name__ == "__main__":
    sys.exit(main())
