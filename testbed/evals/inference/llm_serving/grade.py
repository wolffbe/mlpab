"""LLM-serving task — grader (platform kind).

Asserts:
    A1_answers_present — submission/answers.json names the endpoint
        (truth.json `endpoint_name`) and carries a 5-element `responses` list;
    A2_responses       — each response matches the generator's score (±1e-6),
        in payload order;
    A3_endpoint_exists — the endpoint exists on the platform (state read;
        skipped-pass on adapter `none`).

Usage:
    python -m evals.inference.llm_serving.grade --instance <dir> --adapter <hopsworks|databricks|sagemaker|none>
(cwd must be the run dir — the provider runs graders there.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from evals.common import grade_platform_main, load_answers, state_checker

TOL = 1e-6


def grade(instance_dir: Path, adapter: str, run_dir: Path) -> dict:
    truth = json.loads((Path(instance_dir) / "solution" / "truth.json").read_text())
    want = truth["responses"]
    asserts: list[dict] = []

    def check(name, ok, detail=""):
        asserts.append({"name": name, "passed": bool(ok),
                        **({"detail": detail} if detail else {})})
        return bool(ok)

    answers = load_answers(Path(run_dir) / "submission" / "answers.json")
    responses = (answers or {}).get("responses") if isinstance(answers, dict) else None
    a1_ok = (isinstance(answers, dict)
             and str(answers.get("endpoint_name", "")).strip() == truth["endpoint_name"]
             and isinstance(responses, list) and len(responses) == len(want))
    a1 = check("A1_answers_present", a1_ok,
               "" if a1_ok else
               f"answers.json must name endpoint {truth['endpoint_name']!r} and "
               f"carry a {len(want)}-element 'responses' list")

    a2 = False
    if a1:
        bad = []
        for i, (g, w) in enumerate(zip(responses, want)):
            try:
                ok = abs(float(g) - w) <= TOL
            except (TypeError, ValueError):
                ok = False
            if not ok:
                bad.append(i)
        a2 = check("A2_responses", not bad,
                   f"responses differ from the scorer at indices {bad}" if bad else "")
    else:
        a2 = check("A2_responses", False, "no valid answers to compare")

    if adapter == "none":
        a3 = check("A3_endpoint_exists", True,
                   "no checker adapter (platform none) — skipped")
    else:
        st = state_checker(adapter).get_endpoint(truth["endpoint_name"])
        a3 = check("A3_endpoint_exists", st.get("exists"),
                   "" if st.get("exists") else
                   f"endpoint {truth['endpoint_name']!r} not found: {st}")

    success = a1 and a2 and a3
    return {"family": "llm_serving", "seed": truth["seed"], "success": success,
            "asserts_passed": sum(a["passed"] for a in asserts),
            "asserts_total": len(asserts), "asserts": asserts}


def main(argv=None) -> int:
    return grade_platform_main("llm_serving", grade, argv)


if __name__ == "__main__":
    sys.exit(main())
