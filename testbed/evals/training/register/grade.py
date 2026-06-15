"""Register task — grader (platform kind).

A1 submission/answers.json names the model and version; A2 the answers'
metrics equal data/metrics.json exactly; A3 the registry entry exists on the
platform (state_checker.get_model — where the platform returned metrics, the
overlapping keys are compared numerically ±1e-6 as informative detail, never
as a failure, since not every registry can return metrics). Adapter `none`
(local baseline): A3 is skipped as passed.

Usage:
    python -m evals.training.register.grade --instance <dir> --adapter <name|none>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from evals.common import Suite, grade_platform_main, load_answers, state_checker

FAMILY = "register"
METRIC_TOL = 1e-6


def grade_answers(instance_dir: Path, adapter: str, answers: dict | None) -> dict:
    """The full assert suite over an already-loaded answers dict (the selftest
    gates call this directly — no platform, no files needed)."""
    truth = json.loads((Path(instance_dir) / "solution" / "truth.json").read_text())
    s = Suite()
    check = s.check

    # --- A1: answers present with matching name/version ------------------------
    if answers is None:
        check("A1_answers", False, "no submission/answers.json")
    elif "_parse_error" in answers:
        check("A1_answers", False, f"answers.json is not valid JSON: {answers['_parse_error']}")
    else:
        name_ok = str(answers.get("model_name", "")).strip() == truth["model_name"]
        ver_ok = answers.get("version") == truth["version"]
        check(
            "A1_answers",
            name_ok and ver_ok,
            ""
            if name_ok and ver_ok
            else f"expected model_name={truth['model_name']!r} version={truth['version']}, "
            f"got model_name={answers.get('model_name')!r} version={answers.get('version')!r}",
        )

    # --- A2: answers metrics == the provided metrics exactly --------------------
    a2 = False
    if answers and "_parse_error" not in answers:
        got = answers.get("metrics")
        want = truth["metrics"]
        if not isinstance(got, dict):
            check("A2_metrics", False, "answers.json must contain a 'metrics' object")
        else:
            same_keys = set(got) == set(want)
            a2 = same_keys and all(
                isinstance(got[k], (int, float)) and float(got[k]) == float(want[k]) for k in want
            )
            check("A2_metrics", a2, "" if a2 else "metrics differ from data/metrics.json")
    else:
        check("A2_metrics", False, "no answers to compare metrics from")

    # --- A3: registry entry exists on the platform -------------------------------
    if adapter == "none":
        s.skip("A3_registry_entry", "no platform — registry assert skipped")
    else:
        m = state_checker(adapter).get_model(truth["model_name"], truth["version"])
        ok = bool(m.get("exists"))
        detail = ""
        if ok:
            platform_metrics = m.get("metrics")
            if isinstance(platform_metrics, dict) and platform_metrics:
                overlap = set(platform_metrics) & set(truth["metrics"])
                if overlap:
                    diffs = [
                        k
                        for k in overlap
                        if abs(float(platform_metrics[k]) - float(truth["metrics"][k])) > METRIC_TOL
                    ]
                    detail = (
                        "platform metrics match on " + ", ".join(sorted(overlap))
                        if not diffs
                        else "platform metrics DIFFER on "
                        + ", ".join(sorted(diffs))
                        + " (informative — not a failure)"
                    )
                else:
                    detail = "platform returned metrics but none overlap the provided keys"
            else:
                detail = "platform could not return metrics (informative — exists only)"
        else:
            detail = f"model {truth['model_name']!r} v{truth['version']} not found: {m}"
        # detail_on_pass: the platform-metrics note is informative even when the
        # registry entry exists (e.g. metrics differ but that is not a failure).
        check("A3_registry_entry", ok, detail, detail_on_pass=True)

    return s.report(family=FAMILY, seed=truth["seed"])


def grade(instance_dir: Path, adapter: str, run_dir: Path) -> dict:
    answers = load_answers(Path(run_dir) / "submission" / "answers.json")
    return grade_answers(instance_dir, adapter, answers)


def main(argv=None) -> int:
    return grade_platform_main(FAMILY, grade, argv)


if __name__ == "__main__":
    sys.exit(main())
