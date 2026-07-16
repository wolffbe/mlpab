"""Finetune task — grader (platform kind).

A1 submission/answers.json parses and names the job and model; A2 its
eval_loss and base_eval_loss equal the truth metrics EXACTLY (4 dp — the
script rounds, and the generator keeps every pre-rounding loss away from the
4-dp boundary), with a wrong eval_loss matching a precomputed naive variant
diagnosed by name (no_finetune / undertrained); A3 evidence the job exists on
the platform (state_checker.get_job — exists only, with the platform's job
detail kept informative); A4 the registry entry exists
(state_checker.get_model — where the platform returned metrics, the
overlapping keys are compared numerically ±1e-6 as informative detail, never
as a failure, since not every registry can return metrics). Adapter `none`
(local baseline): A3 and A4 are skipped as passed.

Usage:
    python -m evals.training.llm_finetuning.grade --instance <dir> --adapter <name|none>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from evals.common import Suite, grade_platform_main, load_answers, state_checker, tally

FAMILY = "llm_finetuning"
METRIC_TOL = 1e-6
METRIC_KEYS = ("eval_loss", "base_eval_loss")


def grade_answers(instance_dir: Path, adapter: str, answers: dict | None) -> dict:
    """The full assert suite over an already-loaded answers dict (the selftest
    gates call this directly — no platform, no files needed)."""
    truth = json.loads((Path(instance_dir) / "solution" / "truth.json").read_text())
    s = Suite()
    check = s.check
    diagnostic = None

    # --- A1: answers present with matching job/model names ----------------------
    if answers is None:
        check("A1_format", False, "no submission/answers.json")
    elif "_parse_error" in answers:
        check("A1_format", False, f"answers.json is not valid JSON: {answers['_parse_error']}")
    else:
        missing = [k for k in ("job_name", "model_name", *METRIC_KEYS) if k not in answers]
        job_ok = str(answers.get("job_name", "")).strip() == truth["job_name"]
        model_ok = str(answers.get("model_name", "")).strip() == truth["model_name"]
        problems = ([f"missing keys: {missing}"] if missing else []) + (
            [
                f"expected job_name={truth['job_name']!r} "
                f"model_name={truth['model_name']!r}, "
                f"got job_name={answers.get('job_name')!r} "
                f"model_name={answers.get('model_name')!r}"
            ]
            if not (job_ok and model_ok)
            else []
        )
        check("A1_format", not problems, "; ".join(problems))

    # --- A2: answers metrics == the truth metrics exactly ------------------------
    if answers and "_parse_error" not in answers:
        want = truth["metrics"]
        a2 = all(
            isinstance(answers.get(k), (int, float)) and float(answers[k]) == float(want[k])
            for k in METRIC_KEYS
        )
        check(
            "A2_metrics",
            a2,
            "" if a2 else "eval_loss/base_eval_loss differ from the script's metrics.json",
        )
        if not a2 and isinstance(answers.get("eval_loss"), (int, float)):
            got = float(answers["eval_loss"])
            for vname, vm in truth.get("variant_metrics", {}).items():
                if got == float(vm["eval_loss"]):
                    diagnostic = truth.get("variant_diagnosis", {}).get(vname, vname)
                    break
    else:
        check("A2_metrics", False, "no answers to compare metrics from")

    # --- A3: job evidence on the platform ----------------------------------------
    if adapter == "none":
        s.skip("A3_job_exists", "no platform — job assert skipped")
    else:
        job = state_checker(adapter).get_job(truth["job_name"])
        ok = bool(job.get("exists"))
        detail = ", ".join(
            f"{k}={job[k]}" for k in ("kind", "scheduled", "last_run_state") if k in job
        )
        check(
            "A3_job_exists",
            ok,
            detail if ok else f"job {truth['job_name']!r} not found on the platform: {job}",
        )

    # --- A4: registry entry exists on the platform --------------------------------
    if adapter == "none":
        s.skip("A4_registry_entry", "no platform — registry assert skipped")
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
        check("A4_registry_entry", ok, detail, detail_on_pass=True)

    return {
        "family": FAMILY,
        "seed": truth["seed"],
        **tally(s.asserts),
        "asserts": s.asserts,
        **({"diagnostic": diagnostic} if diagnostic else {}),
    }


def grade(instance_dir: Path, adapter: str, run_dir: Path) -> dict:
    answers = load_answers(Path(run_dir) / "submission" / "answers.json")
    return grade_answers(instance_dir, adapter, answers)


def main(argv=None) -> int:
    return grade_platform_main(FAMILY, grade, argv)


if __name__ == "__main__":
    sys.exit(main())
