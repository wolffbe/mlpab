"""Train task — grader (platform kind).

A1 the predictions table exists on the platform; A2/A3 its content (row count
+ digest; float tolerance comes from the script's round-6 plus canonicalize);
A4 evidence the job exists on the platform (state_checker.get_job — exists
only, with the platform's job detail kept informative). Adapter `none` (local
baseline): content is graded from submission/predictions.csv and A4 is
skipped as passed.

Usage:
    python -m evals.training.train.grade --instance <dir> --adapter <name|none>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


from evals.common import (
    Suite,
    canonicalize,
    digest,
    fetch_table_deliverable,
    grade_platform_main,
    read_csv_or_empty,
    state_checker,
    table_exists_info,
    tally,
)

FAMILY = "train"


def grade(instance_dir: Path, adapter: str, run_dir: Path) -> dict:
    truth = json.loads((Path(instance_dir) / "solution" / "truth.json").read_text())
    spec = truth["spec"]
    s = Suite()
    check = s.check
    diagnostic = None

    # --- A1: table exists / local deliverable exists --------------------------
    produced = None
    read_err = None
    if adapter == "none":
        local = Path(run_dir) / "submission" / f"{truth['table_name']}.csv"
        if check(
            "A1_table_exists",
            local.exists(),
            "" if local.exists() else f"no local deliverable at {local}",
        ):
            produced = read_csv_or_empty(local)
    else:
        info = table_exists_info(adapter, truth["table_name"], truth["table_version"])
        if check(
            "A1_table_exists",
            info is not None,
            ""
            if info is not None
            else f"feature table {truth['table_name']!r} v{truth['table_version']} not found",
        ):
            try:
                produced = fetch_table_deliverable(
                    adapter, truth["table_name"], truth["table_version"], truth.get("record_ids")
                )
            except Exception as e:  # noqa: BLE001
                # Deterministic adapter limit OR an unreadable deliverable — degrade
                # gracefully (grade an empty frame) instead of crashing the grader.
                read_err = f"could not read table back: {e}"

    # --- A2/A3: content --------------------------------------------------------
    if produced is not None:
        try:
            norm = canonicalize(produced, spec)
        except Exception as e:
            check("A2_row_count", False, f"could not normalize: {e}")
            s.skip("A3_content", "skipped: could not normalize the deliverable")
            norm = None
        if norm is not None:
            check(
                "A2_row_count",
                len(norm) == truth["row_count"],
                ""
                if len(norm) == truth["row_count"]
                else f"got {len(norm)}, expected {truth['row_count']}",
            )
            d = digest(norm)
            if not check(
                "A3_content",
                d == truth["digest"],
                "" if d == truth["digest"] else "content digest mismatch",
            ):
                for vname, vdig in truth.get("variant_digests", {}).items():
                    if d == vdig:
                        diagnostic = truth.get("variant_diagnosis", {}).get(vname, vname)
                        break
    elif read_err is not None:
        check("A2_row_count", False, read_err)
        s.skip("A3_content", "skipped: could not read the table back")
    else:
        s.skip("A2_row_count", "skipped: no gradeable deliverable")
        s.skip("A3_content", "skipped: no gradeable deliverable")

    # --- A4: job evidence on the platform ----------------------------------------
    if adapter == "none":
        s.skip("A4_job_exists", "no platform — job assert skipped")
    else:
        job = state_checker(adapter).get_job(truth["job_name"])
        ok = bool(job.get("exists"))
        detail = ", ".join(
            f"{k}={job[k]}" for k in ("kind", "scheduled", "last_run_state") if k in job
        )
        check(
            "A4_job_exists",
            ok,
            detail if ok else f"job {truth['job_name']!r} not found on the platform: {job}",
        )

    return {
        "family": FAMILY,
        "seed": truth["seed"],
        **tally(s.asserts),
        "asserts": s.asserts,
        **({"diagnostic": diagnostic} if diagnostic else {}),
    }


def main(argv=None) -> int:
    return grade_platform_main(FAMILY, grade, argv)


if __name__ == "__main__":
    sys.exit(main())
