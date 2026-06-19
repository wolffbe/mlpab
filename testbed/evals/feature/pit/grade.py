"""F5 (PIT) — grader: assertion suite over a produced training dataset.

The dataset normally arrives THROUGH a platform checker adapter (read the
versioned training dataset back via Hopsworks/Databricks/SageMaker read
paths); until adapters land, `--csv` grades a local file the same way.

Success is binary: A1–A4 must all pass. On a content mismatch (A4) the grader
compares against the generator's known-wrong variants to NAME the failure:
    latest_join  → joined overall-latest rows (used future data)
    ignore_late  → dropped the late-arriving export
    leak_future  → PIT-correct except the leak table (took the future value)

Usage:
    python -m evals.feature.pit.grade --instance /tmp/pit-m7 --csv produced.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from evals.common import Suite, read_csv_or_empty, read_deliverable, tally
from evals.feature.pit.generate import canonicalize, digest

VARIANT_DIAGNOSIS = {
    "latest_join": "joined each account's overall-latest feature rows — values from "
    "AFTER label_time leaked into the training data",
    "ignore_late": "the late-arriving export (transactions_late.csv) was not ingested",
    "leak_future": "PIT-correct except the leak-tempting table, whose post-label row "
    "was used — future information leaked into the training data",
}


def grade(instance_dir: Path, produced: pd.DataFrame) -> dict:
    truth_meta = json.loads((instance_dir / "solution" / "truth.json").read_text())
    columns = truth_meta["columns"]
    s = Suite()
    diagnostic = None

    # A1 — column set (canonicalize would throw on missing columns; check first)
    produced.columns = [str(c).strip().lower() for c in produced.columns]
    missing = [c for c in columns if c not in produced.columns]
    a1 = s.check("A1_columns", not missing, f"missing columns: {missing}")

    if not a1:
        # No gradeable deliverable — the downstream checks are unreachable.
        s.skip("A2_row_count", "skipped: required columns are missing")
        s.skip("A3_labels", "skipped: required columns are missing")
        s.skip("A4_content", "skipped: required columns are missing")
    else:
        norm = canonicalize(produced, columns)
        # A2 — one row per label row
        a2 = s.check(
            "A2_row_count",
            len(norm) == truth_meta["row_count"],
            f"got {len(norm)}, expected {truth_meta['row_count']}",
        )
        # A3 — label passthrough (account_id + churned must match exactly)
        truth = pd.read_csv(instance_dir / "solution" / "truth.csv")
        truth = canonicalize(truth, columns)
        if a2:
            s.check(
                "A3_labels",
                norm[["account_id", "churned"]].equals(truth[["account_id", "churned"]]),
                "account_id/churned do not match labels.csv",
            )
        else:
            s.skip("A3_labels", "skipped: row count differs, label alignment undefined")
        # A4 — content
        d = digest(norm)
        a4 = s.check("A4_content", d == truth_meta["digest"], "content digest mismatch")
        if not a4:
            for vname, vdigest in truth_meta.get("variant_digests", {}).items():
                if d == vdigest:
                    diagnostic = VARIANT_DIAGNOSIS.get(vname, vname)
                    break
            if diagnostic is None and a2:
                # first few differing cells, for a readable report
                neq = norm != truth
                bad = neq.any(axis=1)
                examples = []
                for idx in norm.index[bad][:3]:
                    cols_bad = [c for c in columns if neq.at[idx, c]]
                    examples.append(
                        {
                            "account_id": norm.at[idx, "account_id"],
                            "columns": {
                                c: {"got": norm.at[idx, c], "expected": truth.at[idx, c]}
                                for c in cols_bad
                            },
                        }
                    )
                diagnostic = f"content differs on {int(bad.sum())} rows; examples: {examples}"

    return {
        "family": "pit",
        "seed": truth_meta["seed"],
        **tally(s.asserts),
        "asserts": s.asserts,
        **({"diagnostic": diagnostic} if diagnostic else {}),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instance", type=Path, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="grade a local file (no platform)")
    src.add_argument(
        "--adapter",
        choices=["hopsworks", "databricks", "aws", "azure", "gcp"],
        help="read the deliverable back THROUGH the platform "
        "(name/version from the instance's truth.json)",
    )
    args = ap.parse_args(argv)

    deliverable_err = None
    if args.csv:
        if not args.csv.exists():
            produced = pd.DataFrame()
            deliverable_err = f"no deliverable produced at {args.csv}"
        else:
            produced = read_csv_or_empty(args.csv)
    else:
        meta = json.loads((args.instance / "solution" / "truth.json").read_text())
        # state_checker dispatches to the right adapter for all five platforms
        # (hopsworks/databricks/aws/azure/gcp) — uniform, no per-platform branch.
        from evals.common import state_checker

        # Read back the training dataset through the shared single-attempt read
        # (the HQS service is reliable, so a failure means the deliverable could
        # not be read back). The read NEVER crashes the grader: a deterministic
        # miss (LookupError) or any other read failure degrades to an empty frame
        # ("no results") with the reason recorded.
        try:
            produced = read_deliverable(
                lambda: state_checker(args.adapter).read_training_dataset(
                    meta["dataset_name"], meta["dataset_version"]
                )
            )
        except LookupError as e:
            produced = pd.DataFrame()
            deliverable_err = str(e)
        except Exception as e:  # noqa: BLE001 — deliverable could not be read back
            produced = pd.DataFrame()
            deliverable_err = f"deliverable could not be read back: {e}"

    report = grade(args.instance, produced)
    if deliverable_err:
        report.setdefault("error", deliverable_err)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
