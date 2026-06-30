"""Online-serving task — grader (platform kind).

Asserts:
    A1_table_exists — the `profiles` feature table exists (metadata read;
        skipped-pass on adapter `none`);
    A2_vectors      — submission/answers.json `vectors` matches the truth for
        ALL lookup keys (each a 4-float vector, ±1e-6);
    A3_online_read  — an INDEPENDENT online-store read of 5 lookup keys
        matches the truth. Runs on every adapter whose checker implements
        `get_records` (hopsworks via the default SQL serving path; aws via
        the Feature Store online API). Skipped on `none` and where the
        platform realization has no independent online read path (databricks:
        feature tables are plain Unity Catalog tables read via SQL warehouse).

Usage:
    python -m evals.inference.online.grade --instance <dir> --adapter <hopsworks|databricks|aws|azure|gcp|none>
(cwd must be the run dir — the provider runs graders there.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from evals.common import (
    Suite,
    grade_platform_main,
    load_answers,
    state_checker,
    table_exists_info,
)

TOL = 1e-6


def grade(instance_dir: Path, adapter: str, run_dir: Path) -> dict:
    truth = json.loads((Path(instance_dir) / "solution" / "truth.json").read_text())
    feats, keys = truth["features"], truth["keys"]
    s = Suite()
    check = s.check

    # A1 — feature table exists
    if adapter == "none":
        s.skip("A1_table_exists", "no checker adapter (platform none) — skipped")
    else:
        info = table_exists_info(adapter, truth["table_name"], truth["table_version"])
        check(
            "A1_table_exists",
            info is not None,
            "" if info else f"feature table {truth['table_name']!r} not found",
        )

    # A2 — answered vectors match truth for every lookup key
    answers = load_answers(Path(run_dir) / "submission" / "answers.json")
    vectors = (answers or {}).get("vectors") if isinstance(answers, dict) else None
    if not isinstance(vectors, dict):
        check("A2_vectors", False, "submission/answers.json must contain a 'vectors' object")
    else:
        bad = []
        for k in keys:
            got = vectors.get(k)
            want = truth["vectors"][k]
            try:
                ok = (
                    isinstance(got, list)
                    and len(got) == len(want)
                    and all(abs(float(g) - w) <= TOL for g, w in zip(got, want))
                )
            except (TypeError, ValueError):
                ok = False
            if not ok:
                bad.append(k)
        check("A2_vectors", not bad, f"wrong/missing vectors for keys: {bad[:5]}" if bad else "")

    # A3 — independent online read: the grader reads the lookup keys back from
    # the platform's ONLINE store itself, proving the rows were materialized for
    # low-latency serving (not merely answered from the agent's submission,
    # which A2 cannot distinguish from a CSV / offline read). Runs on every
    # adapter whose checker implements `get_records`; honestly skipped where the
    # platform realization has no independent online read path (e.g. databricks,
    # whose feature tables are plain Unity Catalog tables read via SQL warehouse).
    if adapter == "none":
        s.skip("A3_online_read", "no checker adapter (platform none) — skipped")
    else:
        checker = state_checker(adapter)
        if (
            checker is None
            or not hasattr(checker, "get_records")
            or not getattr(checker, "supports_online_read", True)
        ):
            s.skip(
                "A3_online_read",
                f"no independent online read path for adapter {adapter!r}",
            )
        else:
            try:
                df = checker.get_records(truth["table_name"], keys[:5])
                have_acct = (not df.empty) and ("account_id" in df.columns)
                bad = []
                for k in keys[:5]:
                    row = df[df["account_id"].astype(str) == k] if have_acct else df.iloc[0:0]
                    want = truth["vectors"][k]
                    ok = not row.empty and all(
                        abs(float(row.iloc[0][f]) - w) <= TOL for f, w in zip(feats, want)
                    )
                    if not ok:
                        bad.append(k)
                check(
                    "A3_online_read",
                    not bad,
                    f"online store disagrees for keys: {bad}" if bad else "",
                )
            except Exception as e:
                check("A3_online_read", False, f"online read failed: {e}")

    return s.report(family="online", seed=truth["seed"])


def main(argv=None) -> int:
    return grade_platform_main("online", grade, argv)


if __name__ == "__main__":
    sys.exit(main())
