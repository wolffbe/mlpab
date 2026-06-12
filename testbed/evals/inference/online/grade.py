"""Online-serving task — grader (platform kind).

Asserts:
    A1_table_exists — the `profiles` feature table exists (metadata read;
        skipped-pass on adapter `none`);
    A2_vectors      — submission/answers.json `vectors` matches the truth for
        ALL lookup keys (each a 4-float vector, ±1e-6);
    A3_online_read  — an INDEPENDENT online-store read of 5 lookup keys
        matches the truth (sagemaker only; skipped-pass on `none` and on
        hopsworks/databricks, where the testbed has no independent online
        read path).

Usage:
    python -m evals.inference.online.grade --instance <dir> --adapter <hopsworks|databricks|sagemaker|none>
(cwd must be the run dir — the provider runs graders there.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from evals.common import grade_platform_main, load_answers, state_checker, table_exists_info

TOL = 1e-6


def grade(instance_dir: Path, adapter: str, run_dir: Path) -> dict:
    truth = json.loads((Path(instance_dir) / "solution" / "truth.json").read_text())
    feats, keys = truth["features"], truth["keys"]
    asserts: list[dict] = []

    def check(name, ok, detail=""):
        asserts.append({"name": name, "passed": bool(ok),
                        **({"detail": detail} if detail else {})})
        return bool(ok)

    # A1 — feature table exists
    if adapter == "none":
        a1 = check("A1_table_exists", True,
                   "no checker adapter (platform none) — skipped")
    else:
        info = table_exists_info(adapter, truth["table_name"], truth["table_version"])
        a1 = check("A1_table_exists", info is not None,
                   "" if info else f"feature table {truth['table_name']!r} not found")

    # A2 — answered vectors match truth for every lookup key
    answers = load_answers(Path(run_dir) / "submission" / "answers.json")
    vectors = (answers or {}).get("vectors") if isinstance(answers, dict) else None
    if not isinstance(vectors, dict):
        a2 = check("A2_vectors", False,
                   "submission/answers.json must contain a 'vectors' object")
    else:
        bad = []
        for k in keys:
            got = vectors.get(k)
            want = truth["vectors"][k]
            try:
                ok = (isinstance(got, list) and len(got) == len(want)
                      and all(abs(float(g) - w) <= TOL for g, w in zip(got, want)))
            except (TypeError, ValueError):
                ok = False
            if not ok:
                bad.append(k)
        a2 = check("A2_vectors", not bad,
                   f"wrong/missing vectors for keys: {bad[:5]}" if bad else "")

    # A3 — independent online read (sagemaker only)
    if adapter == "sagemaker":
        try:
            df = state_checker(adapter).get_records(truth["table_name"], keys[:5])
            bad = []
            for k in keys[:5]:
                row = df[df["account_id"].astype(str) == k]
                want = truth["vectors"][k]
                ok = (not row.empty and all(
                    abs(float(row.iloc[0][f]) - w) <= TOL for f, w in zip(feats, want)))
                if not ok:
                    bad.append(k)
            a3 = check("A3_online_read", not bad,
                       f"online store disagrees for keys: {bad}" if bad else "")
        except Exception as e:
            a3 = check("A3_online_read", False, f"online read failed: {e}")
    elif adapter == "none":
        a3 = check("A3_online_read", True,
                   "no checker adapter (platform none) — skipped")
    else:
        a3 = check("A3_online_read", True,
                   "independent online read only implemented for sagemaker")

    success = a1 and a2 and a3
    return {"family": "online", "seed": truth["seed"], "success": success,
            "asserts_passed": sum(a["passed"] for a in asserts),
            "asserts_total": len(asserts), "asserts": asserts}


def main(argv=None) -> int:
    return grade_platform_main("online", grade, argv)


if __name__ == "__main__":
    sys.exit(main())
