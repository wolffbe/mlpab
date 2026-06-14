"""Lineage task — grader (platform kind).

Asserts:
    A1_derived_content — the derived table's content (columns/rows/digest with named
        naive-variant diagnosis), read through the platform adapter, or from
        submission/<table_name>.csv on adapter `none`;
    A2_sources_exist  — the truth.json source tables exist as feature tables (v1)
        (metadata read; skipped-pass on adapter `none`);
    A3_lineage_answer — submission/answers.json `derived_from` equals the
        sorted source list.

Usage:
    python -m evals.ops.lineage.grade --instance <dir> --adapter <hopsworks|databricks|sagemaker|none>
(cwd must be the run dir — the provider runs graders there.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from evals.common import (
    canonicalize,
    digest,
    fetch_table,
    grade_platform_main,
    load_answers,
    table_exists_info,
)


def grade(instance_dir: Path, adapter: str, run_dir: Path) -> dict:
    truth = json.loads((Path(instance_dir) / "solution" / "truth.json").read_text())
    spec, table = truth["spec"], truth["table_name"]
    asserts: list[dict] = []
    diagnostic = None

    def check(name, ok, detail=""):
        asserts.append({"name": name, "passed": bool(ok), **({"detail": detail} if detail else {})})
        return bool(ok)

    # A1 — derived table content
    produced, src_err = None, ""
    if adapter == "none":
        local = Path(run_dir) / "submission" / f"{table}.csv"
        if local.exists():
            produced = pd.read_csv(local)
        else:
            src_err = f"no local deliverable at {local}"
    else:
        try:
            produced = fetch_table(adapter, table, truth["table_version"], truth.get("record_ids"))
        except (LookupError, NotImplementedError) as e:
            src_err = str(e)
    a1 = False
    if produced is None:
        check("A1_derived_content", False, src_err)
    else:
        try:
            norm = canonicalize(produced, spec)
            d = digest(norm)
            a1 = (len(norm) == truth["row_count"]) and d == truth["digest"]
            detail = ""
            if not a1:
                detail = f"content mismatch (rows {len(norm)} vs {truth['row_count']})"
                for vname, vdig in truth.get("variant_digests", {}).items():
                    if d == vdig:
                        diagnostic = truth.get("variant_diagnosis", {}).get(vname, vname)
            check("A1_derived_content", a1, detail)
        except Exception as e:
            check("A1_derived_content", False, f"could not normalize: {e}")

    # A2 — source tables exist
    if adapter == "none":
        a2 = check("A2_sources_exist", True, "no checker adapter (platform none) — skipped")
    else:
        missing = [
            s
            for s in truth["source_tables"]
            if table_exists_info(adapter, s, truth["table_version"]) is None
        ]
        a2 = check(
            "A2_sources_exist", not missing, f"missing source tables: {missing}" if missing else ""
        )

    # A3 — lineage answer
    answers = load_answers(Path(run_dir) / "submission" / "answers.json")
    if not isinstance(answers, dict) or "_parse_error" in (answers or {}):
        a3 = check(
            "A3_lineage_answer", False, "no parseable submission/answers.json with 'derived_from'"
        )
    else:
        got = answers.get("derived_from")
        want = sorted(truth["source_tables"])
        ok = isinstance(got, list) and sorted(str(x).strip().lower() for x in got) == want
        a3 = check("A3_lineage_answer", ok, "" if ok else f"derived_from {got!r} != {want}")

    success = a1 and a2 and a3
    return {
        "family": "lineage",
        "seed": truth["seed"],
        "success": success,
        "asserts_passed": sum(a["passed"] for a in asserts),
        "asserts_total": len(asserts),
        "asserts": asserts,
        **({"diagnostic": diagnostic} if diagnostic else {}),
    }


def main(argv=None) -> int:
    return grade_platform_main("lineage", grade, argv)


if __name__ == "__main__":
    sys.exit(main())
