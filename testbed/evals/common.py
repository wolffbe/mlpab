"""Shared grading plumbing for eval families.

- `canonicalize` / `digest`: the canonical comparable form every grader uses
  (spec'd columns only, canonical sort, dtype casts, 6 dp floats, UTC ISO
  timestamps) — representation noise must never fail a correct solution.
- `fetch_table`: read a feature table back THROUGH a platform's checker
  adapter (the deliverable counts only if the platform returns it).
- `grade_answers_cli` / `load_answers`: the boilerplate for answers.json
  families (drift/skew/leakage).

Platform realization notes for `fetch_table`:
  hopsworks  — feature group read via the query service (offline store).
  databricks — `SELECT *` on the Unity Catalog table via a SQL warehouse.
  sagemaker  — online-store `batch_get_record` for the truth's record ids
               (the offline store materializes asynchronously and Athena is
               outside the testbed policy), so table tasks carry a unique
               per-row record key and require the online store enabled.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def instance_suffix(seed: int) -> str:
    """6-hex suffix derived from the instance seed (deterministic, reproducible).
    Appended to every PLATFORM resource name a task pins (tables, jobs,
    endpoints, models, datasets) so a new run can never collide with a
    half-deleted predecessor's resources — deletion is asynchronous on
    sagemaker (feature groups, endpoints) and databricks (serving endpoints).
    Bases are single-token lowercase alphanumeric: SageMaker feature-group
    names forbid `_`, Hopsworks forbids `-`; pure [a-z0-9] works everywhere."""
    return format(seed & 0xFFFFFF, "06x")


def canonicalize(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Normalize to the canonical comparable form per the truth spec:
    {columns: [...], ts_cols: [...], int_cols: [...], sort_cols: [...]}."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    out = out[spec["columns"]].sort_values(spec["sort_cols"]).reset_index(drop=True)
    for c in spec["columns"]:
        if c in spec.get("ts_cols", []):
            out[c] = pd.to_datetime(out[c], utc=True, format="mixed").dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ")
        elif c in spec.get("int_cols", []):
            out[c] = pd.to_numeric(out[c]).astype("int64")
        elif pd.api.types.is_float_dtype(out[c]) or c in spec.get("float_cols", []):
            out[c] = pd.to_numeric(out[c]).astype(float).round(6)
        elif pd.api.types.is_object_dtype(out[c]):
            out[c] = out[c].astype(str)
    return out


def digest(df: pd.DataFrame) -> str:
    return "sha256:" + hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


def fetch_table(adapter: str, table: str, version: int, record_ids: list[str] | None) -> pd.DataFrame:
    """Read a feature table back through the platform (see module docstring)."""
    if adapter == "hopsworks":
        from evals.adapters.hopsworks import HopsworksChecker
        return HopsworksChecker().read_rows(table, version)
    if adapter == "databricks":
        from evals.adapters.databricks import DatabricksChecker
        return DatabricksChecker().read_rows(table)
    if adapter == "sagemaker":
        from evals.adapters.sagemaker import SageMakerChecker
        if not record_ids:
            raise LookupError("sagemaker table reads need the truth's record ids")
        df = SageMakerChecker().get_records(table, record_ids)
        if df.empty:
            raise LookupError(f"feature group {table!r}: online store returned no records")
        return df
    raise LookupError(f"no checker adapter for {adapter!r}")


def table_exists_info(adapter: str, table: str, version: int):
    """TableInfo via the platform's metadata read, or None."""
    if adapter == "hopsworks":
        from evals.adapters.hopsworks import HopsworksChecker
        return HopsworksChecker().get_feature_table(table, version)
    if adapter == "databricks":
        from evals.adapters.databricks import DatabricksChecker
        return DatabricksChecker().get_feature_table(table)
    if adapter == "sagemaker":
        from evals.adapters.sagemaker import SageMakerChecker
        return SageMakerChecker().get_feature_table(table)
    return None


def grade_table_main(family: str, grade_fn, argv: list[str] | None = None) -> int:
    """CLI shared by table-deliverable families: --instance + (--csv | --adapter).
    `grade_fn(instance_dir, produced_df, adapter_or_none) -> report`."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=Path, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path)
    src.add_argument("--adapter", choices=["hopsworks", "databricks", "sagemaker"])
    args = ap.parse_args(argv)

    truth = json.loads((args.instance / "solution" / "truth.json").read_text())
    if args.csv:
        produced = pd.read_csv(args.csv)
    else:
        try:
            produced = fetch_table(args.adapter, truth["table_name"],
                                   truth.get("table_version", 1), truth.get("record_ids"))
        except (LookupError, NotImplementedError) as e:
            report = {"family": family, "seed": truth["seed"], "success": False,
                      "asserts_passed": 0, "asserts_total": 1,
                      "asserts": [{"name": "A0_deliverable_exists", "passed": False,
                                   "detail": str(e)}]}
            print(json.dumps(report, indent=2, default=str))
            return 1
    report = grade_fn(args.instance, produced, args.adapter)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["success"] else 1


def grade_table_content(family: str, instance_dir: Path, produced: pd.DataFrame) -> dict:
    """The standard table assert suite: columns → row count → content digest,
    with named diagnosis via the generator's known-wrong variant digests."""
    truth = json.loads((instance_dir / "solution" / "truth.json").read_text())
    spec = truth["spec"]
    asserts: list[dict] = []
    diagnostic = None

    def check(name, ok, detail=""):
        asserts.append({"name": name, "passed": bool(ok),
                        **({"detail": detail} if detail and not ok else {})})
        return bool(ok)

    produced.columns = [str(c).strip().lower() for c in produced.columns]
    missing = [c for c in spec["columns"] if c not in produced.columns]
    a1 = check("A1_columns", not missing, f"missing columns: {missing}")
    a2 = a3 = False
    if a1:
        try:
            norm = canonicalize(produced, spec)
        except Exception as e:
            check("A2_canonical_form", False, f"could not normalize: {e}")
            norm = None
        if norm is not None:
            a2 = check("A2_row_count", len(norm) == truth["row_count"],
                       f"got {len(norm)}, expected {truth['row_count']}")
            d = digest(norm)
            a3 = check("A3_content", d == truth["digest"], "content digest mismatch")
            if not a3:
                for vname, vdig in truth.get("variant_digests", {}).items():
                    if d == vdig:
                        diagnostic = truth.get("variant_diagnosis", {}).get(vname, vname)
                        break

    success = a1 and a2 and a3
    return {"family": family, "seed": truth["seed"], "success": success,
            "asserts_passed": sum(a["passed"] for a in asserts),
            "asserts_total": len(asserts), "asserts": asserts,
            **({"diagnostic": diagnostic} if diagnostic else {})}


def load_answers(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return {"_parse_error": str(e)}


def grade_answers_main(family: str, grade_fn, argv: list[str] | None = None) -> int:
    """CLI shared by answers-deliverable families: --instance + --answers."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=Path, required=True)
    ap.add_argument("--answers", type=Path, required=True)
    args = ap.parse_args(argv)

    answers = load_answers(args.answers)
    if answers is None:
        report = {"family": family, "success": False, "asserts_passed": 0,
                  "asserts_total": 1,
                  "asserts": [{"name": "A0_deliverable_exists", "passed": False,
                               "detail": f"no answers file at {args.answers}"}]}
    else:
        report = grade_fn(args.instance, answers)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["success"] else 1


def state_checker(adapter: str):
    """The platform checker instance for state reads (get_model / get_job /
    get_endpoint / get_alert / get_records …), or None for `none`."""
    if adapter == "hopsworks":
        from evals.adapters.hopsworks import HopsworksChecker
        return HopsworksChecker()
    if adapter == "databricks":
        from evals.adapters.databricks import DatabricksChecker
        return DatabricksChecker()
    if adapter == "sagemaker":
        from evals.adapters.sagemaker import SageMakerChecker
        return SageMakerChecker()
    return None


def grade_platform_main(family: str, grade_fn, argv: list[str] | None = None) -> int:
    """CLI shared by platform-kind families: --instance + --adapter <name|none>.
    `grade_fn(instance_dir, adapter_name, run_dir_cwd) -> report` — the grade
    module reads whatever mix of submission/ files and platform state it needs
    (the provider runs graders with cwd = the run dir)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=Path, required=True)
    ap.add_argument("--adapter", required=True,
                    choices=["hopsworks", "databricks", "sagemaker", "none"])
    args = ap.parse_args(argv)
    report = grade_fn(args.instance, args.adapter, Path.cwd())
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["success"] else 1
