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
import time
from pathlib import Path

import pandas as pd

# Read-back can flake on the client even when the server returns data: hsfs
# collapses the real Arrow Flight error into a generic FeatureStoreException
# (see arrow_flight_client.afs_error_handler_wrapper). The server serves the
# rows reliably on repeat, so retry transient read failures before giving up.
# LookupError/NotImplementedError are deterministic adapter limits — not retried.
# Backoff is exponential (5, 10, 20, 30, …s capped) because a fixed 3×5s window
# proved too short for the cluster's flake — a healthy read would be marked a
# false-negative read-back failure, counting against the model unfairly.
FETCH_RETRIES = 5
FETCH_BACKOFF_S = 5
FETCH_BACKOFF_CAP_S = 30

# Server-side signatures that mean the read failed for a DETERMINISTIC reason,
# NOT the transient client flake: the feature group exists in metadata but the
# offline store has no parquet/delta files because nothing was ever ingested.
# The Arrow Flight server states this plainly ("No active delta files found ...
# no data has been written yet"), but hsfs collapses it into the same generic
# `Could not read data using Hopsworks Query Service` as a real transport flake.
# Without this discriminator an empty FG (a genuine ingest failure by the model)
# is retried FETCH_RETRIES times for ~65s and then recorded byte-identically to a
# transport flake — masking a capability gap as infrastructure noise. Verified
# live 2026-06-18: a registered-but-empty FG read raises exactly this, while a
# 2-row FG reads back in 0.59s on the same cluster.
_EMPTY_FG_SIGNATURES = (
    "no active delta files",
    "no data has been written",
)


def _cause_chain(e):
    """Flatten an exception's __cause__/__context__ chain to a list of
    "Type: message" strings. hsfs masks the real Arrow Flight error behind a
    generic FeatureStoreException (arrow_flight_client.afs_error_handler_wrapper),
    so the actionable detail — the server's own message — lives in the chain,
    not in str(e)."""
    out, seen, cur = [], set(), e
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        out.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    return out


def _is_empty_feature_group(e) -> bool:
    blob = " ".join(_cause_chain(e)).lower()
    return any(sig in blob for sig in _EMPTY_FG_SIGNATURES)


def read_with_retry(read_fn):
    """Run a platform read (a zero-arg callable), retrying the transient
    client-side HQS flake with exponential backoff. THE single retry policy for
    every grader read-back — table reads and training-dataset reads alike route
    through here. LookupError/NotImplementedError are deterministic misses,
    re-raised immediately. A read against a registered-but-empty feature group is
    ALSO deterministic — re-raised at once as a LookupError with a distinct
    reason, never retried (retrying an empty FG only burns ~65s). Any other error
    is treated as the read-back flake and retried up to FETCH_RETRIES times, then
    re-raised with the masked root cause appended so the grader's recorded reason
    distinguishes one failure mode from another (the generic FeatureStoreException
    string is identical for every cause)."""
    last = None
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            return read_fn()
        except (LookupError, NotImplementedError):
            raise  # deterministic — retrying cannot change the outcome
        except Exception as e:  # noqa: BLE001 — client read flake; retry
            if _is_empty_feature_group(e):
                # FG exists but no rows were ever ingested — deterministic, and a
                # genuine capability gap (not infra noise). Surface it as such.
                raise LookupError(
                    "feature group exists but has no data (no rows were ingested)"
                ) from e
            last = e
            if attempt < FETCH_RETRIES:
                time.sleep(min(FETCH_BACKOFF_S * 2 ** (attempt - 1), FETCH_BACKOFF_CAP_S))
    # all retries exhausted on a genuine flake — append the masked root cause so
    # the CSV reason is actionable instead of the uniform generic message.
    raise RuntimeError(f"{last} [root cause: {' <- '.join(_cause_chain(last))}]") from last


def fetch_table_with_retry(adapter, table, version, record_ids):
    """Read a feature table back, retrying the transient client-side flake.

    Every grader should read through THIS (or `read_with_retry`), not raw
    `fetch_table`: a bare `fetch_table` call that hits the hsfs
    FeatureStoreException flake crashes the grader (no report)."""
    return read_with_retry(lambda: fetch_table(adapter, table, version, record_ids))


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
                "%Y-%m-%dT%H:%M:%SZ"
            )
        elif c in spec.get("int_cols", []):
            out[c] = pd.to_numeric(out[c]).astype("int64")
        elif pd.api.types.is_float_dtype(out[c]) or c in spec.get("float_cols", []):
            out[c] = pd.to_numeric(out[c]).astype(float).round(6)
        elif pd.api.types.is_object_dtype(out[c]):
            out[c] = out[c].astype(str)
    return out


def digest(df: pd.DataFrame) -> str:
    return "sha256:" + hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


def fetch_table(
    adapter: str, table: str, version: int, record_ids: list[str] | None
) -> pd.DataFrame:
    """Read a feature table back through the platform (see module docstring)."""
    if adapter == "hopsworks":
        from evals.adapters.hopsworks import HopsworksChecker

        return HopsworksChecker().read_rows(table, version)
    if adapter == "databricks":
        from evals.adapters.databricks import DatabricksChecker

        return DatabricksChecker().read_rows(table)
    if adapter == "aws":
        from evals.adapters.aws import SageMakerChecker

        if not record_ids:
            raise LookupError("sagemaker table reads need the truth's record ids")
        df = SageMakerChecker().get_records(table, record_ids)
        if df.empty:
            raise LookupError(f"feature group {table!r}: online store returned no records")
        return df
    if adapter == "azure":
        from evals.adapters.azure import AzureMLChecker

        return AzureMLChecker().read_rows(table, version)
    if adapter == "gcp":
        from evals.adapters.gcp import VertexChecker

        return VertexChecker().read_rows(table, version)
    raise LookupError(f"no checker adapter for {adapter!r}")


def table_exists_info(adapter: str, table: str, version: int):
    """TableInfo via the platform's metadata read, or None."""
    if adapter == "hopsworks":
        from evals.adapters.hopsworks import HopsworksChecker

        return HopsworksChecker().get_feature_table(table, version)
    if adapter == "databricks":
        from evals.adapters.databricks import DatabricksChecker

        return DatabricksChecker().get_feature_table(table)
    if adapter == "aws":
        from evals.adapters.aws import SageMakerChecker

        return SageMakerChecker().get_feature_table(table)
    if adapter == "azure":
        from evals.adapters.azure import AzureMLChecker

        return AzureMLChecker().get_feature_table(table, version)
    if adapter == "gcp":
        from evals.adapters.gcp import VertexChecker

        return VertexChecker().get_feature_table(table, version)
    return None


def tally(asserts: list[dict]) -> dict:
    """Roll an assertion list up into the standard outcome fields.

    Every assert carries a `status` in {"pass","fail","skip"}. `skip` means the
    check was not reached (a prerequisite failed) or not applicable on this
    platform (e.g. the `platform none` baseline) — it neither passes nor fails
    the task. `success` = no failed asserts AND at least one passed (an
    all-skipped suite verified nothing, so it is not a success)."""
    p = sum(a.get("status") == "pass" for a in asserts)
    f = sum(a.get("status") == "fail" for a in asserts)
    s = sum(a.get("status") == "skip" for a in asserts)
    return {
        "success": f == 0 and p >= 1,
        "asserts_passed": p,
        "asserts_failed": f,
        "asserts_skipped": s,
        "total_asserts": len(asserts),
    }


class Suite:
    """Accumulates an assertion suite with three-way status, so the FULL suite
    size is always reported even when later checks are skipped.

    - `check(name, ok, detail)` records a pass/fail and returns the bool. Detail
      is kept only on failure unless `detail_on_pass=True` (for checks whose
      pass carries an informative note, e.g. "platform metrics differ").
    - `skip(name, detail)` records a not-reached / not-applicable assert and
      returns True (non-blocking — skips never fail the task).
    - `report(**extra)` returns the tally fields + the assert list + extras.

    Each assert keeps a legacy `passed` bool (== status == "pass") so existing
    consumers (the first-assert `deliverable_exists` probe) keep working."""

    def __init__(self) -> None:
        self.asserts: list[dict] = []

    def check(self, name: str, ok, detail: str = "", *, detail_on_pass: bool = False) -> bool:
        ok = bool(ok)
        keep_detail = bool(detail) and (not ok or detail_on_pass)
        self.asserts.append(
            {
                "name": name,
                "status": "pass" if ok else "fail",
                "passed": ok,
                **({"detail": detail} if keep_detail else {}),
            }
        )
        return ok

    def skip(self, name: str, detail: str = "") -> bool:
        self.asserts.append(
            {
                "name": name,
                "status": "skip",
                "passed": False,
                **({"detail": detail} if detail else {}),
            }
        )
        return True

    def report(self, **extra) -> dict:
        return {**tally(self.asserts), "asserts": self.asserts, **extra}


def read_csv_or_empty(path) -> pd.DataFrame:
    """Read a CSV deliverable, returning an EMPTY frame when the file is missing
    or empty (0 bytes). Lets a grader still enumerate its full suite (the
    deliverable check fails on missing columns, dependents skip) instead of
    crashing on a `pandas.errors.EmptyDataError` for a 0-byte file."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def grade_table_main(family: str, grade_fn, argv: list[str] | None = None) -> int:
    """CLI shared by table-deliverable families: --instance + (--csv | --adapter).
    `grade_fn(instance_dir, produced_df, adapter_or_none) -> report`.

    A missing/empty `--csv` or an unreadable platform deliverable yields an EMPTY
    DataFrame rather than an early return, so the grader still enumerates its
    full assert suite (deliverable check fails, dependents skip). The reason the
    deliverable was unreadable is surfaced on the report's `error` key so it
    reaches the results CSV (rather than a generic "missing columns")."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=Path, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path)
    src.add_argument("--adapter", choices=["hopsworks", "databricks", "aws", "azure", "gcp"])
    args = ap.parse_args(argv)

    deliverable_err = None
    if args.csv:
        if not args.csv.exists():
            produced = pd.DataFrame()
            deliverable_err = f"no deliverable produced at {args.csv}"
        else:
            produced = read_csv_or_empty(args.csv)
    else:
        truth = json.loads((args.instance / "solution" / "truth.json").read_text())
        try:
            produced = fetch_table_with_retry(
                args.adapter,
                truth["table_name"],
                truth.get("table_version", 1),
                truth.get("record_ids"),
            )
        except (LookupError, NotImplementedError) as e:
            # Deliverable could not be read back — grade an EMPTY frame so the
            # suite still enumerates (deliverable check fails, dependents skip),
            # but keep the real reason for the report.
            produced = pd.DataFrame()
            deliverable_err = str(e)
        except Exception as e:  # noqa: BLE001
            # A client-side read-back flake (e.g. hsfs' masked FeatureStoreException)
            # must NOT crash the grader — that produces no report at all. Degrade to
            # an empty frame and surface the real reason on the report instead.
            produced = pd.DataFrame()
            deliverable_err = f"read-back failed after {FETCH_RETRIES} attempts: {e}"
    report = grade_fn(args.instance, produced, args.adapter)
    if deliverable_err:
        report.setdefault("error", deliverable_err)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["success"] else 1


def grade_table_content(family: str, instance_dir: Path, produced: pd.DataFrame) -> dict:
    """The standard table assert suite: columns → row count → content digest,
    with named diagnosis via the generator's known-wrong variant digests."""
    truth = json.loads((instance_dir / "solution" / "truth.json").read_text())
    spec = truth["spec"]
    s = Suite()
    diagnostic = None

    produced.columns = [str(c).strip().lower() for c in produced.columns]
    missing = [c for c in spec["columns"] if c not in produced.columns]
    a1 = s.check("A1_columns", not missing, f"missing columns: {missing}")
    if not a1:
        # No gradeable deliverable — the content checks are unreachable.
        s.skip("A2_row_count", "skipped: required columns are missing")
        s.skip("A3_content", "skipped: required columns are missing")
    else:
        try:
            norm = canonicalize(produced, spec)
        except Exception as e:
            s.check("A2_row_count", False, f"could not normalize: {e}")
            s.skip("A3_content", "skipped: could not normalize the deliverable")
            norm = None
        if norm is not None:
            s.check(
                "A2_row_count",
                len(norm) == truth["row_count"],
                f"got {len(norm)}, expected {truth['row_count']}",
            )
            d = digest(norm)
            a3 = s.check("A3_content", d == truth["digest"], "content digest mismatch")
            if not a3:
                for vname, vdig in truth.get("variant_digests", {}).items():
                    if d == vdig:
                        diagnostic = truth.get("variant_diagnosis", {}).get(vname, vname)
                        break

    return {
        "family": family,
        "seed": truth["seed"],
        **tally(s.asserts),
        "asserts": s.asserts,
        **({"diagnostic": diagnostic} if diagnostic else {}),
    }


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

    # A missing answers file is passed through as None so the grader still
    # enumerates its full suite (deliverable check fails, dependents skip).
    answers = load_answers(args.answers)
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
    if adapter == "aws":
        from evals.adapters.aws import SageMakerChecker

        return SageMakerChecker()
    if adapter == "azure":
        from evals.adapters.azure import AzureMLChecker

        return AzureMLChecker()
    if adapter == "gcp":
        from evals.adapters.gcp import VertexChecker

        return VertexChecker()
    return None


def grade_platform_main(family: str, grade_fn, argv: list[str] | None = None) -> int:
    """CLI shared by platform-kind families: --instance + --adapter <name|none>.
    `grade_fn(instance_dir, adapter_name, run_dir_cwd) -> report` — the grade
    module reads whatever mix of submission/ files and platform state it needs
    (the provider runs graders with cwd = the run dir)."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=Path, required=True)
    ap.add_argument(
        "--adapter",
        required=True,
        choices=["hopsworks", "databricks", "aws", "azure", "gcp", "none"],
    )
    args = ap.parse_args(argv)
    report = grade_fn(args.instance, args.adapter, Path.cwd())
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["success"] else 1
