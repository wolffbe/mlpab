"""Databricks checker adapter — grader-side reads through the workspace.

Uses the databricks-sdk (`WorkspaceClient()` with DATABRICKS_HOST /
DATABRICKS_TOKEN from the env). All reads go through SQL statement execution
on a SQL warehouse — the same read path a Databricks consumer would use.
Requires `DATABRICKS_WAREHOUSE_ID` in the env (setup.py exports it), else the
first available warehouse is used.

Platform realization conventions (Databricks has no feature-view/training-
dataset primitive in the plain SDK surface):
  * feature tables live in the `workspace.default` schema;
  * a "versioned training dataset named X, version N" is the table `X_vN`
    (fallback: `X`) in the same schema.

The SDK is imported lazily so this module can be imported without
databricks-sdk installed.

CLI (for live probing):
    python -m evals.adapters.databricks describe-fg --name transactions
    python -m evals.adapters.databricks read-td --name churn_training --version 1 --out /tmp/td.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from evals.adapters import TableInfo

SCHEMA = "workspace.default"


class DatabricksChecker:
    def __init__(self) -> None:
        from databricks.sdk import WorkspaceClient  # lazy

        self._w = WorkspaceClient()
        self._warehouse = os.environ.get("DATABRICKS_WAREHOUSE_ID") or self._first_warehouse()

    def _first_warehouse(self) -> str:
        ws = list(self._w.warehouses.list())
        if not ws:
            raise RuntimeError(
                "no SQL warehouse available — set DATABRICKS_WAREHOUSE_ID or "
                "pre-provision one (see configs/platforms/databricks/setup.py)"
            )
        return ws[0].id

    def _sql(self, statement: str) -> pd.DataFrame:
        r = self._w.statement_execution.execute_statement(
            statement=statement, warehouse_id=self._warehouse, wait_timeout="50s",
        )
        state = r.status.state.value if r.status and r.status.state else "?"
        if state != "SUCCEEDED":
            err = getattr(r.status, "error", None)
            raise RuntimeError(f"statement {state}: {getattr(err, 'message', '')}")
        cols = [c.name for c in r.manifest.schema.columns]
        rows = r.result.data_array if r.result and r.result.data_array else []
        return pd.DataFrame(rows, columns=cols)

    # -- feature tables ----------------------------------------------------

    def get_feature_table(self, name: str, version: int | None = None) -> TableInfo | None:
        full = f"{SCHEMA}.{name}"
        try:
            t = self._w.tables.get(full)
        except Exception:
            return None
        pk: list[str] = []
        for c in (getattr(t, "table_constraints", None) or []):
            pkc = getattr(c, "primary_key_constraint", None)
            if pkc is not None:
                pk = list(pkc.child_columns or [])
        return TableInfo(
            name=name,
            version=None,
            primary_key=pk,
            event_time=None,   # Unity Catalog has no event-time concept → capability note
            schema={c.name: str(c.type_name.value if c.type_name else c.type_text)
                    for c in (t.columns or [])},
        )

    def read_rows(self, name: str, version: int | None = None) -> pd.DataFrame:
        return self._sql(f"SELECT * FROM {SCHEMA}.{name}")

    # -- training datasets ---------------------------------------------------

    def read_training_dataset(self, name: str, version: int = 1) -> pd.DataFrame:
        for table in (f"{name}_v{version}", name):
            if self.get_feature_table(table) is not None:
                return self.read_rows(table)
        raise LookupError(
            f"training dataset {name!r} v{version} not found — expected table "
            f"{SCHEMA}.{name}_v{version} (or {SCHEMA}.{name})"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("describe-fg")
    p.add_argument("--name", required=True)
    p = sub.add_parser("read-fg")
    p.add_argument("--name", required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("read-td")
    p.add_argument("--name", required=True)
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    checker = DatabricksChecker()
    if args.cmd == "describe-fg":
        info = checker.get_feature_table(args.name)
        print(json.dumps({"exists": info is not None,
                          **(info.__dict__ if info else {"name": args.name})},
                         default=str, indent=2))
        return 0 if info else 1
    if args.cmd == "read-fg":
        checker.read_rows(args.name).to_csv(args.out, index=False)
    else:
        checker.read_training_dataset(args.name, args.version).to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# --------------------------------------------------------------------------
# Platform-state reads beyond Unity Catalog tables (best-effort dicts:
# graders assert on `exists` plus whichever detail keys the API provides).
# --------------------------------------------------------------------------

def _state_reads(cls):
    def get_model(self, name: str, version: int = 1) -> dict:
        """Registered model: UC registry (workspace.default.<name>) first,
        then the legacy workspace (MLflow) registry."""
        try:
            m = self._w.registered_models.get(f"{SCHEMA}.{name}")
            return {"exists": True, "registry": "unity-catalog",
                    "full_name": m.full_name}
        except Exception:
            pass
        try:
            r = self._w.api_client.do(
                "GET", "/api/2.0/mlflow/registered-models/get",
                query={"name": name})
            versions = (r.get("registered_model") or {}).get("latest_versions") or []
            return {"exists": True, "registry": "workspace-mlflow",
                    "version": max((int(v.get("version", 0)) for v in versions), default=None)}
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def get_job(self, name: str) -> dict:
        """Jobs API: {exists, scheduled, last_run_state, notifications}."""
        try:
            jobs = list(self._w.jobs.list(name=name))
            if not jobs:
                return {"exists": False}
            j = self._w.jobs.get(jobs[0].job_id)
            s = j.settings
            out = {"exists": True,
                   "scheduled": bool(getattr(s, "schedule", None)
                                     or getattr(s, "trigger", None)
                                     or getattr(s, "continuous", None)),
                   "notifications": bool(getattr(s, "email_notifications", None)
                                         or getattr(s, "webhook_notifications", None))}
            try:
                runs = list(self._w.jobs.list_runs(job_id=jobs[0].job_id, limit=1))
                if runs and runs[0].state:
                    out["last_run_state"] = str(
                        runs[0].state.result_state or runs[0].state.life_cycle_state)
            except Exception:
                pass
            return out
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def get_endpoint(self, name: str) -> dict:
        """Model serving endpoint: {exists, status}."""
        try:
            e = self._w.serving_endpoints.get(name)
            state = getattr(e, "state", None)
            return {"exists": True,
                    "status": str(getattr(state, "ready", "") or "")}
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def get_alert(self, name_or_hint: str) -> dict:
        """Alerting realization on Databricks = job notification settings;
        report whether the named JOB carries any (see get_job)."""
        job = get_job(self, name_or_hint)
        return {"exists": bool(job.get("exists") and job.get("notifications")),
                **({"error": job["error"]} if job.get("error") else {})}

    def get_vector_store(self, name: str) -> dict:
        """Vector store realization on Databricks: a Vector Search index (an
        index needs a vector search ENDPOINT first; Direct Vector Access
        indexes accept upserted vectors; index names are 3-part Unity Catalog
        `catalog.schema.name`). Pure REST against the workspace: scan
        /api/2.0/vector-search/endpoints, list each endpoint's indexes, and
        match any index whose name CONTAINS `name` (lenient on the
        catalog.schema prefix)."""
        try:
            eps = self._w.api_client.do("GET", "/api/2.0/vector-search/endpoints") or {}
            ep_names = [e.get("name") for e in (eps.get("endpoints") or [])]
            hits = []
            for ep in ep_names:
                try:
                    r = self._w.api_client.do(
                        "GET", "/api/2.0/vector-search/indexes",
                        query={"endpoint_name": ep}) or {}
                except Exception:
                    continue
                hits += [{"index": idx.get("name"), "endpoint": ep}
                         for idx in (r.get("vector_indexes") or [])
                         if name in (idx.get("name") or "")]
            return {"exists": bool(hits), "kind": "vector-search-index",
                    "matches": hits, "endpoints": ep_names}
        except Exception as e:
            return {"exists": False, "error": str(e)}

    cls.get_model = get_model
    cls.get_job = get_job
    cls.get_endpoint = get_endpoint
    cls.get_alert = get_alert
    cls.get_vector_store = get_vector_store
    return cls


_state_reads(DatabricksChecker)
