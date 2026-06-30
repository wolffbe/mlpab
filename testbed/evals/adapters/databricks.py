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
import time
from pathlib import Path

import pandas as pd

from evals.adapters import TableInfo

CATALOG = "workspace"
DEFAULT_SCHEMA = "default"
SCHEMA = f"{CATALOG}.{DEFAULT_SCHEMA}"  # conventional landing zone / fallback
# Stable SQL warehouse the grader reads through. setup.py provisions it and
# teardown.py never sweeps it, so every readback has a warm, predictable target
# instead of racing whatever warehouse happens to exist (see _first_warehouse).
GRADER_WAREHOUSE_NAME = "mlpab-grader"


class DatabricksChecker:
    # Poll cadence for a statement that is still PENDING/RUNNING after the
    # server-side wait cap (overridable in tests to avoid real sleeps).
    _POLL_SECS = 3
    _POLL_DEADLINE_SECS = 240

    # Databricks has no realization-agnostic INDEPENDENT online read: a feature
    # table is a plain Unity Catalog table, and online materialization may be a
    # feature-serving endpoint, a UC online table, or a Lakebase synced table.
    # get_records only covers feature serving, so the online grader skips its
    # independent-read assert here rather than bias toward that one realization.
    supports_online_read = False

    def __init__(self) -> None:
        from databricks.sdk import WorkspaceClient  # lazy

        self._w = WorkspaceClient()
        self._warehouse = os.environ.get("DATABRICKS_WAREHOUSE_ID") or self._first_warehouse()
        self._fqn_cache: dict[str, str | None] = {}
        # This run's landing-zone schema "<catalog>.<schema>" (runner sets
        # MLPAB_DATABRICKS_SCHEMA per run; falls back to workspace.default for a
        # manual probe). Preferred when resolving / fabricating table names.
        fqn = (os.environ.get("MLPAB_DATABRICKS_SCHEMA") or SCHEMA).strip()
        self._schema = fqn
        self._catalog, self._schema_name = (fqn.split(".", 1) + [DEFAULT_SCHEMA])[:2]
        # Per-run mode: a run-specific landing schema is set, so the agent's
        # deliverables live in it. Scope reads to it (+ workspace.default) so a
        # leftover same-named table/endpoint from another run — which the
        # metastore-wide search would otherwise pick up — can't be graded as
        # this run's output (cross-run false positive).
        self._per_run = bool(os.environ.get("MLPAB_DATABRICKS_SCHEMA"))

    @staticmethod
    def _wh_state(w) -> str:
        s = getattr(w, "state", None)
        return str(getattr(s, "value", s) or "").upper()

    def _first_warehouse(self) -> str:
        ws = list(self._w.warehouses.list())
        if not ws:
            raise RuntimeError(
                "no SQL warehouse available — set DATABRICKS_WAREHOUSE_ID or "
                "pre-provision one (see configs/platforms/databricks/setup.py)"
            )
        # Prefer the stable grader warehouse, then any already-warm warehouse,
        # else the first listed — minimises cold starts and avoids picking a
        # warehouse a parallel run's teardown is about to sweep.
        named = [w for w in ws if getattr(w, "name", None) == GRADER_WAREHOUSE_NAME]
        warm = [w for w in ws if self._wh_state(w) in ("RUNNING", "STARTING")]
        return (named or warm or ws)[0].id

    def _sql(self, statement: str) -> pd.DataFrame:
        r = self._w.statement_execution.execute_statement(
            statement=statement,
            warehouse_id=self._warehouse,
            wait_timeout="50s",
        )

        def _state(resp) -> str:
            return resp.status.state.value if resp.status and resp.status.state else "?"

        # A cold warehouse can leave the statement PENDING/RUNNING past the 50s
        # server-side wait cap. Poll to a hard deadline rather than failing — a
        # warehouse cold start must not be scored as a missing deliverable. The
        # bound is an iteration count (not wall time) so it can't stall when the
        # poll interval is 0.
        poll = max(self._POLL_SECS, 0)
        max_iters = int(self._POLL_DEADLINE_SECS / poll) if poll > 0 else 80
        for _ in range(max(1, max_iters)):
            if _state(r) not in ("PENDING", "RUNNING"):
                break
            time.sleep(poll)
            r = self._w.statement_execution.get_statement(r.statement_id)

        state = _state(r)
        if state != "SUCCEEDED":
            err = getattr(r.status, "error", None)
            raise RuntimeError(f"statement {state}: {getattr(err, 'message', '')}")
        schema_cols = list(r.manifest.schema.columns)
        cols = [c.name for c in schema_cols]
        rows = r.result.data_array if r.result and r.result.data_array else []
        df = pd.DataFrame(rows, columns=cols)
        # The statement API returns every value as a STRING. Cast numeric columns
        # back to numbers using the result's own declared types so the grader's
        # canonicalize() sees real int/float dtypes (otherwise an undeclared float
        # column stays a string and is compared by exact string repr — fragile).
        # Databricks' column type enum names 64-bit ints LONG (not BIGINT), and
        # also uses BYTE/SHORT for narrower ints — match all numeric kinds.
        numeric_kinds = ("INT", "LONG", "SHORT", "BYTE", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC")
        for c in schema_cols:
            tn = getattr(c, "type_name", None)
            t = str(getattr(tn, "value", tn) or getattr(c, "type_text", "") or "").upper()
            if any(k in t for k in numeric_kinds):
                df[c.name] = pd.to_numeric(df[c.name], errors="coerce")
        return df

    def _resolve_fqn(self, name: str) -> str | None:
        """Resolve a bare feature-table name to its fully-qualified
        `<catalog>.<schema>.<name>`. Tasks name a table but never dictate a
        catalog or schema, so a model may land it in any schema of the
        `workspace` catalog (the literal `default`, a self-named `feature_store`)
        or even a catalog it created itself. Search the metastore-wide
        information_schema and prefer this run's schema, then the conventional
        `workspace.default`, then any `workspace.*`, then any non-system
        location. Cached per name; returns None if the name is nowhere in the
        metastore."""
        if name in self._fqn_cache:
            return self._fqn_cache[name]
        self._fqn_cache[name] = fqn = self._search_table(name)
        return fqn

    # Catalogs/schemas that are platform plumbing, never an agent deliverable.
    _SKIP_CATALOGS = {"system", "samples", "__databricks_internal", "hive_metastore"}
    _SKIP_SCHEMAS = {"information_schema"}

    def _search_table(self, name: str) -> str | None:
        # system.information_schema spans the whole metastore (all catalogs);
        # fall back to the workspace catalog's own information_schema if the
        # system catalog is not granted, then give up (caller probes the
        # conventional location).
        for src in ("system.information_schema.tables", f"{CATALOG}.information_schema.tables"):
            try:
                df = self._sql(
                    "SELECT table_catalog, table_schema FROM "
                    f"{src} WHERE table_name = '{name}'"
                )
            except Exception:
                continue
            cands = [
                (str(c), str(s))
                for c, s in zip(
                    df["table_catalog"].tolist() if not df.empty else [],
                    df["table_schema"].tolist() if not df.empty else [],
                )
                if str(c) not in self._SKIP_CATALOGS and str(s) not in self._SKIP_SCHEMAS
            ]
            if self._per_run:
                # Only this run's schema (or the shared default landing zone) —
                # never a leftover agent catalog (e.g. fs_online) from another run.
                allowed = {(self._catalog, self._schema_name), (CATALOG, DEFAULT_SCHEMA)}
                cands = [cs for cs in cands if cs in allowed]
            if not cands:
                continue

            def rank(cs):
                cat, sch = cs
                return (
                    0 if (cat == self._catalog and sch == self._schema_name) else
                    1 if (cat == CATALOG and sch == DEFAULT_SCHEMA) else
                    2 if cat == CATALOG else
                    3 if sch == DEFAULT_SCHEMA else 4,
                    cat,
                    sch,
                )

            cat, sch = sorted(cands, key=rank)[0]
            return f"{cat}.{sch}.{name}"
        return None

    # -- feature tables ----------------------------------------------------

    def get_feature_table(self, name: str, version: int | None = None) -> TableInfo | None:
        full = self._resolve_fqn(name) or f"{self._schema}.{name}"
        try:
            t = self._w.tables.get(full)
        except Exception:
            return None
        pk: list[str] = []
        for c in getattr(t, "table_constraints", None) or []:
            pkc = getattr(c, "primary_key_constraint", None)
            if pkc is not None:
                pk = list(pkc.child_columns or [])
        return TableInfo(
            name=name,
            version=None,
            primary_key=pk,
            event_time=None,  # Unity Catalog has no event-time concept → capability note
            schema={
                c.name: str(c.type_name.value if c.type_name else c.type_text)
                for c in (t.columns or [])
            },
        )

    def read_rows(self, name: str, version: int | None = None) -> pd.DataFrame:
        # Databricks has no native feature-table version: a new version may
        # reuse the bare name (CREATE OR REPLACE) or encode the version in the
        # name (X_2, X_v2). When a version is requested, prefer the suffixed
        # names — mirroring read_training_dataset — so "version N is the
        # deliverable" resolves to the explicitly versioned table, then fall
        # back to the bare name (the single-table realization).
        candidates = (
            [f"{name}_v{version}", f"{name}_{version}", name]
            if version is not None
            else [name]
        )
        fqn = None
        for cand in candidates:
            fqn = self._resolve_fqn(cand)
            if fqn is not None:
                break
        fqn = fqn or f"{self._schema}.{name}"
        try:
            return self._sql(f"SELECT * FROM {fqn}")
        except RuntimeError as e:
            # A missing deliverable is a graceful "not produced" (failed
            # A0_deliverable_exists assert), NOT a grader crash: translate it to
            # LookupError, which grade_table_main catches. Anything else is a
            # genuine read failure and must keep propagating.
            msg = str(e)
            if "TABLE_OR_VIEW_NOT_FOUND" in msg or "cannot be found" in msg:
                raise LookupError(f"table {name} not found in any {CATALOG} schema") from e
            raise

    # -- online reads --------------------------------------------------------

    def _online_pk(self, name: str) -> str:
        """Primary-key column for the feature table (falls back to account_id,
        the online task's key)."""
        info = self.get_feature_table(name)
        if info is not None and info.primary_key:
            return info.primary_key[0]
        return "account_id"

    def _feature_serving_candidates(self, name: str) -> list[str]:
        """Endpoint names that might serve feature table `name` online.

        Combines conventional names with a scan of the workspace: a feature
        serving endpoint serves a FeatureSpec (a custom served entity), so we
        keep only endpoints whose served entities are NOT foundation/external
        models (that excludes the workspace's hosted LLM/embedding endpoints)
        and whose endpoint or entity name references the table.
        """
        cands: list[str] = [
            name,
            f"{name}_serving",
            f"{name}-serving",
            f"{name}_endpoint",
            f"fs_{name}",
            f"feature_{name}",
        ]
        try:
            for e in self._w.serving_endpoints.list():
                try:
                    d = self._w.serving_endpoints.get(e.name)
                    ents = (getattr(d, "config", None) and d.config.served_entities) or []
                    is_model = any(
                        getattr(x, "foundation_model", None) or getattr(x, "external_model", None)
                        for x in ents
                    )
                    if is_model:
                        continue  # hosted LLM/embedding endpoint — never a feature store
                    entity_names = [str(getattr(x, "entity_name", "") or "") for x in ents]
                    if self._per_run:
                        # Per-run: require a served entity in THIS run's schema, so
                        # a leftover endpoint serving another run's table (whose
                        # name happens to be a substring) can't be matched.
                        matched = any(f"{self._schema}." in en for en in entity_names)
                    else:
                        matched = any(name in r for r in [e.name] + entity_names)
                    if matched:
                        cands.append(e.name)
                except Exception:
                    continue
        except Exception:
            pass
        # de-dup, preserve order
        seen: set = set()
        return [c for c in cands if not (c in seen or seen.add(c))]

    def _parse_predictions(self, resp, pk: str, record_ids: list[str]) -> pd.DataFrame:
        """Normalize a feature-serving query response into rows keyed by pk."""
        preds = (
            getattr(resp, "predictions", None)
            or getattr(resp, "outputs", None)
            or getattr(resp, "data", None)
            or []
        )
        if isinstance(preds, dict):
            preds = [preds]
        df = pd.DataFrame(list(preds))
        if df.empty:
            return df
        if pk not in df.columns and len(df) == len(record_ids):
            df.insert(0, pk, [str(r) for r in record_ids])  # endpoint returned no key col
        if pk in df.columns and pk != "account_id":
            df = df.rename(columns={pk: "account_id"})
        return df

    def get_records(
        self, name: str, record_ids: list[str], version: int | None = 1
    ) -> pd.DataFrame:
        """Independent ONLINE read for known keys via a Databricks feature
        serving endpoint (`serving_endpoints.query` — no extra dependency).

        Mirrors the SageMaker/Hopsworks `get_records` contract: a DataFrame with
        the primary-key column (`account_id`) plus the feature columns, one row
        per found key. We do NOT read the offline Unity Catalog table here — A3
        must prove ONLINE materialization, and the UC table is what A1/read_rows
        already cover. If no feature serving endpoint serves the table, the
        result is empty and the suite reports the keys as a mismatch.

        NOTE: the Lakebase/synced-table (PostgreSQL) read path is intentionally
        not used — it would add a Postgres driver dependency. Feature serving is
        the dependency-free online read.
        """
        pk = self._online_pk(name)
        records = [{pk: str(r)} for r in record_ids]
        for ep in self._feature_serving_candidates(name):
            try:
                resp = self._w.serving_endpoints.query(name=ep, dataframe_records=records)
            except Exception:
                continue  # wrong endpoint / not servable this way → try the next
            df = self._parse_predictions(resp, pk, record_ids)
            if not df.empty:
                return df
        return pd.DataFrame()

    # -- training datasets ---------------------------------------------------

    def read_training_dataset(self, name: str, version: int = 1) -> pd.DataFrame:
        # A "versioned training dataset named X" has no canonical table name on
        # Databricks (no training-dataset primitive), so accept the common
        # realizations. Schema/catalog are resolved by get_feature_table, so we
        # only vary the table name here.
        candidates = (
            f"{name}_v{version}",
            name,
            f"{name}_training",
            f"{name}_training_v{version}",
            f"{name}_td",
            f"{name}_td_v{version}",
        )
        for table in candidates:
            if self.get_feature_table(table) is not None:
                return self.read_rows(table)
        raise LookupError(
            f"training dataset {name!r} v{version} not found — tried "
            f"{', '.join(candidates)} across all non-system metastore schemas"
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
        print(
            json.dumps(
                {"exists": info is not None, **(info.__dict__ if info else {"name": args.name})},
                default=str,
                indent=2,
            )
        )
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
    def _model_fqn_candidates(self, name: str) -> list[str]:
        """Fully-qualified registered-model names to try, conventional location
        first. Tasks pin a model NAME but not a schema/catalog, so — like
        feature tables — a model may be registered in any non-system schema.
        Probe this run's schema, then every other non-system schema of every
        non-system catalog."""
        out = [f"{self._schema}.{name}"]
        try:
            for cat in self._w.catalogs.list():
                cn = getattr(cat, "name", None)
                if not cn or cn in self._SKIP_CATALOGS:
                    continue
                for sch in self._w.schemas.list(catalog_name=cn):
                    sn = getattr(sch, "name", None)
                    if not sn or sn in self._SKIP_SCHEMAS:
                        continue
                    fqn = f"{cn}.{sn}.{name}"
                    if fqn not in out:
                        out.append(fqn)
        except Exception:
            pass
        return out

    def _run_metrics(self, run_id: str | None) -> dict:
        """Training metrics from the MLflow run that produced a model version.
        UC's registry surface carries no metrics, but the linked run does, so
        the registered-model deliverable can still be checked for "with its
        metrics". Best-effort: any failure (no run, run gone, no access)
        degrades to {}, never crashes the grader."""
        if not run_id:
            return {}
        try:
            r = self._w.api_client.do(
                "GET", "/api/2.0/mlflow/runs/get", query={"run_id": run_id}
            )
        except Exception:
            return {}
        data = (r.get("run") or {}).get("data") or {}
        return {
            m["key"]: m.get("value")
            for m in (data.get("metrics") or [])
            if m.get("key") is not None
        }

    def _uc_model_metrics(self, full_name: str, version_num: int | None) -> dict:
        """Metrics of the named UC model version (latest if version_num is
        unknown), read from its source MLflow run."""
        try:
            if version_num:
                mv = self._w.model_versions.get(full_name, version_num)
            else:
                versions = list(self._w.model_versions.list(full_name))
                if not versions:
                    return {}
                mv = max(versions, key=lambda v: int(getattr(v, "version", 0) or 0))
        except Exception:
            return {}
        return self._run_metrics(getattr(mv, "run_id", None))

    def get_model(self, name: str, version: int = 1) -> dict:
        """Registered model: the UC registry (searched across non-system
        schemas, conventional workspace.default first) then the legacy
        workspace (MLflow) registry. `metrics` is pulled from the MLflow run
        backing the model version (UC's registry surface has none of its own)."""
        for fqn in self._model_fqn_candidates(name):
            try:
                m = self._w.registered_models.get(fqn)
            except Exception:
                continue
            aliases = getattr(m, "aliases", None) or []
            ver = max((int(getattr(a, "version_num", 0) or 0) for a in aliases), default=None)
            return {
                "exists": True,
                "registry": "unity-catalog",
                "full_name": m.full_name,
                "version": ver,
                "metrics": self._uc_model_metrics(m.full_name, ver),
            }
        try:
            r = self._w.api_client.do(
                "GET", "/api/2.0/mlflow/registered-models/get", query={"name": name}
            )
            versions = (r.get("registered_model") or {}).get("latest_versions") or []
            ver = max((int(v.get("version", 0)) for v in versions), default=None)
            run_id = next(
                (v.get("run_id") for v in versions if int(v.get("version", 0)) == ver),
                None,
            )
            return {
                "exists": True,
                "registry": "workspace-mlflow",
                "version": ver,
                "metrics": self._run_metrics(run_id),
            }
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def get_job(self, name: str) -> dict:
        """Jobs API: {exists, scheduled, last_run_state, notifications}."""
        try:
            jobs = list(self._w.jobs.list(name=name))
            if not jobs:
                # The exact-name filter missed — the model may have cased or
                # affixed the name. Fall back to a case-insensitive exact, then
                # substring, match over all jobs.
                low = name.lower()

                def _jname(j):
                    return str(getattr(getattr(j, "settings", None), "name", "") or "")

                allj = list(self._w.jobs.list())
                jobs = (
                    [j for j in allj if _jname(j).lower() == low]
                    or [j for j in allj if low in _jname(j).lower()]
                )
            if not jobs:
                return {"exists": False}
            j = self._w.jobs.get(jobs[0].job_id)
            s = j.settings
            out = {
                "exists": True,
                "scheduled": bool(
                    getattr(s, "schedule", None)
                    or getattr(s, "trigger", None)
                    or getattr(s, "continuous", None)
                ),
                "notifications": bool(
                    getattr(s, "email_notifications", None)
                    or getattr(s, "webhook_notifications", None)
                ),
            }
            try:
                runs = list(self._w.jobs.list_runs(job_id=jobs[0].job_id, limit=1))
                if runs and runs[0].state:
                    out["last_run_state"] = str(
                        runs[0].state.result_state or runs[0].state.life_cycle_state
                    )
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
            return {"exists": True, "status": str(getattr(state, "ready", "") or "")}
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def get_alert(self, name_or_hint: str) -> dict:
        """Alerting realization on Databricks = job notification settings;
        report whether the named JOB carries any (see get_job)."""
        job = get_job(self, name_or_hint)
        exists = bool(job.get("exists") and job.get("notifications"))
        return {
            "exists": exists,
            "count": 1 if exists else 0,  # parity with the {exists, count} contract
            **({"error": job["error"]} if job.get("error") else {}),
        }

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
                    r = (
                        self._w.api_client.do(
                            "GET", "/api/2.0/vector-search/indexes", query={"endpoint_name": ep}
                        )
                        or {}
                    )
                except Exception:
                    continue
                hits += [
                    {"index": idx.get("name"), "endpoint": ep}
                    for idx in (r.get("vector_indexes") or [])
                    if name in (idx.get("name") or "")
                ]
            return {
                "exists": bool(hits),
                "kind": "vector-search-index",
                "matches": hits,
                "endpoints": ep_names,
            }
        except Exception as e:
            return {"exists": False, "error": str(e)}

    cls._model_fqn_candidates = _model_fqn_candidates
    cls._run_metrics = _run_metrics
    cls._uc_model_metrics = _uc_model_metrics
    cls.get_model = get_model
    cls.get_job = get_job
    cls.get_endpoint = get_endpoint
    cls.get_alert = get_alert
    cls.get_vector_store = get_vector_store
    return cls


_state_reads(DatabricksChecker)
