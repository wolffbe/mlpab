"""Gemini Enterprise Agent Platform (GCP) checker adapter — grader-side reads through the project.

Gemini Enterprise Agent Platform's feature store is BigQuery-backed: a Feature Group is defined over a
BigQuery table, so feature/training-dataset reads go straight through the
BigQuery client (`SELECT * FROM <project>.<dataset>.<name>`) — the same path a
the platform consumer uses. Model/job/endpoint/vector-search reads go through the
aiplatform SDK. Native Vector Search (Matching Engine) means get_vector_store
reports native_ann=True (unlike SageMaker/Azure).

Auth: a service-account key at GOOGLE_APPLICATION_CREDENTIALS (standard ADC);
GCP_PROJECT / GCP_LOCATION / GCP_BQ_DATASET scope the reads. SDKs imported
lazily so this module imports without google-cloud-* installed.

NOTE: live-validate against a real project before trusting A3–A5 — the BigQuery
table-naming and feature-group→BQ mapping are convention-based here.

CLI (for live probing):
    python -m evals.adapters.gcp describe-fg --name transactions
    python -m evals.adapters.gcp read-fg --name transactions --out /tmp/fg.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from evals.adapters import TableInfo


class VertexChecker:
    def __init__(self) -> None:
        self.project = os.environ["GCP_PROJECT"]
        self.location = os.environ.get("GCP_LOCATION", "us-central1")
        self.dataset = os.environ.get("GCP_BQ_DATASET", "mlpab")

    # -- BigQuery-backed feature reads -------------------------------------
    def _bq(self):
        from google.cloud import bigquery  # lazy
        return bigquery.Client(project=self.project)

    def _table_ref(self, name: str) -> str:
        return f"{self.project}.{self.dataset}.{name}"

    def get_feature_table(self, name: str, version: int | None = None) -> TableInfo | None:
        from google.cloud import bigquery
        bq = self._bq()
        try:
            t = bq.get_table(self._table_ref(name))
        except Exception:
            return None
        schema = {f.name: f.field_type for f in t.schema}
        # the platform feature groups carry an entity-id column + a feature-timestamp;
        # surface them if present (convention: entity_id / feature_timestamp).
        pk = [c for c in ("entity_id", "entity_id_column") if c in schema]
        event = next((c for c in ("feature_timestamp", "event_timestamp") if c in schema), None)
        return TableInfo(name=name, version=version, primary_key=pk,
                         event_time=event, schema=schema)

    def read_rows(self, name: str, version: int | None = None) -> pd.DataFrame:
        return self._bq().query(f"SELECT * FROM `{self._table_ref(name)}`").to_dataframe()

    def read_training_dataset(self, name: str, version: int = 1) -> pd.DataFrame:
        for table in (f"{name}_v{version}", name):
            if self.get_feature_table(table) is not None:
                return self.read_rows(table)
        raise LookupError(
            f"training dataset {name!r} v{version} not found — expected BigQuery "
            f"table {self._table_ref(name + f'_v{version}')} (or {self._table_ref(name)})")


def _state_reads(cls):
    def _init_ai(self):
        from google.cloud import aiplatform
        aiplatform.init(project=self.project, location=self.location)
        return aiplatform

    def get_model(self, name: str, version: int = 1) -> dict:
        """the platform model registry: match by display_name; metrics from the latest
        model evaluation if one was uploaded."""
        try:
            ai = _init_ai(self)
            models = list(ai.Model.list(filter=f'display_name="{name}"'))
            if not models:
                return {"exists": False}
            m = models[0]
            metrics = None
            try:
                evals = list(m.list_model_evaluations())
                if evals:
                    metrics = dict(evals[0].metrics) if evals[0].metrics else None
            except Exception:
                pass
            return {"exists": True, "resource_name": m.resource_name,
                    "version": getattr(m, "version_id", None), "metrics": metrics}
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def get_job(self, name: str) -> dict:
        """Custom / pipeline jobs by display_name; scheduled = a Schedule exists."""
        try:
            ai = _init_ai(self)
            hits = []
            for lister in (ai.CustomJob, ai.PipelineJob):
                try:
                    hits += [j for j in lister.list(filter=f'display_name="{name}"')]
                except Exception:
                    pass
            if not hits:
                return {"exists": False}
            j = hits[0]
            scheduled = False
            try:
                scheduled = any(name in (s.display_name or "")
                                for s in ai.PipelineJobSchedule.list())
            except Exception:
                pass
            return {"exists": True, "last_run_state": str(getattr(j, "state", "")),
                    "scheduled": scheduled, "kind": type(j).__name__}
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def get_endpoint(self, name: str) -> dict:
        try:
            ai = _init_ai(self)
            eps = list(ai.Endpoint.list(filter=f'display_name="{name}"'))
            if not eps:
                return {"exists": False}
            e = eps[0]
            return {"exists": True,
                    "status": "deployed" if e.list_models() else "no-model"}
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def get_alert(self, name_or_hint: str) -> dict:
        """Alerting on GCP = Cloud Monitoring alert policy whose name matches."""
        try:
            from google.cloud import monitoring_v3
            client = monitoring_v3.AlertPolicyServiceClient()
            project = f"projects/{self.project}"
            hits = [p.display_name for p in client.list_alert_policies(name=project)
                    if name_or_hint in (p.display_name or "")]
            return {"exists": bool(hits), "count": len(hits), "matches": hits}
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def get_vector_store(self, name: str) -> dict:
        """Native Gemini Enterprise Agent Platform Vector Search (Matching Engine): match an index by
        display_name. native_ann=True — the platform has first-class ANN."""
        try:
            ai = _init_ai(self)
            idxs = [i for i in ai.MatchingEngineIndex.list()
                    if name in (i.display_name or "")]
            return {"exists": bool(idxs), "kind": "vertex-vector-search",
                    "native_ann": True,
                    "matches": [i.display_name for i in idxs]}
        except Exception as e:
            return {"exists": False, "error": str(e)}

    cls.get_model = get_model
    cls.get_job = get_job
    cls.get_endpoint = get_endpoint
    cls.get_alert = get_alert
    cls.get_vector_store = get_vector_store
    return cls


_state_reads(VertexChecker)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("describe-fg"); p.add_argument("--name", required=True)
    p = sub.add_parser("read-fg")
    p.add_argument("--name", required=True); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("read-td")
    p.add_argument("--name", required=True); p.add_argument("--version", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    ck = VertexChecker()
    if args.cmd == "describe-fg":
        info = ck.get_feature_table(args.name)
        print(json.dumps({"exists": info is not None,
                          **(info.__dict__ if info else {"name": args.name})},
                         default=str, indent=2))
        return 0 if info else 1
    if args.cmd == "read-fg":
        ck.read_rows(args.name).to_csv(args.out, index=False)
    else:
        ck.read_training_dataset(args.name, args.version).to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
