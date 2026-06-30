"""Azure ML checker adapter — grader-side reads through the workspace.

Two MLClient scopes: the WORKSPACE (models, jobs, endpoints) and the managed
FEATURE STORE (feature sets, entities). Auth is a service principal that
DefaultAzureCredential reads from AZURE_TENANT_ID / AZURE_CLIENT_ID /
AZURE_CLIENT_SECRET; AZURE_SUBSCRIPTION_ID / AZURE_RESOURCE_GROUP /
AZUREML_WORKSPACE_NAME / AZUREML_FEATURE_STORE_NAME scope the clients.

⚠ LIVE-VALIDATION REQUIRED: Azure's managed feature store materializes to an
ADLS offline store and the no-Spark read path is convention-based here
(read_rows resolves a registered Data asset's backing parquet). Verify against a
real workspace before trusting feature/training-dataset reads; the state reads
(model/job/endpoint) use stable SDK surfaces. Vector search is Azure AI Search,
which is OFF the Azure ML interface — get_vector_store reports native_ann=False
(the same asymmetry SageMaker has).

SDKs imported lazily so this module imports without azure-ai-ml installed.

CLI (for live probing):
    python -m evals.adapters.azure describe-fg --name transactions --version 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from evals.adapters import TableInfo


class AzureMLChecker:
    def __init__(self) -> None:
        from azure.ai.ml import MLClient  # lazy
        from azure.identity import DefaultAzureCredential

        self._cred = DefaultAzureCredential()
        sub = os.environ["AZURE_SUBSCRIPTION_ID"]
        rg = os.environ["AZURE_RESOURCE_GROUP"]
        self._ws = MLClient(self._cred, sub, rg, os.environ["AZUREML_WORKSPACE_NAME"])
        fs = os.environ.get("AZUREML_FEATURE_STORE_NAME")
        self._fs = MLClient(self._cred, sub, rg, fs) if fs else self._ws
        # Per-run prefix (runner sets MLPAB_AZURE_PREFIX). Azure ML has one
        # workspace/feature store (no per-run namespace), so the agent is told to
        # name every asset `<run>-<name>` and the reads below prepend the same.
        # Empty on a manual probe → bare names.
        self._prefix = os.environ.get("MLPAB_AZURE_PREFIX") or ""

    def _q(self, name: str) -> str:
        """Prefix an asset name with this run's id (no-op when unset)."""
        return f"{self._prefix}-{name}" if self._prefix else name

    # -- managed feature store -------------------------------------------------
    def get_feature_table(self, name: str, version: int | None = None) -> TableInfo | None:
        try:
            fset = self._fs.feature_sets.get(name=self._q(name), version=str(version or 1))
        except Exception:
            return None
        feats = getattr(fset, "features", None) or []
        schema = {f.name: str(getattr(f, "type", getattr(f, "data_type", ""))) for f in feats}
        spec = getattr(fset, "specification", None)
        pk = list(
            getattr(fset, "index_columns", None) or getattr(spec, "index_columns", None) or []
        )
        event = getattr(spec, "source_timestamp_column", None) or getattr(
            fset, "timestamp_column", None
        )
        return TableInfo(
            name=name, version=int(version or 1), primary_key=pk, event_time=event, schema=schema
        )

    def read_rows(self, name: str, version: int | None = None) -> pd.DataFrame:
        """Best-effort, NO-SPARK: resolve the backing parquet of the registered
        Data asset named `name` (the feature set's materialized/source data) and
        read it via the ADLS filesystem with the SP credential."""
        try:
            asset = self._ws.data.get(name=self._q(name), version=str(version or 1))
            return _read_uri(asset.path, self._cred)
        except Exception as e:
            raise LookupError(
                f"could not read feature data {name!r} v{version} on Azure ML "
                f"(no-Spark offline read is convention-based — see adapter doc): {e}"
            )

    def read_training_dataset(self, name: str, version: int = 1) -> pd.DataFrame:
        for n, v in ((f"{name}_v{version}", "1"), (name, str(version)), (name, "1")):
            try:
                asset = self._ws.data.get(name=self._q(n), version=v)
                return _read_uri(asset.path, self._cred)
            except Exception:
                continue
        raise LookupError(
            f"training dataset {name!r} v{version} not found as an Azure ML Data asset"
        )


def _read_uri(uri: str, cred) -> pd.DataFrame:
    """Read CSV/Parquet behind an abfss:// or https:// ADLS data-asset path."""
    import pyarrow.dataset as ds

    storage_options = {"anon": False}
    # adlfs understands abfss:// with a DefaultAzureCredential-derived token.
    if uri.startswith(("abfss://", "az://", "azureml://")):
        import adlfs  # noqa: F401 — registers the fsspec handler

        storage_options = {"credential": cred}
    # Don't guess from a substring: a registered data-asset folder URI carries
    # no extension (azureml://.../paths/<guid>) yet is almost always parquet, so
    # a `.csv` default would mis-read parquet bytes. Order the candidate formats
    # by any explicit extension, else try parquet first (the common backing
    # format) then csv, attempting both the arrow-dataset and pandas readers.
    low = uri.lower()
    if ".csv" in low:
        fmts = ["csv", "parquet"]
    elif ".parquet" in low:
        fmts = ["parquet", "csv"]
    else:
        fmts = ["parquet", "csv"]
    last_err: Exception | None = None
    for fmt in fmts:
        try:
            return ds.dataset(uri, format=fmt).to_table().to_pandas()
        except Exception as e:
            last_err = e
        try:
            reader = pd.read_parquet if fmt == "parquet" else pd.read_csv
            return reader(uri, storage_options=storage_options)
        except Exception as e:
            last_err = e
    raise last_err if last_err else RuntimeError(f"could not read {uri}")


def _state_reads(cls):
    def get_model(self, name: str, version: int = 1) -> dict:
        name = self._q(name)
        try:
            m = self._ws.models.get(name=name, version=str(version))
        except Exception:
            try:  # latest if the version differs
                ms = list(self._ws.models.list(name=name))
                if not ms:
                    return {"exists": False}
                m = self._ws.models.get(name=name, version=ms[0].latest_version)
            except Exception as e:
                return {"exists": False, "error": str(e)}
        props = dict(getattr(m, "properties", None) or {})
        tags = dict(getattr(m, "tags", None) or {})
        metrics = {
            k: v
            for k, v in {**props, **tags}.items()
            if any(t in k.lower() for t in ("metric", "auc", "rmse", "acc", "loss"))
        }
        return {"exists": True, "version": getattr(m, "version", None), "metrics": metrics or None}

    def get_job(self, name: str) -> dict:
        name = self._q(name)
        try:
            j = self._ws.jobs.get(name)
        except Exception:
            return {"exists": False}
        scheduled = False
        try:
            scheduled = any(name in (s.name or "") for s in self._ws.schedules.list())
        except Exception:
            pass
        return {
            "exists": True,
            "last_run_state": str(getattr(j, "status", "")),
            "scheduled": scheduled,
            "kind": getattr(j, "type", None),
        }

    def get_endpoint(self, name: str) -> dict:
        name = self._q(name)
        for getter in ("online_endpoints", "batch_endpoints"):
            try:
                e = getattr(self._ws, getter).get(name)
                return {
                    "exists": True,
                    "status": str(getattr(e, "provisioning_state", "") or ""),
                    "kind": getter,
                }
            except Exception:
                continue
        return {"exists": False}

    def get_alert(self, name_or_hint: str) -> dict:
        """Alerting on Azure = Azure Monitor metric-alert rules (azure-mgmt-monitor)
        whose name matches; best-effort (needs the mgmt SDK + Monitoring Reader)."""
        try:
            from azure.mgmt.monitor import MonitorManagementClient

            sub = os.environ["AZURE_SUBSCRIPTION_ID"]
            rg = os.environ["AZURE_RESOURCE_GROUP"]
            mon = MonitorManagementClient(self._cred, sub)
            hint = self._q(name_or_hint)
            hits = [
                r.name
                for r in mon.metric_alerts.list_by_resource_group(rg)
                if hint in (r.name or "")
            ]
            return {"exists": bool(hits), "count": len(hits), "matches": hits}
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def get_vector_store(self, name: str) -> dict:
        """Azure ML has no native vector store; Azure AI Search is a SEPARATE
        service off the Azure ML interface (asymmetry documented like SageMaker)."""
        return {
            "exists": False,
            "native_ann": False,
            "note": "Azure AI Search is off the Azure ML interface",
        }

    cls.get_model = get_model
    cls.get_job = get_job
    cls.get_endpoint = get_endpoint
    cls.get_alert = get_alert
    cls.get_vector_store = get_vector_store
    return cls


_state_reads(AzureMLChecker)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("describe-fg")
    p.add_argument("--name", required=True)
    p.add_argument("--version", type=int, default=1)
    p = sub.add_parser("read-fg")
    p.add_argument("--name", required=True)
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    ck = AzureMLChecker()
    if args.cmd == "describe-fg":
        info = ck.get_feature_table(args.name, args.version)
        print(
            json.dumps(
                {"exists": info is not None, **(info.__dict__ if info else {"name": args.name})},
                default=str,
                indent=2,
            )
        )
        return 0 if info else 1
    ck.read_rows(args.name, args.version).to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
