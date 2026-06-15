"""Hopsworks checker adapter — grader-side reads through the cluster.

Uses the Hopsworks SDK (`hopsworks.login()` with HOPSWORKS_API_KEY /
HOPSWORKS_HOST from the env) for metadata + data reads. TRUST RULE: run this
with a python whose `hopsworks` package was installed from the COMMITTED base
wheel (build/hopsworks/sdk/hopsworks-0-py3-none-any.whl, built from the pinned
ref) — never from an engineer-modifiable copy. The base testbed .venv stays
free of interface packages, so the natural home is a dedicated grader venv
(same pattern as the preflight check venv).

The SDK is imported lazily so this module can be imported (e.g. by tests, or
for the Checker protocol) without hopsworks installed.

CLI (for live probing):
    python -m evals.adapters.hopsworks describe-fg --name transactions
    python -m evals.adapters.hopsworks read-fg     --name transactions --out /tmp/fg.csv
    python -m evals.adapters.hopsworks read-td     --name churn_training --version 1 --out /tmp/td.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from evals.adapters import TableInfo


class HopsworksChecker:
    """Checker over the run's Hopsworks project. The runner exports
    HOPSWORKS_PROJECT (the name setup.py created), so we log in to THAT project
    by name — essential once parallel runs leave more than one project on the
    cluster, where a bare login() would prompt or attach to the wrong one. When
    unset (standalone use against a single-project key), login() falls back to
    auto-selecting the only project."""

    def __init__(self) -> None:
        import hopsworks  # lazy: see module docstring

        self._project = hopsworks.login(project=os.environ.get("HOPSWORKS_PROJECT"))
        self._fs = self._project.get_feature_store()

    # -- feature tables ----------------------------------------------------

    def _get_fg(self, name: str, version: int | None = None):
        try:
            fg = self._fs.get_feature_group(name, version=version)
        except Exception:
            return None
        # Some client versions return None instead of raising on a miss.
        return fg

    def get_feature_table(self, name: str, version: int | None = None) -> TableInfo | None:
        fg = self._get_fg(name, version)
        if fg is None:
            return None
        return TableInfo(
            name=fg.name,
            version=getattr(fg, "version", None),
            primary_key=list(getattr(fg, "primary_key", []) or []),
            event_time=getattr(fg, "event_time", None),
            schema={f.name: str(f.type) for f in (getattr(fg, "features", None) or [])},
        )

    def read_rows(self, name: str, version: int | None = None) -> pd.DataFrame:
        fg = self._get_fg(name, version)
        if fg is None:
            raise LookupError(f"feature table {name!r} not found")
        return fg.read()

    # -- training datasets ---------------------------------------------------

    def read_training_dataset(self, name: str, version: int = 1) -> pd.DataFrame:
        """The materialized training dataset `version` of the feature view
        `name`. Reassembles (X, y) into one frame when the feature view was
        created with labels — the asserts work on the full canonical frame."""
        try:
            fv = self._fs.get_feature_view(name)
        except Exception as e:
            raise LookupError(f"feature view {name!r} not found: {e}") from e
        if fv is None:
            raise LookupError(f"feature view {name!r} not found")

        got = fv.get_training_data(training_dataset_version=version)
        # hsfs returns (X, y); y is None/empty when the view declares no labels.
        if isinstance(got, tuple):
            x, y = got[0], (got[1] if len(got) > 1 else None)
            if y is not None and len(getattr(y, "columns", [])) > 0:
                return pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
            return x
        return got


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("describe-fg", help="print TableInfo for a feature table as JSON")
    p.add_argument("--name", required=True)
    p.add_argument("--version", type=int)

    p = sub.add_parser("read-fg", help="dump a feature table to CSV")
    p.add_argument("--name", required=True)
    p.add_argument("--version", type=int)
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("read-td", help="dump a training dataset to CSV")
    p.add_argument("--name", required=True)
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)

    args = ap.parse_args(argv)
    checker = HopsworksChecker()

    if args.cmd == "describe-fg":
        info = checker.get_feature_table(args.name, args.version)
        if info is None:
            print(json.dumps({"exists": False, "name": args.name}))
            return 1
        print(json.dumps({"exists": True, **info.__dict__}, default=str, indent=2))
        return 0

    if args.cmd == "read-fg":
        checker.read_rows(args.name, args.version).to_csv(args.out, index=False)
    else:
        checker.read_training_dataset(args.name, args.version).to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# --------------------------------------------------------------------------
# Platform-state reads beyond the feature store (best-effort dicts: graders
# assert on `exists` plus whichever detail keys the platform can provide).
# --------------------------------------------------------------------------


def _state_reads(cls):
    def get_model(self, name: str, version: int = 1) -> dict:
        """Model registry entry: {exists, version, metrics}."""
        try:
            mr = self._project.get_model_registry()
            m = mr.get_model(name, version=version)
            if m is None:
                return {"exists": False}
            return {
                "exists": True,
                "version": getattr(m, "version", None),
                "metrics": dict(getattr(m, "training_metrics", None) or {}),
            }
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def get_job(self, name: str) -> dict:
        """Platform job: {exists, scheduled, last_run_state}."""
        try:
            jobs_api = self._project.get_jobs_api()
            job = jobs_api.get_job(name)
            if job is None:
                return {"exists": False}
            out = {
                "exists": True,
                "scheduled": bool(
                    getattr(job, "job_schedule", None) or getattr(job, "schedule", None)
                ),
            }
            try:
                exes = job.get_executions() or []
                if exes:
                    last = exes[0]
                    out["last_run_state"] = str(
                        getattr(last, "final_status", None) or getattr(last, "state", "")
                    )
            except Exception:
                pass
            return out
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def get_endpoint(self, name: str) -> dict:
        """Model deployment: {exists, status}."""
        try:
            ms = self._project.get_model_serving()
            dep = ms.get_deployment(name)
            if dep is None:
                return {"exists": False}
            status = ""
            try:
                status = str(dep.get_state().status)
            except Exception:
                pass
            return {"exists": True, "status": status}
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def get_alert(self, name_or_hint: str) -> dict:
        """Alert routes/triggers are project-level config in Hopsworks; report
        whether ANY job alert exists (best-effort: the alerts API surface
        varies by version)."""
        try:
            c_mod = __import__("hopsworks_common.client", fromlist=["client"])
            c = c_mod.get_instance()
            res = c._send_request("GET", ["project", self._project.id, "service", "alerts"]) or {}
            items = res.get("items") or res.get("alerts") or []
            return {"exists": bool(items), "count": len(items)}
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def get_vector_store(self, name: str) -> dict:
        """Vector store realization on Hopsworks: the embedding index lives ON
        a feature group (hsfs.embedding.EmbeddingIndex/EmbeddingFeature; CLI
        `hops fg create --embedding col:dim[:metric]`), queried via
        fg.find_neighbors / `hops fg knn`. `exists` is True only when a
        feature group `name` (ANY version) carries an embedding index
        (fg.embedding_index is not None)."""
        try:
            try:
                fgs = self._fs.get_feature_groups(name=name) or []
            except Exception:
                fgs = []
            if not fgs:
                fg = self._get_fg(name)
                fgs = [fg] if fg is not None else []
            if not fgs:
                return {"exists": False, "error": f"feature group {name!r} not found"}
            indexed = [fg for fg in fgs if getattr(fg, "embedding_index", None) is not None]
            out = {
                "exists": bool(indexed),
                "kind": "feature-group-embedding-index",
                "versions": [getattr(fg, "version", None) for fg in fgs],
            }
            if not indexed:
                out["error"] = "feature group exists but carries no embedding index"
            return out
        except Exception as e:
            return {"exists": False, "error": str(e)}

    cls.get_model = get_model
    cls.get_job = get_job
    cls.get_endpoint = get_endpoint
    cls.get_alert = get_alert
    cls.get_vector_store = get_vector_store
    return cls


_state_reads(HopsworksChecker)
