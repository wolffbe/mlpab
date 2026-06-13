"""Gemini Enterprise Agent Platform (GCP) platform teardown — sweep agent-created resources.

Run by `mlpab` at the START and END of every run. Best-effort, NEVER raises.
Deletes in cost order: endpoints (undeploy first — a deployed model bills a node
continuously) → vector-search index endpoints + indexes → batch/custom/pipeline
jobs → registered models → feature groups → the BigQuery dataset's TABLES
(preserving the dataset itself, which setup.py guarantees).

⚠ LIVE-VALIDATION REQUIRED before any real run — missed the platform endpoints / index
endpoints bill continuously.
"""
from __future__ import annotations

import importlib
import os


def _resolve(*candidates):
    """First importable `module:attr` among candidates, else None. The Vertex
    Feature Store classes moved across SDK versions (top-level `aiplatform`
    vs `vertexai.resources.preview.feature_store`); try each known location so
    the sweep doesn't silently no-op on a class that isn't where we guessed."""
    for modname, attr in candidates:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        cls = getattr(mod, attr, None)
        if cls is not None:
            return cls
    return None


def _sweep(label, lister, deleter):
    try:
        items = list(lister())
    except Exception as e:
        print(f"[gcp teardown] list {label} skipped: {e}")
        return
    for it in items:
        try:
            deleter(it)
            print(f"[gcp teardown] deleted {label} {getattr(it, 'display_name', it)!r}")
        except Exception as e:
            print(f"[gcp teardown] delete {label} skipped: {e}")


def main() -> None:
    project = os.environ.get("GCP_PROJECT")
    location = os.environ.get("GCP_LOCATION", "us-central1")
    dataset = os.environ.get("GCP_BQ_DATASET", "mlpab")
    if not project:
        print("[gcp teardown] missing GCP_PROJECT — skipping")
        return
    try:
        from google.cloud import aiplatform
        aiplatform.init(project=project, location=location)
    except Exception as e:
        print(f"[gcp teardown] aiplatform init skipped: {e}")
        aiplatform = None

    if aiplatform is not None:
        def _undeploy_delete(ep):
            try:
                ep.undeploy_all()
            except Exception:
                pass
            ep.delete(force=True)
        _sweep("endpoint", aiplatform.Endpoint.list, _undeploy_delete)
        _sweep("index-endpoint", aiplatform.MatchingEngineIndexEndpoint.list,
               lambda e: _undeploy_delete(e))
        _sweep("vector-index", aiplatform.MatchingEngineIndex.list, lambda i: i.delete())
        # Feature Online Stores are Bigtable/Optimized-backed and BILL CONTINUOUSLY;
        # delete them (force removes their feature views) + the feature groups.
        # These classes are NOT top-level aiplatform attributes — they live under
        # vertexai.resources.preview.feature_store — so resolve them explicitly,
        # and shout if neither location has them (otherwise a billing leak is
        # invisible).
        def _force_del(r):
            try:
                r.delete(force=True)
            except Exception:
                r.delete()
        for label, attr in (("feature-online-store", "FeatureOnlineStore"),
                            ("feature-group", "FeatureGroup")):
            cls = _resolve(
                ("vertexai.resources.preview.feature_store", attr),
                ("vertexai.resources.preview", attr),
                ("google.cloud.aiplatform", attr),
            )
            if cls is not None and hasattr(cls, "list"):
                _sweep(label, cls.list, _force_del)
            else:
                print(f"[gcp teardown] WARNING: could not resolve {attr}; "
                      f"{label}s NOT swept and may bill continuously — verify SDK path")
        for lister in (getattr(aiplatform, "CustomJob", None),
                       getattr(aiplatform, "PipelineJob", None),
                       getattr(aiplatform, "BatchPredictionJob", None)):
            if lister is not None:
                _sweep(lister.__name__, lister.list,
                       lambda j: j.delete() if str(getattr(j, "state", "")).endswith(
                           ("SUCCEEDED", "FAILED", "CANCELLED")) else None)
        _sweep("model", aiplatform.Model.list, lambda m: m.delete())

    # BigQuery: drop the dataset's TABLES (feature groups + training datasets),
    # keep the dataset (plumbing).
    try:
        from google.cloud import bigquery
        bq = bigquery.Client(project=project)
        for t in bq.list_tables(f"{project}.{dataset}"):
            try:
                bq.delete_table(t.reference, not_found_ok=True)
                print(f"[gcp teardown] dropped table {t.table_id}")
            except Exception as e:
                print(f"[gcp teardown] drop table {t.table_id} skipped: {e}")
    except Exception as e:
        print(f"[gcp teardown] BigQuery sweep skipped: {e}")


if __name__ == "__main__":
    main()
