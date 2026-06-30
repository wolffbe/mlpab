"""Gemini Enterprise Agent Platform (GCP) platform teardown — sweep agent-created resources.

Run by `mlpab` at the START and END of every run. Best-effort, NEVER raises.
Deletes in cost order: endpoints (undeploy first — a deployed model bills a node
continuously) → vector-search index endpoints + indexes → batch/custom/pipeline
jobs → registered models → feature groups → the run's BigQuery tables/dataset.

Two modes, mirroring the databricks teardown:

  * PER-RUN (default, when MLPAB_GCP_PREFIX is set): delete only THIS run's
    resources, so two runs can share one project without their start/end
    teardowns deleting each other's work. The runner mints a per-run id that is
    a NAME PREFIX (`mlpab<hex>_`) on every aiplatform/monitoring resource's
    display_name AND names this run's BigQuery dataset (GCP_BQ_DATASET=
    mlpab_<hex>, this run's feature-table landing zone). aiplatform resources
    are selected by display_name prefix; the per-run dataset is dropped whole
    (delete_contents) rather than emptied table-by-table.

  * FULL SWEEP (`teardown.py --all`, or when no prefix is set): the original
    account-wide behaviour — delete EVERY resource of the swept types and empty
    (but preserve) the shared GCP_BQ_DATASET, as a between-batch janitor.

⚠ LIVE-VALIDATION REQUIRED before any real run — missed the platform endpoints / index
endpoints bill continuously.
"""

from __future__ import annotations

import importlib
import os

# Per-run identity (set by src/mlpab/runner.py). Empty on a manual invocation →
# full-sweep mode. _RUN_MODE is finalised in main()/verify() (PREFIX set and not
# --all). The aiplatform display_name the agent is told to use is
# `<PREFIX>_<name>`, so a startswith(PREFIX) test selects this run's resources.
PREFIX = os.environ.get("MLPAB_GCP_PREFIX") or ""
_RUN_MODE = False


def _scoped(obj) -> bool:
    """In per-run mode keep only resources whose display_name carries this run's
    prefix; in full-sweep mode keep everything."""
    if not _RUN_MODE:
        return True
    name = getattr(obj, "display_name", None) or ""
    return bool(name) and str(name).startswith(PREFIX)


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
        if not _scoped(it):
            continue
        try:
            deleter(it)
            print(f"[gcp teardown] deleted {label} {getattr(it, 'display_name', it)!r}")
        except Exception as e:
            print(f"[gcp teardown] delete {label} skipped: {e}")


def main(all_mode: bool = False) -> None:
    global _RUN_MODE
    _RUN_MODE = bool(PREFIX) and not all_mode
    project = os.environ.get("GCP_PROJECT")
    location = os.environ.get("GCP_LOCATION", "us-central1")
    dataset = os.environ.get("GCP_BQ_DATASET", "mlpab")
    if not project:
        print("[gcp teardown] missing GCP_PROJECT — skipping")
        return
    print(
        f"[gcp teardown] per-run mode: prefix {PREFIX!r}, dataset {dataset!r}"
        if _RUN_MODE
        else f"[gcp teardown] full sweep ({'forced (--all)' if all_mode else 'no run prefix set'})"
    )
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
        _sweep(
            "index-endpoint",
            aiplatform.MatchingEngineIndexEndpoint.list,
            lambda e: _undeploy_delete(e),
        )
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

        for label, attr in (
            ("feature-online-store", "FeatureOnlineStore"),
            ("feature-group", "FeatureGroup"),
        ):
            cls = _resolve(
                ("vertexai.resources.preview.feature_store", attr),
                ("vertexai.resources.preview", attr),
                ("google.cloud.aiplatform", attr),
            )
            if cls is not None and hasattr(cls, "list"):
                _sweep(label, cls.list, _force_del)
            else:
                print(
                    f"[gcp teardown] WARNING: could not resolve {attr}; "
                    f"{label}s NOT swept and may bill continuously — verify SDK path"
                )
        for lister in (
            getattr(aiplatform, "CustomJob", None),
            getattr(aiplatform, "PipelineJob", None),
            getattr(aiplatform, "BatchPredictionJob", None),
        ):
            if lister is not None:
                _sweep(
                    lister.__name__,
                    lister.list,
                    lambda j: (
                        j.delete()
                        if str(getattr(j, "state", "")).endswith(
                            ("SUCCEEDED", "FAILED", "CANCELLED")
                        )
                        else None
                    ),
                )
        _sweep("model", aiplatform.Model.list, lambda m: m.delete())

    # BigQuery: in per-run mode drop the run's whole dataset (delete_contents
    # cascades its feature/training tables); in full-sweep mode keep the old
    # behaviour — empty the shared dataset's TABLES but preserve the dataset
    # itself (plumbing the next run relies on, guaranteed by setup.py).
    try:
        from google.cloud import bigquery

        bq = bigquery.Client(project=project)
        # Drop the dataset WHOLE only when it is genuinely a per-run dataset
        # (the runner names it `mlpab_<run>`). A per-run-mode invocation that
        # still points at the SHARED base dataset (e.g. a manual prefix-only run)
        # must NOT nuke it — fall back to emptying its tables.
        if _RUN_MODE and dataset.startswith("mlpab_"):
            bq.delete_dataset(
                f"{project}.{dataset}", delete_contents=True, not_found_ok=True
            )
            print(f"[gcp teardown] dropped per-run dataset {dataset}")
        else:
            for t in bq.list_tables(f"{project}.{dataset}"):
                try:
                    bq.delete_table(t.reference, not_found_ok=True)
                    print(f"[gcp teardown] dropped table {t.table_id}")
                except Exception as e:
                    print(f"[gcp teardown] drop table {t.table_id} skipped: {e}")
    except Exception as e:
        print(f"[gcp teardown] BigQuery sweep skipped: {e}")


def verify(all_mode: bool = False) -> int:
    """Twofold teardown check (`teardown.py verify`): (1) Vertex AI CONNECTS,
    then (2) no billed Vertex ENDPOINTS survive (a deployed endpoint bills
    continuously). In per-run mode only THIS run's prefixed endpoints count as a
    leak; with --all (or no prefix) any surviving endpoint does. Exit non-zero on
    no-connection or a leak; the runner WARNS on a leak. Read-only, best-effort."""
    global _RUN_MODE
    _RUN_MODE = bool(PREFIX) and not all_mode
    project = os.environ.get("GCP_PROJECT")
    location = os.environ.get("GCP_LOCATION", "us-central1")
    if not project:
        print("[gcp verify-teardown] missing GCP_PROJECT")
        return 1
    try:
        from google.cloud import aiplatform

        aiplatform.init(project=project, location=location)
        eps = [
            getattr(e, "display_name", None) or getattr(e, "name", "?")
            for e in aiplatform.Endpoint.list()
            if _scoped(e)
        ]
    except Exception as e:
        print(f"[gcp verify-teardown] NO CONNECTION: {e}")
        return 1
    if eps:
        print("[gcp verify-teardown] LEAKS: endpoints " + ", ".join(map(str, eps)))
        return 1
    print("[gcp verify-teardown] OK: connected; no endpoints remain")
    return 0


if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]
    all_mode = "--all" in argv
    do_verify = "verify" in argv
    sys.exit(verify(all_mode) if do_verify else (main(all_mode) or 0))
