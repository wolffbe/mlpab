"""Azure ML platform teardown — sweep agent-created resources.

Run by `mlpab` at the START and END of every run. Best-effort, NEVER raises.
Deletes in cost order (online/batch endpoints → jobs → models → feature sets →
data assets), preserving the WORKSPACE and the FEATURE STORE themselves (those
are platform plumbing setup.py guarantees, not agent state).

Two modes, mirroring the databricks teardown:

  * PER-RUN (default, when MLPAB_AZURE_PREFIX is set): delete only THIS run's
    resources, so two runs can share one workspace without their start/end
    teardowns deleting each other's work. Azure ML has one workspace/feature
    store (no per-run namespace), so the per-run id is a NAME PREFIX
    (`mlpab<hex>-`) the agent stamps on every asset and this sweep matches on.

  * FULL SWEEP (`teardown.py --all`, or when no prefix is set): the original
    account-wide behaviour — delete EVERY asset of the swept types, as a
    between-batch janitor.

⚠ LIVE-VALIDATION REQUIRED before any real run — a teardown that misses a
resource type leaks billable Azure resources (managed online endpoints in
particular run a VM continuously).
"""

from __future__ import annotations

import os

# Per-run identity (set by src/mlpab/runner.py). Empty on a manual invocation →
# full-sweep mode. _RUN_MODE is finalised in main()/verify(). The agent is told
# to name assets `<PREFIX>-<name>`, so startswith(PREFIX) selects this run's.
PREFIX = os.environ.get("MLPAB_AZURE_PREFIX") or ""
_RUN_MODE = False


def _scoped(name) -> bool:
    """In per-run mode keep only names carrying this run's prefix; in full-sweep
    mode keep everything."""
    if not _RUN_MODE:
        return True
    return bool(name) and str(name).startswith(PREFIX)


def _ml(*, workspace: str | None = None, feature_store: str | None = None):
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    cred = DefaultAzureCredential()
    sub = os.environ["AZURE_SUBSCRIPTION_ID"]
    rg = os.environ["AZURE_RESOURCE_GROUP"]
    # A feature store is a distinct workspace KIND: feature_sets ops require the
    # client scoped via feature_store_name=, not workspace_name= (the 4th
    # positional). Passing the FS name as workspace_name silently lists nothing.
    if feature_store:
        return MLClient(cred, sub, rg, feature_store_name=feature_store)
    if workspace:
        return MLClient(cred, sub, rg, workspace)
    return MLClient(cred, sub, rg)


def _sweep(label, lister, deleter):
    try:
        items = list(lister())
    except Exception as e:
        print(f"[azure teardown] list {label} skipped: {e}")
        return
    for it in items:
        name = getattr(it, "name", None) or getattr(it, "id", None)
        if not _scoped(name):
            continue
        try:
            deleter(name)
            print(f"[azure teardown] deleted {label} {name!r}")
        except Exception as e:
            print(f"[azure teardown] delete {label} {name!r} skipped: {e}")


def main(all_mode: bool = False) -> None:
    global _RUN_MODE
    _RUN_MODE = bool(PREFIX) and not all_mode
    print(
        f"[azure teardown] per-run mode: prefix {PREFIX!r}"
        if _RUN_MODE
        else f"[azure teardown] full sweep ({'forced (--all)' if all_mode else 'no run prefix set'})"
    )
    try:
        ws = _ml(workspace=os.environ.get("AZUREML_WORKSPACE_NAME"))
    except Exception as e:
        print(f"[azure teardown] no workspace client: {e}")
        return
    # cost order: serving first (a live endpoint bills continuously)
    _sweep(
        "online-endpoint",
        lambda: ws.online_endpoints.list(),
        lambda n: ws.online_endpoints.begin_delete(n),
    )
    _sweep(
        "batch-endpoint",
        lambda: ws.batch_endpoints.list(),
        lambda n: ws.batch_endpoints.begin_delete(n),
    )
    # Compute clusters/instances bill continuously while provisioned (a training
    # job's compute target the agent created) — delete them right after serving.
    _sweep("compute", lambda: ws.compute.list(), lambda n: ws.compute.begin_delete(name=n))
    _sweep("job", lambda: ws.jobs.list(), lambda n: ws.jobs.begin_cancel(n))
    _sweep("model", lambda: ws.models.list(), lambda n: ws.models.archive(name=n))
    _sweep("data-asset", lambda: ws.data.list(), lambda n: ws.data.archive(name=n))
    fs = os.environ.get("AZUREML_FEATURE_STORE_NAME")
    if fs:
        try:
            fsc = _ml(feature_store=fs)
            _sweep(
                "feature-set",
                lambda: fsc.feature_sets.list(),
                lambda n: fsc.feature_sets.archive(name=n),
            )
        except Exception as e:
            print(f"[azure teardown] feature-store sweep skipped: {e}")


def verify(all_mode: bool = False) -> int:
    """Twofold teardown check (`teardown.py verify`): (1) Azure ML CONNECTS, then
    (2) no billed online/batch ENDPOINTS survive (a managed online endpoint runs
    a VM continuously). In per-run mode only THIS run's prefixed endpoints count
    as a leak; with --all (or no prefix) any surviving endpoint does. Exit
    non-zero on no-connection or a leak; the runner WARNS on a leak. Read-only,
    best-effort."""
    global _RUN_MODE
    _RUN_MODE = bool(PREFIX) and not all_mode
    try:
        ws = _ml(workspace=os.environ.get("AZUREML_WORKSPACE_NAME"))
        online = [
            getattr(e, "name", "?")
            for e in ws.online_endpoints.list()
            if _scoped(getattr(e, "name", None))
        ]
        batch = [
            getattr(e, "name", "?")
            for e in ws.batch_endpoints.list()
            if _scoped(getattr(e, "name", None))
        ]
    except Exception as e:
        print(f"[azure verify-teardown] NO CONNECTION: {e}")
        return 1
    leaks = [f"online-endpoint {n!r}" for n in online] + [f"batch-endpoint {n!r}" for n in batch]
    if leaks:
        print("[azure verify-teardown] LEAKS remain:\n  - " + "\n  - ".join(leaks))
        return 1
    print("[azure verify-teardown] OK: connected; no endpoints remain")
    return 0


if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]
    all_mode = "--all" in argv
    do_verify = "verify" in argv
    sys.exit(verify(all_mode) if do_verify else (main(all_mode) or 0))
