"""Azure ML platform teardown — sweep agent-created resources.

Run by `mlpab` at the START and END of every run. Best-effort, NEVER raises.
Deletes in cost order (online/batch endpoints → jobs → models → feature sets →
data assets), preserving the WORKSPACE and the FEATURE STORE themselves (those
are platform plumbing setup.py guarantees, not agent state).

⚠ LIVE-VALIDATION REQUIRED before any real run — a teardown that misses a
resource type leaks billable Azure resources (managed online endpoints in
particular run a VM continuously).
"""
from __future__ import annotations

import os


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
        try:
            deleter(name)
            print(f"[azure teardown] deleted {label} {name!r}")
        except Exception as e:
            print(f"[azure teardown] delete {label} {name!r} skipped: {e}")


def main() -> None:
    try:
        ws = _ml(workspace=os.environ.get("AZUREML_WORKSPACE_NAME"))
    except Exception as e:
        print(f"[azure teardown] no workspace client: {e}")
        return
    # cost order: serving first (a live endpoint bills continuously)
    _sweep("online-endpoint", lambda: ws.online_endpoints.list(),
           lambda n: ws.online_endpoints.begin_delete(n))
    _sweep("batch-endpoint", lambda: ws.batch_endpoints.list(),
           lambda n: ws.batch_endpoints.begin_delete(n))
    # Compute clusters/instances bill continuously while provisioned (a training
    # job's compute target the agent created) — delete them right after serving.
    _sweep("compute", lambda: ws.compute.list(),
           lambda n: ws.compute.begin_delete(name=n))
    _sweep("job", lambda: ws.jobs.list(),
           lambda n: ws.jobs.begin_cancel(n))
    _sweep("model", lambda: ws.models.list(),
           lambda n: ws.models.archive(name=n))
    _sweep("data-asset", lambda: ws.data.list(),
           lambda n: ws.data.archive(name=n))
    fs = os.environ.get("AZUREML_FEATURE_STORE_NAME")
    if fs:
        try:
            fsc = _ml(feature_store=fs)
            _sweep("feature-set", lambda: fsc.feature_sets.list(),
                   lambda n: fsc.feature_sets.archive(name=n))
        except Exception as e:
            print(f"[azure teardown] feature-store sweep skipped: {e}")


if __name__ == "__main__":
    main()
