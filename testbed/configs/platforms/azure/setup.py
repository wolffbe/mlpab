"""Azure ML platform setup — ensure the managed feature store the agent cannot
create, and hand its name to the agent.

Run by `mlpab` (the interface `serve:` step) at the START of every run, right
after teardown.py. Mirrors the sagemaker/databricks contract: best-effort,
NEVER raises out of main() (a setup hiccup must not fail an agent run), and
exports any discovered env to the agent via $MLPAB_PLATFORM_ENV (KEY=VALUE).

What it guarantees: a managed FEATURE STORE exists (the FTI feature primitive
the feature tasks land in). Creating one provisions a feature-store workspace —
slow — so we reuse AZUREML_FEATURE_STORE_NAME if set, else ensure a default
`mlpabfs` and export its name. The workspace itself is assumed pre-provisioned
(AZUREML_WORKSPACE_NAME). Uses azure-ai-ml (installed in the run venv).

⚠ LIVE-VALIDATION REQUIRED — the feature-store create call is convention-based.
"""

from __future__ import annotations

import os

DEFAULT_FS = "mlpabfs"


def _export(key: str, value: str) -> None:
    path = os.environ.get("MLPAB_PLATFORM_ENV")
    if path and value:
        with open(path, "a") as fh:
            fh.write(f"{key}={value}\n")


def main() -> None:
    fs_name = os.environ.get("AZUREML_FEATURE_STORE_NAME") or DEFAULT_FS
    sub = os.environ.get("AZURE_SUBSCRIPTION_ID")
    rg = os.environ.get("AZURE_RESOURCE_GROUP")
    location = os.environ.get("AZURE_LOCATION", "eastus")
    if not (sub and rg):
        print("[azure setup] missing AZURE_SUBSCRIPTION_ID/RESOURCE_GROUP — skipping")
        return
    try:
        from azure.ai.ml import MLClient
        from azure.ai.ml.entities import FeatureStore
        from azure.identity import DefaultAzureCredential

        cred = DefaultAzureCredential()
        ml = MLClient(cred, sub, rg)
        try:
            ml.feature_stores.get(name=fs_name)
            print(f"[azure setup] feature store {fs_name!r} already exists")
        except Exception:
            print(f"[azure setup] creating feature store {fs_name!r} in {location} …")
            ml.feature_stores.begin_create(FeatureStore(name=fs_name, location=location)).result()
            print(f"[azure setup] created feature store {fs_name!r}")
        _export("AZUREML_FEATURE_STORE_NAME", fs_name)
    except Exception as e:
        print(f"[azure setup] best-effort skip: {e}")


def verify() -> int:
    """Twofold setup check (`setup.py verify`): (1) Azure ML CONNECTS, then (2)
    the managed feature store setup guarantees is PRESENT. Connection is the hard
    gate; the feature store is best-effort. Read-only."""
    fs_name = os.environ.get("AZUREML_FEATURE_STORE_NAME") or DEFAULT_FS
    sub = os.environ.get("AZURE_SUBSCRIPTION_ID")
    rg = os.environ.get("AZURE_RESOURCE_GROUP")
    if not (sub and rg):
        print("[azure verify-setup] missing AZURE_SUBSCRIPTION_ID/RESOURCE_GROUP")
        return 1
    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        ml = MLClient(DefaultAzureCredential(), sub, rg)
        list(ml.workspaces.list())  # forces a real authenticated round-trip
    except Exception as e:
        print(f"[azure verify-setup] NO CONNECTION: {e}")
        return 1
    try:
        ml.feature_stores.get(name=fs_name)
        print(f"[azure verify-setup] OK: connected; feature store {fs_name!r} present")
    except Exception as e:
        print(
            f"[azure verify-setup] OK: connected; feature store {fs_name!r} absent "
            f"(best-effort, not failing): {e}"
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(verify() if sys.argv[1:2] == ["verify"] else (main() or 0))
