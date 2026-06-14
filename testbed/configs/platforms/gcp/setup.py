"""Gemini Enterprise Agent Platform (GCP) platform setup — ensure the feature-store plumbing.

Run by `mlpab` (the interface `serve:` step) at the START of every run, right
after teardown.py. Best-effort, NEVER raises out of main(). the platform's feature
store is BigQuery-backed, so the plumbing the agent needs is a BigQuery DATASET
(the feature-group backing store) — ensure it exists. Uses the
google-cloud-bigquery client installed in the run venv.

⚠ LIVE-VALIDATION REQUIRED — dataset location must match GCP_LOCATION's multi-
region; adjust if the project uses a specific region.
"""

from __future__ import annotations

import os


def _export_gcloud_token() -> None:
    """Mint a cloud-platform access token from ADC and export it to
    $MLPAB_PLATFORM_ENV as CLOUDSDK_AUTH_ACCESS_TOKEN.

    The gcloud CLI honors that env var as a pre-obtained bearer token, so a
    CLI-interface run authenticates gcloud WITHOUT a service-account key file —
    which `gcloud auth activate-service-account`/`gcloud auth login --cred-file`
    would require and which REJECT the impersonated-SA ADC file this project
    uses. ADC (`google.auth.default`) consumes that file transparently
    (authorized_user source → impersonated SA), so the token is valid for the
    impersonated identity. mlpab reads $MLPAB_PLATFORM_ENV back into both the
    login check and the agent env, so the agent's gcloud calls inherit it. The
    Python SDK ignores the var (it uses ADC directly), so this is a no-op for
    SDK runs. Best-effort: on any failure gcloud falls back to its own auth.

    Run under the plumbing venv (mlpab's $MLPAB_PLUMBING_PY), which has
    google-auth; the token lives ~1h, far longer than a run.
    """
    env_file = os.environ.get("MLPAB_PLATFORM_ENV")
    if not env_file:
        return
    try:
        import google.auth
        import google.auth.transport.requests as greq

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(greq.Request())
        if creds.token:
            with open(env_file, "a") as f:
                f.write(f"CLOUDSDK_AUTH_ACCESS_TOKEN={creds.token}\n")
            print("[gcp setup] exported CLOUDSDK_AUTH_ACCESS_TOKEN for gcloud (from ADC)")
    except Exception as e:
        print(f"[gcp setup] gcloud token export skipped: {e}")


def main() -> None:
    # Hand gcloud a bearer token from ADC first (independent of the dataset
    # plumbing below, and needed even when GCP_PROJECT is unset).
    _export_gcloud_token()
    project = os.environ.get("GCP_PROJECT")
    dataset = os.environ.get("GCP_BQ_DATASET", "mlpab")
    location = os.environ.get("GCP_LOCATION", "us-central1")
    if not project:
        print("[gcp setup] missing GCP_PROJECT — skipping")
        return
    try:
        from google.cloud import bigquery

        bq = bigquery.Client(project=project)
        ref = bigquery.Dataset(f"{project}.{dataset}")
        # BigQuery datasets live in a (multi-)region; derive a sane default.
        ref.location = "US" if location.startswith("us") else location
        bq.create_dataset(ref, exists_ok=True)
        print(f"[gcp setup] BigQuery dataset {project}.{dataset} ready ({ref.location})")
    except Exception as e:
        print(f"[gcp setup] best-effort skip: {e}")


def verify() -> int:
    """Twofold setup check (`setup.py verify`): (1) BigQuery CONNECTS, then (2)
    the backing dataset setup guarantees is PRESENT. Connection is the hard gate;
    the dataset is best-effort. Read-only."""
    project = os.environ.get("GCP_PROJECT")
    dataset = os.environ.get("GCP_BQ_DATASET", "mlpab")
    if not project:
        print("[gcp verify-setup] missing GCP_PROJECT")
        return 1
    try:
        from google.cloud import bigquery

        bq = bigquery.Client(project=project)
        list(bq.list_datasets(max_results=1))  # authenticated round-trip
    except Exception as e:
        print(f"[gcp verify-setup] NO CONNECTION: {e}")
        return 1
    try:
        bq.get_dataset(f"{project}.{dataset}")
        print(f"[gcp verify-setup] OK: connected; dataset {project}.{dataset} present")
    except Exception as e:
        print(
            f"[gcp verify-setup] OK: connected; dataset {project}.{dataset} absent "
            f"(best-effort, not failing): {e}"
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(verify() if sys.argv[1:2] == ["verify"] else (main() or 0))
