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


def main() -> None:
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


if __name__ == "__main__":
    main()
