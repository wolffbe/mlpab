"""Daily feature pipeline for the `incremental3a432c` events feature group.

Runs on the Hopsworks platform as a scheduled (recurring) PYTHON job. On each
fire it ingests the day's new increment file(s) into the feature group. New
increments are expected to land in HopsFS under ``Resources/increments/`` with
the same schema as the registered feature group:

    row_id (string, PK), account_id (string),
    event_time (bigint, epoch ms, event-time column),
    amount (double), category (string)

The job window is provided by the scheduler via the HOPS_START_TIME /
HOPS_END_TIME environment variables; we ingest every increment file that has
not yet been loaded. insert() upserts on row_id, so re-processing a file is
idempotent.
"""
import os
import glob

import pandas as pd
import hopsworks

FG_NAME = "incremental3a432c"
FG_VERSION = 1
# HopsFS directory new daily increments are dropped into.
INCREMENTS_DIR = "Resources/increments"


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    dataset_api = project.get_dataset_api()

    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

    # Pull any increment files staged in HopsFS down to the job's local cwd.
    local_dir = "ingest_tmp"
    os.makedirs(local_dir, exist_ok=True)
    try:
        remote_files = dataset_api.list_files(INCREMENTS_DIR, 0, 1000)[1]
    except Exception:
        remote_files = []

    csv_paths = []
    for entry in remote_files:
        remote_path = entry.get("attributes", {}).get("path") or entry.get("path")
        if remote_path and remote_path.endswith(".csv"):
            local_path = os.path.join(local_dir, os.path.basename(remote_path))
            dataset_api.download(remote_path, local_path, overwrite=True)
            csv_paths.append(local_path)

    if not csv_paths:
        # Fall back to any increment CSVs shipped alongside the job script.
        csv_paths = sorted(glob.glob("increment_*.csv"))

    if not csv_paths:
        print("No new increment files found; nothing to ingest.")
        return

    for path in sorted(csv_paths):
        df = pd.read_csv(path)
        if df.empty:
            continue
        print(f"Ingesting {len(df)} rows from {path} into {FG_NAME} v{FG_VERSION}")
        # Writes both offline and online stores (FG is online-enabled).
        fg.insert(df)

    print("Daily incremental ingest complete.")


if __name__ == "__main__":
    main()
