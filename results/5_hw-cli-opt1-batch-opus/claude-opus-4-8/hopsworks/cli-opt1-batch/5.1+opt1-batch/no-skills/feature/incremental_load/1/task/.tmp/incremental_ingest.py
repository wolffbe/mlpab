"""Daily incremental ingestion pipeline for the `incremental3abc31` events table.

Runs as a recurring Hopsworks job. On each scheduled fire the platform exposes
the data window through the HOPS_START_TIME / HOPS_END_TIME environment
variables. The job connects to the feature store, locates the daily increment
file(s) that landed for that window, and inserts the new rows into the
`incremental3abc31` feature group (version 1). Because the feature group is
online-enabled, the insert backfills the offline store and materializes the
rows to the online store for low-latency lookup.
"""

import os
import glob

import hopsworks


# Feature group this pipeline feeds.
FG_NAME = "incremental3abc31"
FG_VERSION = 1

# Directory on the project's dataset where new daily increment files land.
# Future increments arrive here with the same schema as the bootstrap files.
INCREMENT_DIR = os.environ.get(
    "INCREMENT_DIR", "Resources/incremental3abc31_increments"
)


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()

    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

    # The schedule attaches a data window per fire; surface it for logging.
    start = os.environ.get("HOPS_START_TIME")
    end = os.environ.get("HOPS_END_TIME")
    print(f"Incremental ingest window: start={start} end={end}")

    dataset_api = project.get_dataset_api()

    # Pull any new increment files for this window down to the local worker,
    # then insert them. New files follow the increment_*.csv naming convention.
    local_dir = "ingest_tmp"
    os.makedirs(local_dir, exist_ok=True)

    try:
        remote_files = dataset_api.list_files(INCREMENT_DIR, 0, 1000)
    except Exception as exc:  # directory may not exist yet on first fires
        print(f"No increment directory {INCREMENT_DIR} yet ({exc}); nothing to ingest.")
        return

    total = 0
    import pandas as pd

    for entry in remote_files[1]:
        remote_path = entry["attributes"]["path"]
        if not remote_path.endswith(".csv"):
            continue
        local_path = dataset_api.download(remote_path, local_dir, overwrite=True)
        df = pd.read_csv(local_path)
        if df.empty:
            continue
        fg.insert(df)
        total += len(df)
        print(f"Inserted {len(df)} rows from {remote_path}")

    print(f"Daily incremental ingest complete: {total} rows into {FG_NAME} v{FG_VERSION}.")


if __name__ == "__main__":
    main()
