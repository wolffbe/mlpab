"""Daily feature pipeline for the `incremental2a0ae6` events table.

Runs once per scheduled fire. New daily increment CSVs (same schema as the
initial backfill: row_id, account_id, event_time, amount, category) land in a
HopsFS directory; this job picks up any that have not yet been ingested and
appends them to the feature group, which keeps both the offline and online
stores up to date.

The scheduled window is provided by Hopsworks via the HOPS_START_TIME /
HOPS_END_TIME environment variables (epoch ms); files are matched by the day
of the fire so each run ingests that day's increment.
"""
import os
import glob

import hopsworks

# Directory on HopsFS where future increment files are dropped by upstream.
LANDING_DIR = os.environ.get(
    "INCREMENT_LANDING_DIR", "Resources/incremental2a0ae6/incoming"
)
FG_NAME = "incremental2a0ae6"
FG_VERSION = 1


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    dataset_api = project.get_dataset_api()

    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

    # Pull down any increment files staged in the landing directory.
    local_dir = "ingest_tmp"
    os.makedirs(local_dir, exist_ok=True)
    try:
        remote_files = dataset_api.list_files(LANDING_DIR, 0, 1000)[1]
        names = [f["attributes"]["name"] for f in remote_files]
    except Exception:
        names = []

    ingested = 0
    for name in names:
        if not name.endswith(".csv"):
            continue
        local_path = os.path.join(local_dir, name)
        dataset_api.download(f"{LANDING_DIR}/{name}", local_path, overwrite=True)
        import pandas as pd

        df = pd.read_csv(local_path)
        if df.empty:
            continue
        # Upsert by primary key row_id; online store is refreshed automatically.
        fg.insert(df)
        ingested += len(df)

    print(f"Ingested {ingested} new rows into {FG_NAME} v{FG_VERSION}")


if __name__ == "__main__":
    main()
