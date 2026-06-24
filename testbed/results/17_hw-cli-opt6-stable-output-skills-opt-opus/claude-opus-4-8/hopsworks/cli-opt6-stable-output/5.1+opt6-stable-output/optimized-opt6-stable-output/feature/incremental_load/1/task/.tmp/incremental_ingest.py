"""Daily incremental ingestion job for the `incremental1b589f` feature group.

Runs on the platform on a daily schedule. Each fire it picks up newly arrived
increment file(s) for the run's data window and inserts them into the feature
group. The feature group is online-enabled, so each insert refreshes both the
offline and online stores (upsert on the `row_id` record key).

The schedule passes the data window via the HOPS_START_TIME / HOPS_END_TIME
environment variables (epoch ms). Increment files are expected to land under a
HopsFS landing directory as `increment_*.csv` with the documented schema:
    row_id (string), account_id (string), event_time (bigint, epoch ms),
    amount (double), category (string).
"""
import os
import glob

import pandas as pd
import hopsworks

FG_NAME = "incremental1b589f"
FG_VERSION = 1
# HopsFS landing directory where future daily increments arrive.
LANDING_DIR = os.environ.get("INCREMENT_LANDING_DIR", "Resources/incremental1b589f_landing")


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

    dataset_api = project.get_dataset_api()

    # Discover increment files for this run. Download any increment_*.csv that
    # have landed; in a real deployment the producer drops one file per day.
    work_dir = os.environ.get("HOPS_JOB_TMP", ".")
    local_files = []
    try:
        entries = dataset_api.list_files(LANDING_DIR, 0, 1000)[1]
        for entry in entries:
            remote = entry["attributes"]["path"]
            base = os.path.basename(remote)
            if base.startswith("increment_") and base.endswith(".csv"):
                local = os.path.join(work_dir, base)
                dataset_api.download(remote, local, overwrite=True)
                local_files.append(local)
    except Exception as exc:  # landing dir may not exist yet on first fires
        print(f"No landing directory yet or list failed: {exc}")

    if not local_files:
        # Fall back to any increment files staged next to the job script.
        local_files = sorted(glob.glob("increment_*.csv"))

    if not local_files:
        print("No new increment files to ingest this run.")
        return

    total = 0
    for path in sorted(local_files):
        df = pd.read_csv(path)
        df["row_id"] = df["row_id"].astype(str)
        df["account_id"] = df["account_id"].astype(str)
        df["event_time"] = df["event_time"].astype("int64")
        df["amount"] = df["amount"].astype("float64")
        df["category"] = df["category"].astype(str)
        fg.insert(df)
        total += len(df)
        print(f"Ingested {len(df)} rows from {path}")

    print(f"Daily ingestion complete: {total} rows into {FG_NAME} v{FG_VERSION}")


if __name__ == "__main__":
    main()
