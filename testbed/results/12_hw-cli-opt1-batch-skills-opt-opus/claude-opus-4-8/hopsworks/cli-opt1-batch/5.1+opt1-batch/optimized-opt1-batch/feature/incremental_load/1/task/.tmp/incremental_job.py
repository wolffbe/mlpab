"""Recurring daily feature pipeline for the `incremental0c46ea` events table.

Runs as a scheduled Hopsworks job (daily). On each fire it picks up any new
daily increment CSV files that have been dropped into the project's HopsFS
landing directory and ingests them into feature group `incremental0c46ea` v1.

New increments share the schema documented in data/schema.md:
    row_id (string, key), account_id (string),
    event_time (bigint, epoch ms, event-time column),
    amount (double), category (string)

Ingestion is an upsert on `row_id`, so re-processing a file is idempotent and
both the offline and online stores are kept up to date.
"""

import os

import pandas as pd
import hopsworks

# HopsFS directory where future daily increments are dropped (one CSV per day).
LANDING_DIR = os.environ.get(
    "INCREMENT_LANDING_DIR", "Resources/incremental0c46ea_increments"
)
FG_NAME = "incremental0c46ea"
FG_VERSION = 1
SCHEMA = ["row_id", "account_id", "event_time", "amount", "category"]


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

    dataset_api = project.get_dataset_api()

    # Discover the increment files currently waiting in the landing directory.
    try:
        entries = dataset_api.list_files(LANDING_DIR, 0, 1000)[1]
        paths = [e["attributes"]["path"] for e in entries]
    except Exception:
        paths = []
    csv_paths = sorted(p for p in paths if p.endswith(".csv"))

    if not csv_paths:
        print(f"No new increment files found under {LANDING_DIR}; nothing to ingest.")
        return

    frames = []
    for remote in csv_paths:
        local = dataset_api.download(remote, overwrite=True)
        frames.append(pd.read_csv(local))
        print(f"Loaded increment {remote}")

    df = pd.concat(frames, ignore_index=True)[SCHEMA]
    df["row_id"] = df["row_id"].astype(str)
    df["account_id"] = df["account_id"].astype(str)
    df["event_time"] = df["event_time"].astype("int64")
    df["amount"] = df["amount"].astype("float64")
    df["category"] = df["category"].astype(str)

    fg.insert(df)
    print(f"Ingested {len(df)} rows into {FG_NAME} v{FG_VERSION} (offline + online).")


if __name__ == "__main__":
    main()
