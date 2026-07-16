"""Daily ingestion job: load event increments into the incremental76da9e feature group.

Reads every increment CSV in Resources/incremental76da9e_data and upserts it
into feature group incremental76da9e (v1). Inserts are keyed on row_id, so
re-processing already-ingested files is idempotent and newly arrived daily
increment files are picked up automatically on each scheduled run.
"""

import os

import hopsworks
import pandas as pd

DATA_DIR = "Resources/incremental76da9e_data"
FG_NAME = "incremental76da9e"
FG_VERSION = 1

project = hopsworks.login()
fs = project.get_feature_store()
dataset_api = project.get_dataset_api()

fg = fs.get_or_create_feature_group(
    name=FG_NAME,
    version=FG_VERSION,
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
    description="Daily events increments",
)

paths = [p for p in dataset_api.list(DATA_DIR) if p.endswith(".csv")]
if not paths:
    print(f"No increment files found in {DATA_DIR}; nothing to ingest.")
else:
    frames = []
    for path in sorted(paths):
        local = dataset_api.download(path, overwrite=True)
        df = pd.read_csv(local)
        df["row_id"] = df["row_id"].astype(str)
        df["account_id"] = df["account_id"].astype(str)
        df["event_time"] = df["event_time"].astype("int64")
        df["amount"] = df["amount"].astype("float64")
        df["category"] = df["category"].astype(str)
        frames.append(df)
        print(f"Read {len(df)} rows from {path}")
        try:
            os.remove(local)
        except OSError:
            pass

    data = pd.concat(frames, ignore_index=True)
    print(f"Inserting {len(data)} rows into {FG_NAME} v{FG_VERSION}")
    fg.insert(data, wait=True)
    print("Ingestion complete.")
