"""Ingestion job for feature group incremental76da9e (version 1).

Runs daily on the Hopsworks cluster. It picks up all increment CSV files
staged under Resources/incremental76da9e/ in the project datasets and
upserts them into the feature group (offline + online). New daily
increment files uploaded to that directory are ingested on the next run;
already-ingested rows are simply upserted again via the row_id key.
"""

import os

import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()
ds_api = project.get_dataset_api()

DATA_DIR = "Resources/incremental76da9e"

paths = []
for item in ds_api.list(DATA_DIR):
    if isinstance(item, dict):
        item = item.get("attributes", item).get("path", "")
    path = str(item)
    if path.endswith(".csv"):
        paths.append(path)

paths.sort()
print("increment files found:", paths)
if not paths:
    raise SystemExit("no increment files found in " + DATA_DIR)

frames = []
for path in paths:
    local = ds_api.download(path, overwrite=True)
    frames.append(pd.read_csv(local))
    os.remove(local)

df = pd.concat(frames, ignore_index=True)
df["row_id"] = df["row_id"].astype(str)
df["account_id"] = df["account_id"].astype(str)
df["category"] = df["category"].astype(str)
df["event_time"] = df["event_time"].astype("int64")
df["amount"] = df["amount"].astype("float64")
print("rows to upsert:", len(df))

fg = fs.get_or_create_feature_group(
    name="incremental76da9e",
    version=1,
    description="Daily events increments",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
)

fg.insert(df, wait=True)
print("ingestion complete:", len(df), "rows upserted into incremental76da9e v1")
