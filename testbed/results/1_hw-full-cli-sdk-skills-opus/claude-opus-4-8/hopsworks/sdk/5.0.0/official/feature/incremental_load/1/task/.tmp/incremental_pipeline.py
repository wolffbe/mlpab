"""Daily feature pipeline: ingest new increment files into incremental614551.

Runs on a daily schedule. Reads any increment CSV files staged in the project's
Resources/incremental614551_increments directory in HopsFS, inserts the new rows
into the feature group (online + offline), and moves processed files aside so
they are not re-ingested. The feature group upserts on row_id, so re-ingesting an
already-loaded row is idempotent.
"""
import os
import pandas as pd
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
dataset_api = project.get_dataset_api()

fg = fs.get_feature_group("incremental614551", version=1)

INCOMING = "Resources/incremental614551_increments"
PROCESSED = "Resources/incremental614551_increments/processed"

try:
    dataset_api.mkdir(INCOMING)
except Exception:
    pass
try:
    dataset_api.mkdir(PROCESSED)
except Exception:
    pass

try:
    entries = dataset_api.list_files(INCOMING, 0, 1000)[1]
    paths = [e["attributes"]["path"] for e in entries]
except Exception:
    paths = []

csvs = [p for p in paths if p.endswith(".csv")]
print("Found increment files:", csvs)

local_dir = "/tmp/incremental614551_increments"
os.makedirs(local_dir, exist_ok=True)

total = 0
for remote in csvs:
    fname = remote.split("/")[-1]
    local = os.path.join(local_dir, fname)
    dataset_api.download(remote, local, overwrite=True)
    df = pd.read_csv(local, dtype={"row_id": str, "account_id": str, "category": str})
    df["event_time"] = df["event_time"].astype("int64")
    df["amount"] = df["amount"].astype("float64")
    print(f"{fname}: {len(df)} rows")
    fg.insert(df, wait=True)
    total += len(df)
    # archive processed file
    try:
        dataset_api.move(remote, f"{PROCESSED}/{fname}")
    except Exception as e:
        print("archive failed:", e)

print("Total new rows ingested:", total)
