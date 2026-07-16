"""Daily incremental ingestion job for feature group `incremental614551`.

Runs on the Hopsworks platform on a daily schedule. It scans an inbox
directory on the Hopsworks File System for newly-arrived daily increment
CSV files (same schema as the initial load), loads each one, inserts it
into the feature group (offline + online), and moves the processed file
to an archive directory so it is not re-ingested.
"""
import os
import glob
import pandas as pd
import hopsworks

# Inbox + archive locations on the Hopsworks File System (HDFS-style paths).
INBOX = "Resources/incremental614551_inbox"
ARCHIVE = "Resources/incremental614551_archive"

proj = hopsworks.login()
fs = proj.get_feature_store()
dataset_api = proj.get_dataset_api()

fg = fs.get_or_create_feature_group(
    name="incremental614551",
    version=1,
    description="Daily events increments table",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
)

# Make sure inbox/archive exist.
for d in (INBOX, ARCHIVE):
    try:
        if not dataset_api.exists(d):
            dataset_api.mkdir(d)
    except Exception as e:
        print("mkdir/exists check skipped for", d, ":", e)

# List new increment files in the inbox.
try:
    entries = dataset_api.list_files(INBOX, 0, 1000)[1]
    paths = [e["attributes"]["path"] for e in entries
             if e["attributes"]["name"].endswith(".csv")]
except Exception as e:
    print("Could not list inbox, nothing to ingest:", e)
    paths = []

print("New increment files found:", paths)

local_tmp = "/tmp/incremental614551_ingest"
os.makedirs(local_tmp, exist_ok=True)

total = 0
for remote in sorted(paths):
    name = os.path.basename(remote)
    local = dataset_api.download(remote, local_path=local_tmp, overwrite=True)
    df = pd.read_csv(local)
    print("Ingesting", name, len(df), "rows")
    fg.insert(df, write_options={"wait_for_job": True})
    total += len(df)
    # Archive the processed file so it is not ingested again tomorrow.
    try:
        dataset_api.move(remote, ARCHIVE + "/" + name, overwrite=True)
    except Exception as e:
        print("Could not archive", name, ":", e)

print("Daily ingestion complete. Rows ingested:", total)
