"""Daily incremental-load pipeline for the `incremental4200a8` feature group.

Runs on the Hopsworks platform on a daily schedule. On each fire it picks up any
new daily increment file(s) that have landed since the last run and ingests them
into the feature group. New increments share the documented schema:
    row_id (string, record key), account_id (string),
    event_time (bigint, epoch ms), amount (double), category (string).

Increment files are expected to land in a HopsFS dataset directory; each fire
ingests files not yet loaded. Ingestion is an upsert on the record key row_id,
so re-processing a file is idempotent.
"""
import os
import glob
import pandas as pd
import hopsworks

FG_NAME = "incremental4200a8"
FG_VERSION = 1

# HopsFS directory where future daily increment files are dropped.
INCREMENT_DIR = os.environ.get("INCREMENT_DIR", "Resources/incremental4200a8/increments")
# Marker file recording which increments have already been ingested.
STATE_FILE = os.environ.get(
    "INGESTED_STATE", "Resources/incremental4200a8/_ingested.txt"
)


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    dataset_api = project.get_dataset_api()

    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

    # Determine which increment files have already been ingested.
    already = set()
    try:
        local_state = dataset_api.download(STATE_FILE, overwrite=True)
        with open(local_state) as f:
            already = {ln.strip() for ln in f if ln.strip()}
    except Exception:
        already = set()

    # Pull the increment directory locally and find new files.
    try:
        files = dataset_api.list(INCREMENT_DIR)
    except Exception:
        files = []

    new_local_files = []
    new_names = []
    for remote in files:
        base = os.path.basename(remote.rstrip("/"))
        if not base.endswith(".csv") or base in already:
            continue
        local = dataset_api.download(
            os.path.join(INCREMENT_DIR, base), overwrite=True
        )
        new_local_files.append(local)
        new_names.append(base)

    if not new_local_files:
        print("No new increments to ingest.")
        return

    for local, base in zip(new_local_files, new_names):
        df = pd.read_csv(local)
        df["row_id"] = df["row_id"].astype(str)
        df["account_id"] = df["account_id"].astype(str)
        df["event_time"] = df["event_time"].astype("int64")
        df["amount"] = df["amount"].astype("float64")
        df["category"] = df["category"].astype(str)
        print(f"Ingesting {base}: {len(df)} rows")
        fg.insert(df)
        already.add(base)

    # Persist updated state.
    with open("_ingested.txt", "w") as f:
        f.write("\n".join(sorted(already)) + "\n")
    dataset_api.upload(
        "_ingested.txt",
        os.path.dirname(STATE_FILE),
        overwrite=True,
    )
    print(f"Done. Ingested {len(new_names)} new increment file(s).")


if __name__ == "__main__":
    main()
