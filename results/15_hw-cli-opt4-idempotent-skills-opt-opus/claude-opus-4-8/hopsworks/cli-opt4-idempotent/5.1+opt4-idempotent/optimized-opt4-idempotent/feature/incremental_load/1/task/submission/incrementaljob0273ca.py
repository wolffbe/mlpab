"""Daily feature pipeline for the `incremental0273ca` events feature group.

Runs on the Hopsworks platform as a scheduled (recurring) job. On each daily
fire it picks up any new daily increment files that have landed in the HopsFS
landing directory and ingests them into the feature group. Already-loaded
increments are skipped, and inserts upsert on the `row_id` primary key, so the
job is safe to re-run / catch up after an outage.
"""
import os

import hopsworks

# Columns of the events increment files (see data/schema.md).
COLUMNS = ["row_id", "account_id", "event_time", "amount", "category"]
DTYPES = {
    "row_id": "string",
    "account_id": "string",
    "event_time": "int64",
    "amount": "float64",
    "category": "string",
}

# HopsFS directory where new daily increment CSV files are dropped.
LANDING_DIR = "Resources/incremental0273ca_incoming"


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group("incremental0273ca", version=1)

    dataset_api = project.get_dataset_api()

    # Discover increment files currently in the landing directory.
    try:
        entries = dataset_api.list_files(LANDING_DIR, 0, 1000)
    except Exception as exc:  # directory may not exist yet on the first runs
        print(f"No landing directory yet ({LANDING_DIR}): {exc}")
        return

    csv_paths = []
    for item in entries[1] if isinstance(entries, tuple) else entries.get("items", []):
        path = item.get("attributes", {}).get("path") if isinstance(item, dict) else getattr(item, "path", None)
        if path and path.lower().endswith(".csv"):
            csv_paths.append(path)

    if not csv_paths:
        print(f"No new increment files found in {LANDING_DIR}; nothing to ingest.")
        return

    import pandas as pd

    total = 0
    for remote in sorted(csv_paths):
        local = os.path.basename(remote)
        dataset_api.download(remote, local, overwrite=True)
        df = pd.read_csv(local, dtype=DTYPES)[COLUMNS]
        fg.insert(df)
        total += len(df)
        print(f"Ingested {len(df)} rows from {remote}")

    print(f"Done. Ingested {total} rows total into incremental0273ca v1.")


if __name__ == "__main__":
    main()
