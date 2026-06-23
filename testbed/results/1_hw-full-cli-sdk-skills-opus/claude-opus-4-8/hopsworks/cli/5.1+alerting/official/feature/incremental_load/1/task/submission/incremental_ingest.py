"""Daily incremental ingestion pipeline for the `incremental614551` feature group.

Runs as a recurring Hopsworks job (`incrementaljob614551`). On each daily fire it
scans a HopsFS drop directory for new increment CSV files (same schema as the
events table), loads any not yet ingested, and inserts their rows into the
feature group. The feature group is online-enabled, so inserts land in both the
offline and online stores.

New increment files are expected to arrive at:
    Resources/incremental614551/increments/increment_*.csv

A marker directory (Resources/incremental614551/ingested/) records which files
have already been processed so re-fires are idempotent.
"""

import io
import os

import pandas as pd
import hopsworks

FG_NAME = "incremental614551"
FG_VERSION = 1
DROP_DIR = "Resources/incremental614551/increments"
MARKER_DIR = "Resources/incremental614551/ingested"

# Expected column dtypes (epoch milliseconds for event_time).
DTYPES = {
    "row_id": str,
    "account_id": str,
    "event_time": "int64",
    "amount": "float64",
    "category": str,
}


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    dataset_api = project.get_dataset_api()

    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

    # Ensure the drop and marker directories exist.
    for d in (DROP_DIR, MARKER_DIR):
        try:
            dataset_api.mkdir(d)
        except Exception:
            pass

    # List candidate increment files in the drop directory.
    try:
        entries = dataset_api.list_files(DROP_DIR, 0, 1000)
    except Exception as e:
        print(f"No drop directory contents to ingest yet: {e}")
        return

    # `list_files` returns (count, items); normalise to a list of paths.
    items = entries[1] if isinstance(entries, tuple) else entries
    files = []
    for it in items:
        path = it.get("attributes", {}).get("path") if isinstance(it, dict) else getattr(it, "path", None)
        if path and path.endswith(".csv"):
            files.append(path)

    # Determine which files are already ingested.
    try:
        done_entries = dataset_api.list_files(MARKER_DIR, 0, 1000)
        done_items = done_entries[1] if isinstance(done_entries, tuple) else done_entries
        done = set()
        for it in done_items:
            p = it.get("attributes", {}).get("path") if isinstance(it, dict) else getattr(it, "path", None)
            if p:
                done.add(os.path.basename(p).replace(".done", ""))
    except Exception:
        done = set()

    new_files = [f for f in sorted(files) if os.path.basename(f) not in done]
    if not new_files:
        print("No new increments to ingest.")
        return

    total = 0
    for path in new_files:
        base = os.path.basename(path)
        local = dataset_api.download(path, overwrite=True)
        df = pd.read_csv(local, dtype=DTYPES)
        fg.insert(df)
        total += len(df)
        print(f"Ingested {len(df)} rows from {base}")
        # Mark as ingested.
        marker = f"{base}.done"
        with open(marker, "w") as fh:
            fh.write("ok")
        try:
            dataset_api.upload(marker, MARKER_DIR, overwrite=True)
        except Exception as e:
            print(f"Warning: could not write marker for {base}: {e}")

    print(f"Done. Ingested {total} rows across {len(new_files)} new increment file(s).")


if __name__ == "__main__":
    main()
