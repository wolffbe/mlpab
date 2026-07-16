"""Daily feature pipeline for the `incremental16fc4b` events feature group.

Runs on the platform on a daily schedule. Each fire ingests any new daily
increment files that have landed in the project's HopsFS landing zone since
the last run, appending their rows to the feature group (offline + online).

The landing zone is a HopsFS directory into which new increment CSVs
(`increment_*.csv`, same schema as the historical backfill) are dropped each
day. Already-ingested files are tracked via a marker directory so re-runs are
idempotent.
"""
import os

import hopsworks

FG_NAME = "incremental16fc4b"
FG_VERSION = 1
LANDING_DIR = "Resources/incremental16fc4b_increments"
PROCESSED_DIR = "Resources/incremental16fc4b_increments/_processed"


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)
    dataset_api = project.get_dataset_api()

    # Make sure the landing + processed marker directories exist.
    for d in (LANDING_DIR, PROCESSED_DIR):
        try:
            if not dataset_api.exists(d):
                dataset_api.mkdir(d)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not ensure directory {d}: {exc}")

    # Discover increment files waiting in the landing zone.
    try:
        entries = dataset_api.list_files(LANDING_DIR, 0, 1000)
        names = []
        # list_files returns (inode_count, items) on most SDK versions.
        items = entries[1] if isinstance(entries, tuple) else entries
        for it in items:
            path = it.get("attributes", {}).get("path") if isinstance(it, dict) else getattr(it, "path", str(it))
            base = os.path.basename(path.rstrip("/"))
            if base.startswith("increment_") and base.endswith(".csv"):
                names.append(base)
    except Exception as exc:  # noqa: BLE001
        print(f"No landing files found / cannot list landing zone: {exc}")
        names = []

    if not names:
        print("No new increments to ingest this run.")
        return

    import pandas as pd

    ingested = 0
    for name in sorted(names):
        remote = f"{LANDING_DIR}/{name}"
        local = f"/tmp/{name}"
        try:
            dataset_api.download(remote, local, overwrite=True)
            df = pd.read_csv(local)
            fg.insert(df)
            ingested += len(df)
            # Mark as processed so we don't re-ingest on the next fire.
            try:
                dataset_api.move(remote, f"{PROCESSED_DIR}/{name}")
            except Exception:  # noqa: BLE001
                pass
            print(f"Ingested {len(df)} rows from {name}")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to ingest {name}: {exc}")

    print(f"Daily ingestion complete: {ingested} rows into {FG_NAME} v{FG_VERSION}")


if __name__ == "__main__":
    main()
