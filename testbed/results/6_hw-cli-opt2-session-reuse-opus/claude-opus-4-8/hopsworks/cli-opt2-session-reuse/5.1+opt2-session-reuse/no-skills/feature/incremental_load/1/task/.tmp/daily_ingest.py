"""Recurring daily ingestion job for the incremental865b5b events feature group.

Scheduled to fire once per day. On each fire it picks up newly arrived daily
increment files (same schema as the bootstrap increments) and appends them to
the feature group, which materializes them to both the offline and online
stores. The per-fire data window is provided by the scheduler via the
HOPS_START_TIME / HOPS_END_TIME environment variables.
"""
import os
import hopsworks

FG_NAME = "incremental865b5b"
FG_VERSION = 1
# HopsFS directory where new daily increment files are dropped.
INCREMENT_DIR = "Resources/incremental865b5b/incoming"


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

    dataset_api = project.get_dataset_api()

    # The scheduler passes the per-fire data window via env vars.
    start_time = os.environ.get("HOPS_START_TIME", "")
    end_time = os.environ.get("HOPS_END_TIME", "")
    print("Ingestion window: start=%s end=%s" % (start_time, end_time))

    # Discover newly-arrived increment files in the incoming directory.
    try:
        listing = dataset_api.list_files(INCREMENT_DIR, 0, 1000)
    except Exception as e:
        print("No incoming directory yet (%s): %s" % (INCREMENT_DIR, e))
        return

    import pandas as pd

    entries = listing[1] if isinstance(listing, tuple) else listing.get("items", [])
    inserted = 0
    for entry in entries:
        if isinstance(entry, dict):
            path = entry.get("attributes", {}).get("path")
        else:
            path = getattr(entry, "path", None)
        if not path or not path.endswith(".csv"):
            continue
        local = os.path.basename(path)
        dataset_api.download(path, local, overwrite=True)
        df = pd.read_csv(local)
        if len(df) == 0:
            continue
        fg.insert(df)
        inserted += len(df)
        print("Ingested %d rows from %s" % (len(df), path))

    print("Done. Total rows ingested this run: %d" % inserted)


if __name__ == "__main__":
    main()
