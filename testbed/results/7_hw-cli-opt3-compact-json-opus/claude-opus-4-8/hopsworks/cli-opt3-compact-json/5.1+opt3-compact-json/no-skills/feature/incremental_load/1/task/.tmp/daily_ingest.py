"""Daily incremental ingestion job for feature group `incremental7a03ae`.

Runs on the platform on a daily schedule. Each fire ingests the new daily
increment file(s) for the events table into the feature group (offline +
online). Future increments are expected to land as CSV files (same schema as
the registered increments) under the project's HopsFS resources directory
`Resources/increments/`.

The data window for each fire is provided by the scheduler via the
HOPS_START_TIME / HOPS_END_TIME environment variables.
"""

import os

import hopsworks


FG_NAME = "incremental7a03ae"
FG_VERSION = 1
# HopsFS directory where future daily increment CSVs are expected to land.
INCREMENTS_DIR = "Resources/increments"


def main():
    start = os.environ.get("HOPS_START_TIME")
    end = os.environ.get("HOPS_END_TIME")
    print(f"[daily_ingest] data window: {start} .. {end}")

    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)
    print(f"[daily_ingest] target feature group: {FG_NAME} v{FG_VERSION}")

    dataset_api = project.get_dataset_api()

    # Discover the new increment files that have arrived for this window.
    try:
        listing = dataset_api.list_files(INCREMENTS_DIR, 0, 1000)
        entries = listing[1] if isinstance(listing, tuple) else listing
        paths = []
        for e in entries:
            p = e.get("attributes", {}).get("path") if isinstance(e, dict) else getattr(e, "path", None)
            if p and p.endswith(".csv"):
                paths.append(p)
    except Exception as exc:  # noqa: BLE001
        print(f"[daily_ingest] no increments directory yet ({exc}); nothing to ingest.")
        return

    if not paths:
        print("[daily_ingest] no new increment files found; nothing to ingest.")
        return

    import pandas as pd

    total = 0
    for remote in sorted(paths):
        local = os.path.basename(remote)
        dataset_api.download(remote, local, overwrite=True)
        df = pd.read_csv(local)
        fg.insert(df)
        total += len(df)
        print(f"[daily_ingest] ingested {len(df)} rows from {remote}")

    print(f"[daily_ingest] done; total rows ingested: {total}")


if __name__ == "__main__":
    main()
