"""Daily incremental ingestion pipeline for feature group `incremental242d54`.

Runs on the platform on a daily schedule. Each fire, it looks for newly
arrived increment files in the HopsFS landing directory and appends their
rows to the feature group. The feature group's primary key (`row_id`) makes
the insert an upsert, so re-processing a file is safe (idempotent).

The data window for the fire is provided by the scheduler via the
HOPS_START_TIME / HOPS_END_TIME environment variables (epoch); the pipeline
uses them to filter rows by `event_time` when present.
"""
import os

import hopsworks


# HopsFS directory where new daily increment files land. New increments with
# the same schema as the seed files keep arriving here.
LANDING_DIR = "Resources/incremental242d54_increments"
FG_NAME = "incremental242d54"
FG_VERSION = 1


def _window_bounds():
    """Return (start_ms, end_ms) for this fire, or (None, None) if unset."""
    def _to_ms(v):
        if not v:
            return None
        v = int(float(v))
        # Normalize seconds -> milliseconds if it looks like epoch seconds.
        return v * 1000 if v < 10_000_000_000 else v

    return _to_ms(os.environ.get("HOPS_START_TIME")), _to_ms(
        os.environ.get("HOPS_END_TIME")
    )


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    dataset_api = project.get_dataset_api()

    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

    start_ms, end_ms = _window_bounds()
    print(f"Ingestion window: start={start_ms} end={end_ms}")

    # Discover increment files that have landed.
    try:
        entries = dataset_api.list_files(LANDING_DIR, 0, 1000)[1]
        paths = [e["attributes"]["path"] for e in entries]
    except Exception as exc:  # landing dir may not exist yet on first runs
        print(f"No landing directory yet ({LANDING_DIR}): {exc}")
        return

    csv_paths = [p for p in paths if p.endswith(".csv")]
    if not csv_paths:
        print("No new increment files to ingest.")
        return

    import pandas as pd

    total = 0
    for remote in csv_paths:
        local = os.path.basename(remote)
        dataset_api.download(remote, local, overwrite=True)
        df = pd.read_csv(local)
        if start_ms is not None and "event_time" in df.columns:
            df = df[df["event_time"] >= start_ms]
        if end_ms is not None and "event_time" in df.columns:
            df = df[df["event_time"] < end_ms]
        if len(df) == 0:
            continue
        fg.insert(df)
        total += len(df)
        print(f"Ingested {len(df)} rows from {remote}")

    print(f"Daily ingestion complete: {total} rows into {FG_NAME} v{FG_VERSION}")


if __name__ == "__main__":
    main()
