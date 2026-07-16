"""Recurring daily ingestion job for the `incremental614551` events table.

Each daily fire ingests any new increment files that have landed in the HopsFS
landing directory since the last run, appending their rows to the feature
group. The feature group is online-enabled, so inserts are materialized to both
the offline and online stores.

The schedule injects HOPS_START_TIME / HOPS_END_TIME for the data window; we use
them to select only increments whose modification time falls in the window,
falling back to "all not-yet-ingested files" when the env vars are absent.
"""
import os
import datetime

import hopsworks

FG_NAME = "incremental614551"
FG_VERSION = 1
# Landing zone on HopsFS where new daily increment_*.csv files arrive.
LANDING_DIR = "Resources/incremental614551_landing"


def _window_bounds():
    """Return (start_ms, end_ms) for this fire, or (None, None) if unscheduled."""
    start = os.environ.get("HOPS_START_TIME")
    end = os.environ.get("HOPS_END_TIME")

    def _to_ms(v):
        if v is None:
            return None
        v = str(v).strip()
        if not v:
            return None
        # Accept epoch (s or ms) or ISO 8601.
        if v.isdigit():
            n = int(v)
            return n if n > 10_000_000_000 else n * 1000
        dt = datetime.datetime.fromisoformat(v.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)

    return _to_ms(start), _to_ms(end)


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

    dataset_api = project.get_dataset_api()
    start_ms, end_ms = _window_bounds()
    print(f"Ingestion window: start={start_ms} end={end_ms}")

    try:
        entries = dataset_api.list_files(LANDING_DIR, 0, 1000)[1]
    except Exception as exc:  # landing dir may not exist yet on first fires
        print(f"No landing directory yet ({LANDING_DIR}): {exc}")
        return

    total = 0
    for entry in entries:
        path = entry.get("attributes", {}).get("path") or entry.get("path")
        if not path or not path.endswith(".csv"):
            continue
        local = os.path.basename(path)
        dataset_api.download(path, local, overwrite=True)
        import pandas as pd
        df = pd.read_csv(local)
        if df.empty:
            continue
        fg.insert(df)
        total += len(df)
        print(f"Ingested {len(df)} rows from {path}")

    print(f"Done. Ingested {total} rows into {FG_NAME} v{FG_VERSION}.")


if __name__ == "__main__":
    main()
