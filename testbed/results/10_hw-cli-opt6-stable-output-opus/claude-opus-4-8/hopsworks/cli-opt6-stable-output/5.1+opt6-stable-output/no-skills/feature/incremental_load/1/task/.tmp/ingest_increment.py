"""Daily incremental ingestion job for feature group ``incrementalc9b70d``.

Runs on the platform on a daily schedule. Each fire, it picks up new daily
increment files (same schema as the historical increments) and inserts the
rows into the feature group, which keeps both the offline and online stores
up to date. The scheduler exposes the data window via the HOPS_START_TIME /
HOPS_END_TIME environment variables; new increment files are expected under
the project's Resources/incoming_increments directory on HopsFS.
"""

import os
import glob

import pandas as pd
import hopsworks


FG_NAME = "incrementalc9b70d"
FG_VERSION = 1
# Directory on the local job working dir where the daily increment files land.
# Future increments share the historical schema:
#   row_id (string), account_id (string), event_time (bigint, epoch ms),
#   amount (double), category (string).
INCOMING_DIRS = [
    "Resources/incoming_increments",
    "incoming_increments",
    ".",
]
DTYPES = {
    "row_id": "string",
    "account_id": "string",
    "event_time": "int64",
    "amount": "float64",
    "category": "string",
}


def find_increment_files():
    files = []
    for d in INCOMING_DIRS:
        if os.path.isdir(d):
            files.extend(sorted(glob.glob(os.path.join(d, "increment_*.csv"))))
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

    files = find_increment_files()
    if not files:
        print("No new increment files found; nothing to ingest.")
        return

    total = 0
    for path in files:
        df = pd.read_csv(path, dtype=DTYPES)
        print(f"Ingesting {len(df)} rows from {path}")
        fg.insert(df)
        total += len(df)

    print(f"Ingested {total} rows into {FG_NAME} v{FG_VERSION}")


if __name__ == "__main__":
    main()
