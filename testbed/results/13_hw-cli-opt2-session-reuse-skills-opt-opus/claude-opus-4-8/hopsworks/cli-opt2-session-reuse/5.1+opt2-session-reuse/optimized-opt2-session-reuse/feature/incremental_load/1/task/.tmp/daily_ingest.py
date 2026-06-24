"""Daily feature pipeline for the `incremental365a10` events feature group.

Runs on a daily schedule. Each fire, it picks up the newly-arrived daily
increment file (same schema as the historical increments: row_id, account_id,
event_time, amount, category) and inserts it into the feature group. New rows
land in both the offline store (for training / time-travel) and the online
store (for low-latency lookup), since the feature group is online-enabled.

The increment for a given day is expected under the HopsFS directory
``Resources/incremental365a10/`` as ``increment_<YYYY-MM-DD>.csv``. The run
date is taken from the scheduler-provided ``HOPS_END_TIME`` (the data-window
end for this fire) and falls back to today's date for manual runs.
"""
import os
import datetime

import pandas as pd
import hopsworks

FG_NAME = "incremental365a10"
FG_VERSION = 1
INCREMENT_DIR = "Resources/incremental365a10"


def _run_date():
    end = os.environ.get("HOPS_END_TIME")
    if end:
        # HOPS_END_TIME is epoch milliseconds for the data-window end.
        try:
            ts = int(end)
            return datetime.datetime.utcfromtimestamp(ts / 1000.0).date()
        except (ValueError, TypeError):
            pass
    return datetime.date.today()


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

    run_date = _run_date()
    fname = f"increment_{run_date.isoformat()}.csv"
    hdfs_path = f"{INCREMENT_DIR}/{fname}"

    dataset_api = project.get_dataset_api()
    local_path = dataset_api.download(hdfs_path, overwrite=True)

    df = pd.read_csv(local_path)
    df["row_id"] = df["row_id"].astype(str)
    df["account_id"] = df["account_id"].astype(str)
    df["event_time"] = df["event_time"].astype("int64")
    df["amount"] = df["amount"].astype("float64")
    df["category"] = df["category"].astype(str)

    fg.insert(df)
    print(f"Ingested {len(df)} rows from {hdfs_path} into {FG_NAME} v{FG_VERSION}")


if __name__ == "__main__":
    main()
