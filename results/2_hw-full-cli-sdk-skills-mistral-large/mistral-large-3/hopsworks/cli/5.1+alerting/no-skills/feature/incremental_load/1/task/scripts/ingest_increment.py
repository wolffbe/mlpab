#!/usr/bin/env python3
"""
Daily ingestion script for incremental event data.
Expects a single argument: the path to the new increment CSV file.
"""

import hopsworks
import pandas as pd
import sys


def main():
    if len(sys.argv) != 2:
        print("Usage: ingest_increment.py <path_to_increment_csv>")
        sys.exit(1)

    increment_path = sys.argv[1]

    # Connect to Hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Read the increment
    df = pd.read_csv(increment_path)

    # Get or create the feature group
    fg = fs.get_or_create_feature_group(
        name="incremental97a30c",
        version=1,
        description="Feature group for incremental event data",
        primary_key=["row_id"],
        event_time="event_time",
        online_enabled=True,
    )

    # Insert the data
    fg.insert(df)


if __name__ == "__main__":
    main()