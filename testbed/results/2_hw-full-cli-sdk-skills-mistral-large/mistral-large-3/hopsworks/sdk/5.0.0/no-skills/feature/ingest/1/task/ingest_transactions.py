#!/usr/bin/env python3
"""
Script to register and ingest data into a feature table named `transactions342d2a`,
version 1, on the Hopsworks platform.

The script:
1. Connects to Hopsworks using environment variables for authentication.
2. Reads the two CSV files (transactions_export_1.csv and transactions_export_2.csv).
3. Deduplicates rows based on `row_id` to ensure no duplicates are ingested.
4. Registers the feature table with record key `row_id` and event-time column `event_time`.
5. Ingests the data into the feature table.
6. Enables online access for low-latency lookups.
"""

import os
import pandas as pd
import hopsworks


def main():
    # Connect to Hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Read the CSV files
    df1 = pd.read_csv("data/transactions_export_1.csv")
    df2 = pd.read_csv("data/transactions_export_2.csv")
    
    # Combine and deduplicate based on `row_id`
    combined_df = pd.concat([df1, df2])
    combined_df = combined_df.drop_duplicates(subset=["row_id"], keep="first")

    # Register the feature table
    transactions_fg = fs.create_feature_group(
        name="transactions342d2a",
        version=1,
        description="Transactions feature table with record key `row_id` and event-time column `event_time`.",
        primary_key=["row_id"],
        event_time="event_time",
        online_enabled=True,  # Enable online access
    )

    # Ingest the data
    transactions_fg.insert(combined_df)

    print("Feature table 'transactions342d2a', version 1, has been registered and ingested successfully.")
    print("Online access is enabled for low-latency lookups.")


if __name__ == "__main__":
    main()