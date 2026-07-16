#!/usr/bin/env python3
"""
Register a feature table named `accounts5fc5d2`, version 1, on Hopsworks with record key `row_id`
and event-time column `updated_at` (epoch milliseconds). Load the batches so the table's final
contents are each row's LATEST revision — exactly one row per row_id. Enable online access.
"""

import hopsworks
import pandas as pd
import os

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Read batch files
batch_files = [
    "./data/batch_1.csv",
    "./data/batch_2.csv",
    "./data/batch_3.csv",
]

# Combine batches and keep the latest revision for each row_id
dfs = []
for file in batch_files:
    df = pd.read_csv(file)
    dfs.append(df)

combined_df = pd.concat(dfs)
latest_df = combined_df.sort_values(by="updated_at", ascending=False).drop_duplicates(subset=["row_id"], keep="first")

# Register feature table
accounts_fg = fs.create_feature_group(
    name="accounts5fc5d2",
    version=1,
    description="Accounts table with latest revisions for each row_id",
    primary_key=["row_id"],
    event_time="updated_at",
    online_enabled=True,
)

# Save the latest revisions
accounts_fg.insert(latest_df)

print("Feature table 'accounts5fc5d2', version 1, has been registered and populated with the latest revisions.")
print("Online access is enabled for low-latency lookup.")