#!/usr/bin/env python3
"""
Script to register and ingest data into a Hopsworks feature group.
"""

import hopsworks
import pandas as pd
import os

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Read the data files
df1 = pd.read_csv("data/transactions_export_1.csv")
df2 = pd.read_csv("data/transactions_export_2.csv")

# Combine the dataframes and drop duplicates based on row_id
combined_df = pd.concat([df1, df2]).drop_duplicates(subset=["row_id"], keep="first")

# Create the feature group
transactions_fg = fs.create_feature_group(
    name="transactions342d2a",
    version=1,
    description="Transactions feature group with deduplicated rows from overlapping exports",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,  # Enable online access
)

# Ingest the data
transactions_fg.insert(combined_df)

print("Feature group 'transactions342d2a' version 1 has been created and data ingested successfully.")