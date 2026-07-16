#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog, sql

# Read input files
requests_df = pd.read_csv("data/requests.csv")
profiles_df = pd.read_csv("data/profiles.csv")

# Merge requests and profiles on account_id
merged_df = pd.merge(requests_df, profiles_df, on="account_id", how="left")

# Compute distance_deg
def calculate_distance_deg(row):
    lat_diff = row["request_lat"] - row["home_lat"]
    lon_diff = row["request_lon"] - row["home_lon"]
    distance = np.sqrt(lat_diff**2 + lon_diff**2)
    return round(distance, 6)

merged_df["distance_deg"] = merged_df.apply(calculate_distance_deg, axis=1)

# Compute score
merged_df["score"] = merged_df.apply(
    lambda row: round(row["base_score"] - 0.1 * row["distance_deg"], 6), axis=1
)

# Select required columns
result_df = merged_df[["request_id", "account_id", "distance_deg", "score"]]

# Initialize Databricks WorkspaceClient
w = WorkspaceClient()

# Define schema and table name
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")
table_name = "scoreda4f6e2"
full_table_name = f"{schema_name}.{table_name}"

# Write to Delta table
spark_df = w.create_dataframe(result_df)
spark_df.write.save_as_table(
    name=full_table_name,
    mode="overwrite",
    format="delta"
)

# Enable online table for low-latency lookup
online_table_name = f"{table_name}_online"
w.online_tables.create(
    name=online_table_name,
    spec=catalog.OnlineTableSpec(
        primary_key_columns=["request_id"],
        source_table_full_name=full_table_name,
        run_trigger=catalog.OnlineTableSpecTriggerType.ON_DEMAND
    )
)

print(f"Feature table {full_table_name} created successfully.")
print(f"Online table {online_table_name} enabled for low-latency lookup.")