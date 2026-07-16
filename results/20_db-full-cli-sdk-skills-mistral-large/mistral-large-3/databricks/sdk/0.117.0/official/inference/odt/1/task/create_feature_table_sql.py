#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql, catalog

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
table_name = f"{schema_name}.scoreda4f6e2"

# Use SQL to create the feature table
create_query = f"""
CREATE OR REPLACE TABLE {table_name} (
    request_id STRING,
    account_id STRING,
    distance_deg DOUBLE,
    score DOUBLE
)
USING DELTA
"""

w.statement_execution.execute_statement(
    warehouse_id="4dfab06c923fe3cc",  # Use the available warehouse ID
    catalog=schema_name.split(".")[0],
    schema=schema_name.split(".")[1],
    statement=create_query
)

# Use SQL to insert data into the feature table
for _, row in result_df.iterrows():
    insert_query = f"""
    INSERT INTO {table_name} (request_id, account_id, distance_deg, score)
    VALUES ('{row['request_id']}', '{row['account_id']}', {row['distance_deg']}, {row['score']})
    """
    w.statement_execution.execute_statement(
        warehouse_id="4dfab06c923fe3cc",  # Use the available warehouse ID
        catalog=schema_name.split(".")[0],
        schema=schema_name.split(".")[1],
        statement=insert_query
    )

# Enable online table for low-latency lookup
online_table_name = f"{schema_name}.scoreda4f6e2_online"
online_query = f"""
CREATE OR REFRESH LIVE TABLE {online_table_name} AS
SELECT * FROM {table_name}
"""

w.statement_execution.execute_statement(
    warehouse_id="4dfab06c923fe3cc",  # Use the available warehouse ID
    catalog=schema_name.split(".")[0],
    schema=schema_name.split(".")[1],
    statement=online_query
)

print(f"Feature table {table_name} created successfully.")
print(f"Online table {online_table_name} enabled for low-latency lookup.")