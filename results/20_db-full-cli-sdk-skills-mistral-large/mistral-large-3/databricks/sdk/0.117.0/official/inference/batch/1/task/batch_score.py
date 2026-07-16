#!/usr/bin/env python3
"""
Batch scoring script for Databricks.

Processes feature history, computes scores, and creates a feature table
named `scores4a1a3b` (version 1) in the specified schema.
"""

import json
import pandas as pd
import numpy as np
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog

# Load inputs
T = 1773234000000  # As-of timestamp

# Load feature history
feature_history = pd.read_csv("data/feature_history.csv")

# Load model
with open("data/model.json", "r") as f:
    model = json.load(f)

weights = model["weights"]
bias = model["bias"]

# Filter feature history to revisions at or before T
feature_history["event_time"] = pd.to_numeric(feature_history["event_time"])
valid_revisions = feature_history[feature_history["event_time"] <= T]

# For each account, select the most recent revision
most_recent_revisions = valid_revisions.sort_values("event_time").groupby("account_id").last().reset_index()

# Compute scores
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

most_recent_revisions["score"] = sigmoid(
    most_recent_revisions["f1"] * weights["f1"] +
    most_recent_revisions["f2"] * weights["f2"] +
    most_recent_revisions["f3"] * weights["f3"] +
    bias
)

# Round to 6 decimal places
most_recent_revisions["score"] = most_recent_revisions["score"].round(6)

# Prepare final DataFrame for table
scores_df = most_recent_revisions[["account_id", "score"]]

# Initialize Databricks WorkspaceClient
w = WorkspaceClient()

# Get catalog and schema from environment variables
catalog_name = os.getenv("MLPAB_DATABRICKS_CATALOG", "hive_metastore")
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")
full_schema_name = f"{catalog_name}.{schema_name}"

# Create a temporary Delta table in memory and register it
print(f"Creating table `{full_schema_name}.scores4a1a3b`...")

# Convert DataFrame to JSON for ingestion
scores_json = scores_df.to_dict(orient="records")

# Use SQL to create and insert data into the table
create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {full_schema_name}.scores4a1a3b (
    account_id STRING,
    score DOUBLE
)
USING DELTA
"""

warehouses = list(w.warehouses.list())
if not warehouses:
    raise ValueError("No warehouses available in the workspace.")

w.statement_execution.execute_statement(
    warehouse_id=warehouses[0].id,
    catalog=catalog_name,
    schema=schema_name,
    statement=create_table_sql
)

# Insert data into the table
for record in scores_json:
    insert_sql = f"""
    INSERT INTO {full_schema_name}.scores4a1a3b (account_id, score)
    VALUES ('{record['account_id']}', {record['score']})
    """
    w.statement_execution.execute_statement(
        warehouse_id=warehouses[0].id,
        catalog=catalog_name,
        schema=schema_name,
        statement=insert_sql
    )

# Enable synced table for low-latency lookup
print("Enabling synced table for low-latency lookup...")
w.statement_execution.execute_statement(
    warehouse_id=warehouses[0].id,
    catalog=catalog_name,
    schema=schema_name,
    statement=f"ALTER TABLE {full_schema_name}.scores4a1a3b SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)

# Create a streaming job to sync the table for low-latency access
sync_sql = f"""
CREATE OR REFRESH STREAMING TABLE {full_schema_name}.scores4a1a3b_sync
AS SELECT * FROM STREAM(read_change_feed("{full_schema_name}.scores4a1a3b", "LATEST"))
"""

w.statement_execution.execute_statement(
    warehouse_id=warehouses[0].id,
    catalog=catalog_name,
    schema=schema_name,
    statement=sync_sql
)

print("Batch scoring and table creation completed successfully.")