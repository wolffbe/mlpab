#!/usr/bin/env python3
"""
Create a feature table named `scoreda4f6e2`, version 1, in the specified schema.
Columns: request_id, account_id, distance_deg, score.
Enable online access for low-latency lookup.
"""

import os
import math
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import *
from databricks.sdk.service import sql
from databricks.sdk.service import files

# Load data
requests_df = pd.read_csv("data/requests.csv")
profiles_df = pd.read_csv("data/profiles.csv")

# Merge requests with profiles
merged_df = pd.merge(
    requests_df, 
    profiles_df, 
    on="account_id", 
    how="left"
)

# Compute on-demand features
def compute_distance_deg(row):
    lat_diff = row["request_lat"] - row["home_lat"]
    lon_diff = row["request_lon"] - row["home_lon"]
    distance = math.sqrt(lat_diff**2 + lon_diff**2)
    return round(distance, 6)

def compute_score(row):
    distance_deg = row["distance_deg"]
    score = row["base_score"] - 0.1 * distance_deg
    return round(score, 6)

merged_df["distance_deg"] = merged_df.apply(compute_distance_deg, axis=1)
merged_df["score"] = merged_df.apply(compute_score, axis=1)

# Select required columns
result_df = merged_df[["request_id", "account_id", "distance_deg", "score"]]

# Write to local CSV
local_csv_path = "staged_data.csv"
result_df.to_csv(local_csv_path, index=False)

# Initialize Databricks WorkspaceClient
w = WorkspaceClient()

# Schema and table details
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # e.g., workspace.mlpabceaa54
catalog_name, schema_name = schema_name.split(".")
table_name = "scoreda4f6e2"
full_table_name = f"{catalog_name}.{schema_name}.{table_name}"

# Create schema if it doesn't exist
try:
    w.schemas.create(
        name=schema_name,
        catalog_name=catalog_name,
    )
except Exception as e:
    print(f"Schema may already exist or error: {e}")

# Upload the CSV to DBFS
dbfs_path = f"dbfs:/FileStore/{os.getenv('MLPAB_DATABRICKS_PREFIX')}_staged_data.csv"
with open(local_csv_path, "rb") as f:
    w.dbfs.upload(dbfs_path, f, overwrite=True)

# Create the feature table from the staged data
try:
    w.tables.create(
        name=table_name,
        catalog_name=catalog_name,
        schema_name=schema_name,
        table_type=TableType.MANAGED,
        data_source_format=DataSourceFormat.DELTA,
        columns=[
            Column(name="request_id", type_name=ColumnTypeName.STRING, position=0, nullable=False),
            Column(name="account_id", type_name=ColumnTypeName.STRING, position=1, nullable=False),
            Column(name="distance_deg", type_name=ColumnTypeName.DOUBLE, position=2, nullable=False),
            Column(name="score", type_name=ColumnTypeName.DOUBLE, position=3, nullable=False),
        ],
        storage_location=f"dbfs:/user/hive/warehouse/{catalog_name}.db/{schema_name}/{table_name}",
    )
    
    # Load data into the table using SQL
    load_query = f"""
    COPY INTO {full_table_name}
    FROM '{dbfs_path}'
    FILEFORMAT = CSV
    FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true')
    """
    w.statement_execution.execute_statement(
        warehouse_id=w.warehouses.list()[0].id,
        catalog=catalog_name,
        schema=schema_name,
        statement=load_query,
    )
except Exception as e:
    print(f"Table creation or data load failed: {e}")
    raise

# Create an online table for low-latency access
try:
    w.online_tables.create(
        name=table_name,
        catalog_name=catalog_name,
        schema_name=schema_name,
        primary_key_columns=["request_id"],
        source_table_full_name=full_table_name,
    )
except Exception as e:
    print(f"Online table creation failed: {e}")
    raise

print(f"Feature table {full_table_name} created successfully with online access enabled.")