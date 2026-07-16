#!/usr/bin/env python3
"""
Create a feature table named `scoreda4f6e2`, version 1, in the specified schema.
Columns: request_id, account_id, distance_deg, score.
Enable online access for low-latency lookup.

This script:
1. Computes the features locally (allowed for small data).
2. Creates the feature table directly using the Databricks SDK.
3. Enables online access.
"""

import os
import math
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import *

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

# Create the feature table with online access
try:
    # Create the table
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
    
    # Write data to the table using SQL
    from databricks.sdk.service import sql
    
    # Create a temporary view for the DataFrame
    spark_df = w.sc.get_or_create().createDataFrame(result_df)
    spark_df.createOrReplaceTempView("temp_feature_data")
    
    # Insert data into the table
    insert_query = f"""
    INSERT INTO {full_table_name}
    SELECT * FROM temp_feature_data
    """
    w.statement_execution.execute_statement(
        warehouse_id=w.warehouses.list()[0].id,
        catalog=catalog_name,
        schema=schema_name,
        statement=insert_query,
    )
    
    # Enable online access
    w.online_tables.create(
        name=table_name,
        catalog_name=catalog_name,
        schema_name=schema_name,
        primary_key_columns=["request_id"],
        source_table_full_name=full_table_name,
    )
    
    print(f"Feature table {full_table_name} created successfully with online access enabled.")
    
except Exception as e:
    print(f"Failed to create feature table: {e}")
    raise