#!/usr/bin/env python3
"""
Register a feature table named `accounts7b3169`, version 1, in the schema provided by MLPAB_DATABRICKS_SCHEMA.
Load the three batch files (data/batch_1.csv, data/batch_2.csv, data/batch_3.csv) into the table,
ensuring the latest revision for each `row_id` is retained.
Enable online/real-time access for low-latency lookup.
"""

import os
from databricks.sdk import WorkspaceClient

# Environment variables
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # e.g., "workspace.mlpab61803d"
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]    # e.g., "mlpab61803d"
CATALOG, SCHEMA_NAME = SCHEMA.split(".")

# Feature table details
FEATURE_TABLE_NAME = "accounts7b3169"
FEATURE_TABLE_FULL_NAME = f"{CATALOG}.{SCHEMA_NAME}.{FEATURE_TABLE_NAME}"
PRIMARY_KEY = "row_id"
EVENT_TIME_COLUMN = "updated_at"
BATCH_FILES = [
    "./data/batch_1.csv",
    "./data/batch_2.csv",
    "./data/batch_3.csv",
]

# Initialize WorkspaceClient
w = WorkspaceClient()

# Get the first available warehouse
warehouse = next(w.warehouses.list(), None)
if not warehouse:
    raise Exception("No warehouses available in the workspace.")

# Step 1: Create a Delta table to hold the raw data
print("Creating Delta table for raw data...")
raw_table_name = f"{CATALOG}.{SCHEMA_NAME}.{PREFIX}_raw_accounts"

w.statement_execution.execute_statement(
    catalog=CATALOG,
    schema=SCHEMA_NAME,
    warehouse_id=warehouse.id,
    statement=f"""
    CREATE TABLE IF NOT EXISTS {raw_table_name} (
        row_id STRING,
        status STRING,
        balance DOUBLE,
        updated_at LONG
    )
    USING DELTA
    """
)

# Step 2: Load batch files into the raw table
print("Loading batch files into raw table...")
for batch_file in BATCH_FILES:
    w.statement_execution.execute_statement(
        catalog=CATALOG,
        schema=SCHEMA_NAME,
        warehouse_id=warehouse.id,
        statement=f"""
        COPY INTO {raw_table_name}
        FROM '{batch_file}'
        FILEFORMAT = CSV
        FORMAT_OPTIONS('header' = 'true', 'inferSchema' = 'true')
        """
    )

# Step 3: Create a Delta table for the latest revisions
print("Creating Delta table for latest revisions...")
w.statement_execution.execute_statement(
    catalog=CATALOG,
    schema=SCHEMA_NAME,
    warehouse_id=warehouse.id,
    statement=f"""
    CREATE OR REPLACE TABLE {FEATURE_TABLE_FULL_NAME} AS
    WITH ranked_rows AS (
        SELECT
            *,
            ROW_NUMBER() OVER (PARTITION BY {PRIMARY_KEY} ORDER BY {EVENT_TIME_COLUMN} DESC) as rn
        FROM {raw_table_name}
    )
    SELECT
        row_id, status, balance, updated_at
    FROM ranked_rows
    WHERE rn = 1
    """
)

# Step 4: Enable online store for low-latency lookup
print("Enabling online store...")
w.statement_execution.execute_statement(
    catalog=CATALOG,
    schema=SCHEMA_NAME,
    warehouse_id=warehouse.id,
    statement=f"""
    CREATE OR REFRESH ONLINE TABLE {FEATURE_TABLE_FULL_NAME}_online
    FROM {FEATURE_TABLE_FULL_NAME};
    """
)

print(f"Feature table {FEATURE_TABLE_FULL_NAME} and online store created successfully.")