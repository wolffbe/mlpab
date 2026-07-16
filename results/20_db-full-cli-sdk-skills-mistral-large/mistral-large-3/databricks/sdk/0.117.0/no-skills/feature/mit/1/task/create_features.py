#!/usr/bin/env python3
"""
Create the derived feature table `featuresfbc05f` (version 1) on Databricks.
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql
from databricks.sdk.service.catalog import *

# Initialize WorkspaceClient
w = WorkspaceClient()

# Environment variables
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")
prefix = os.getenv("MLPAB_DATABRICKS_PREFIX")
catalog_name = schema_name.split(".")[0]
schema = schema_name.split(".")[1]

# Ensure the schema exists
try:
    w.schemas.get(f"{catalog_name}.{schema}")
except Exception:
    w.schemas.create(name=schema, catalog_name=catalog_name)

# Get the first available SQL Warehouse
warehouses = list(w.warehouses.list())
if not warehouses:
    raise Exception("No SQL Warehouses available.")
warehouse = warehouses[0]
warehouse_id = warehouse.id

# Create tables for transactions and fx_rates
transactions_table = f"{catalog_name}.{schema}.transactions"
fx_rates_table = f"{catalog_name}.{schema}.fx_rates"

# Create a Volume for input files
volume_name = f"{prefix}_volume"
volume_path = f"/Volumes/{catalog_name}/{schema}/{volume_name}"

try:
    w.volumes.create(
        name=volume_name,
        catalog_name=catalog_name,
        schema_name=schema,
        volume_type=VolumeType.MANAGED,
    )
except Exception:
    pass  # Volume may already exist

# Upload data to the Volume
transactions_volume_path = f"{volume_path}/transactions.csv"
fx_rates_volume_path = f"{volume_path}/fx_rates.csv"

with open("data/transactions.csv", "rb") as f:
    w.files.upload(transactions_volume_path, f, overwrite=True)

with open("data/fx_rates.csv", "rb") as f:
    w.files.upload(fx_rates_volume_path, f, overwrite=True)

# Create transactions table
w.statement_execution.execute_statement(
    warehouse_id=warehouse_id,
    catalog=catalog_name,
    schema=schema,
    statement=f"""
    CREATE TABLE IF NOT EXISTS {transactions_table} (
        row_id STRING,
        account_id STRING,
        event_time TIMESTAMP,
        amount DOUBLE,
        currency STRING
    )
    USING CSV
    OPTIONS (
        path "{transactions_volume_path}",
        header "true",
        inferSchema "true"
    )
    """,
)

# Create fx_rates table
w.statement_execution.execute_statement(
    warehouse_id=warehouse_id,
    catalog=catalog_name,
    schema=schema,
    statement=f"""
    CREATE TABLE IF NOT EXISTS {fx_rates_table} (
        currency STRING,
        fx_rate DOUBLE
    )
    USING CSV
    OPTIONS (
        path "{fx_rates_volume_path}",
        header "true",
        inferSchema "true"
    )
    """,
)

# Compute features and save as featuresfbc05f
feature_table = f"{catalog_name}.{schema}.featuresfbc05f"

query = f"""
WITH joined_data AS (
    SELECT
        t.row_id,
        t.account_id,
        t.event_time,
        t.amount * f.fx_rate AS amount_usd,
        t.amount,
        t.currency
    FROM {transactions_table} t
    JOIN {fx_rates_table} f ON t.currency = f.currency
),
weekend_flag AS (
    SELECT
        row_id,
        CASE
            WHEN DAYOFWEEK(event_time) IN (1, 7) THEN 1  -- 1=Sunday, 7=Saturday
            ELSE 0
        END AS is_weekend
    FROM joined_data
),
seven_day_sum AS (
    SELECT
        j.row_id,
        j.account_id,
        j.event_time,
        SUM(j.amount) OVER (
            PARTITION BY j.account_id
            ORDER BY j.event_time
            RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND CURRENT ROW
        ) AS amount_7d
    FROM joined_data j
)
SELECT
    j.row_id,
    j.account_id,
    UNIX_TIMESTAMP(j.event_time) * 1000 AS event_time,  -- Convert to epoch milliseconds
    j.amount_usd,
    w.is_weekend,
    s.amount_7d
FROM joined_data j
JOIN weekend_flag w ON j.row_id = w.row_id
JOIN seven_day_sum s ON j.row_id = s.row_id
"""

w.statement_execution.execute_statement(
    warehouse_id=warehouse_id,
    catalog=catalog_name,
    schema=schema,
    statement=f"CREATE OR REPLACE TABLE {feature_table} AS {query}",
)

print(f"Feature table {feature_table} created successfully.")