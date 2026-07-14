#!/usr/bin/env python3
"""
Final script to create the feature table.
Uses persistent tables instead of temp views.
"""
import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import PublishSpec, OnlineStore
from databricks.sdk.service.sql import Format

# Environment variables
MLPAB_DATABRICKS_SCHEMA = os.environ.get("MLPAB_DATABRICKS_SCHEMA", "workspace.mlpabca664c")
MLPAB_DATABRICKS_PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpabca664c")

# Parse schema
catalog, schema = MLPAB_DATABRICKS_SCHEMA.split(".")

# Table configuration
TABLE_NAME = "scaleda1a1c9"
ONLINE_STORE_NAME = f"{MLPAB_DATABRICKS_PREFIX}_online_store"
ONLINE_TABLE_NAME = f"{MLPAB_DATABRICKS_PREFIX}_{TABLE_NAME}_v1"
FULL_TABLE_NAME = f"{catalog}.{schema}.{TABLE_NAME}"

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Table name: {TABLE_NAME}")
print(f"Full table name: {FULL_TABLE_NAME}")

# Initialize workspace client
ws = WorkspaceClient()

# Step 1: Create schema
print("\n=== Step 1: Creating schema ===")
create_schema_sql = f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}"
response = ws.statement_execution.execute_statement(
    statement=create_schema_sql,
    warehouse_id="4dfab06c923fe3cc",
    wait_timeout="10s"
)
print(f"Schema creation: {response.status.state}")

# Step 2: Create tables from CSV files in the schema
print("\n=== Step 2: Creating tables from CSV files ===")

# Create table for training data
create_train_table_sql = f"""
CREATE OR REPLACE TABLE {catalog}.{schema}.train_raw (
    row_id STRING,
    f1 DOUBLE,
    f2 DOUBLE,
    f3 DOUBLE,
    f4 DOUBLE
) USING CSV
OPTIONS (path 'file:/Workspace/features_train.csv', header 'true', inferSchema 'true')
"""
response = ws.statement_execution.execute_statement(
    statement=create_train_table_sql,
    warehouse_id="4dfab06c923fe3cc",
    wait_timeout="15s"
)
print(f"Train table creation: {response.status.state}")
if response.status.error:
    print(f"Error: {response.status.error}")

# Create table for serving data
create_serve_table_sql = f"""
CREATE OR REPLACE TABLE {catalog}.{schema}.serve_raw (
    row_id STRING,
    f1 DOUBLE,
    f2 DOUBLE,
    f3 DOUBLE,
    f4 DOUBLE
) USING CSV
OPTIONS (path 'file:/Workspace/features_serve.csv', header 'true', inferSchema 'true')
"""
response = ws.statement_execution.execute_statement(
    statement=create_serve_table_sql,
    warehouse_id="4dfab06c923fe3cc",
    wait_timeout="15s"
)
print(f"Serve table creation: {response.status.state}")
if response.status.error:
    print(f"Error: {response.status.error}")

# Step 3: Compute statistics from training data
print("\n=== Step 3: Computing statistics from training data ===")

stats_sql = f"""
SELECT
    AVG(f1) as mean_f1,
    STDDEV_POP(f1) as std_f1,
    AVG(f2) as mean_f2,
    STDDEV_POP(f2) as std_f2,
    AVG(f3) as mean_f3,
    STDDEV_POP(f3) as std_f3,
    AVG(f4) as mean_f4,
    STDDEV_POP(f4) as std_f4
FROM {catalog}.{schema}.train_raw
"""
response = ws.statement_execution.execute_statement(
    statement=stats_sql,
    warehouse_id="4dfab06c923fe3cc",
    wait_timeout="10s",
    format=Format.JSON_ARRAY
)
print(f"Stats query: {response.status.state}")
if response.status.error:
    print(f"Error: {response.status.error}")
    exit(1)

if response.result and response.result.data_array:
    stats = response.result.data_array[0]
    print(f"Statistics: {stats}")
    mean_f1, std_f1 = float(stats[0]), float(stats[1])
    mean_f2, std_f2 = float(stats[2]), float(stats[3])
    mean_f3, std_f3 = float(stats[4]), float(stats[5])
    mean_f4, std_f4 = float(stats[6]), float(stats[7])
    print(f"f1: mean={mean_f1}, std={std_f1}")
    print(f"f2: mean={mean_f2}, std={std_f2}")
    print(f"f3: mean={mean_f3}, std={std_f3}")
    print(f"f4: mean={mean_f4}, std={std_f4}")
else:
    print("No statistics returned")
    exit(1)

# Step 4: Create the final table with standardized values
print("\n=== Step 4: Creating final table with standardized values ===")

create_final_table_sql = f"""
CREATE OR REPLACE TABLE {catalog}.{schema}.{TABLE_NAME} (
    row_id STRING,
    split STRING,
    f1 DOUBLE,
    f2 DOUBLE,
    f3 DOUBLE,
    f4 DOUBLE
) USING DELTA
"""
response = ws.statement_execution.execute_statement(
    statement=create_final_table_sql,
    warehouse_id="4dfab06c923fe3cc",
    wait_timeout="10s"
)
print(f"Final table creation: {response.status.state}")

# Insert training data (standardized)
insert_train_sql = f"""
INSERT INTO {catalog}.{schema}.{TABLE_NAME}
SELECT
    row_id,
    'train' as split,
    ROUND((f1 - {mean_f1}) / {std_f1}, 6) as f1,
    ROUND((f2 - {mean_f2}) / {std_f2}, 6) as f2,
    ROUND((f3 - {mean_f3}) / {std_f3}, 6) as f3,
    ROUND((f4 - {mean_f4}) / {std_f4}, 6) as f4
FROM {catalog}.{schema}.train_raw
"""
response = ws.statement_execution.execute_statement(
    statement=insert_train_sql,
    warehouse_id="4dfab06c923fe3cc",
    wait_timeout="15s"
)
print(f"Insert training data: {response.status.state}")
if response.status.error:
    print(f"Error: {response.status.error}")

# Insert serving data (standardized)
insert_serve_sql = f"""
INSERT INTO {catalog}.{schema}.{TABLE_NAME}
SELECT
    row_id,
    'serve' as split,
    ROUND((f1 - {mean_f1}) / {std_f1}, 6) as f1,
    ROUND((f2 - {mean_f2}) / {std_f2}, 6) as f2,
    ROUND((f3 - {mean_f3}) / {std_f3}, 6) as f3,
    ROUND((f4 - {mean_f4}) / {std_f4}, 6) as f4
FROM {catalog}.{schema}.serve_raw
"""
response = ws.statement_execution.execute_statement(
    statement=insert_serve_sql,
    warehouse_id="4dfab06c923fe3cc",
    wait_timeout="15s"
)
print(f"Insert serving data: {response.status.state}")
if response.status.error:
    print(f"Error: {response.status.error}")

# Step 5: Verify the table
print("\n=== Step 5: Verifying the table ===")

verify_sql = f"""
SELECT COUNT(*) as total_count,
       COUNT(CASE WHEN split = 'train' THEN 1 END) as train_count,
       COUNT(CASE WHEN split = 'serve' THEN 1 END) as serve_count,
       ROUND(AVG(CASE WHEN split = 'train' THEN f1 END), 6) as avg_f1_train,
       ROUND(AVG(CASE WHEN split = 'serve' THEN f1 END), 6) as avg_f1_serve
FROM {catalog}.{schema}.{TABLE_NAME}
"""
response = ws.statement_execution.execute_statement(
    statement=verify_sql,
    warehouse_id="4dfab06c923fe3cc",
    wait_timeout="10s",
    format=Format.JSON_ARRAY
)
print(f"Verification query: {response.status.state}")
if response.status.error:
    print(f"Error: {response.status.error}")
if response.result and response.result.data_array:
    print(f"Verification result: {response.result.data_array}")

# Step 6: Create online store
print("\n=== Step 6: Creating online store ===")

try:
    online_store = ws.feature_store.create_online_store(
        OnlineStore(
            name=ONLINE_STORE_NAME,
            capacity="SMALL"
        )
    )
    print(f"Created online store: {online_store.name} (ID: {online_store.id})")
except Exception as e:
    print(f"Error creating online store: {e}")
    # Check if it already exists
    stores = ws.feature_store.list_online_stores()
    existing_store = None
    for store in stores:
        if store.name == ONLINE_STORE_NAME:
            existing_store = store
            break
    if existing_store:
        print(f"Online store already exists: {existing_store.name} (ID: {existing_store.id})")
        online_store = existing_store
    else:
        raise

# Wait for online store to be ready
print("Waiting for online store to be ready...")
max_retries = 30
for i in range(max_retries):
    store_info = ws.feature_store.get_online_store(online_store.id)
    print(f"Store state: {store_info.state}")
    if store_info.state == "READY":
        break
    time.sleep(10)
else:
    print("Timeout waiting for online store to be ready")

# Step 7: Publish table to online store
print("\n=== Step 7: Publishing table to online store ===")

publish_spec = PublishSpec(
    online_store=online_store.id,
    online_table_name=ONLINE_TABLE_NAME,
    publish_mode="SNAPSHOT"
)

try:
    publish_response = ws.feature_store.publish_table(
        source_table_name=FULL_TABLE_NAME,
        publish_spec=publish_spec
    )
    print(f"Publish response: {publish_response}")
    print("Table published to online store successfully!")
except Exception as e:
    print(f"Error publishing table: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Done ===")
