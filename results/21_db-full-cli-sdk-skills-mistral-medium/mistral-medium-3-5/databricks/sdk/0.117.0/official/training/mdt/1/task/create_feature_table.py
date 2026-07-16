#!/usr/bin/env python3
"""
Script to create a feature table with standardized values from training and serving splits.
"""
import os
import csv
import json
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode, OnlineStore

# Environment variables
MLPAB_DATABRICKS_SCHEMA = os.environ.get("MLPAB_DATABRICKS_SCHEMA", "workspace.mlpabca664c")
MLPAB_DATABRICKS_PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpabca664c")

# Parse schema
catalog, schema = MLPAB_DATABRICKS_SCHEMA.split(".")

# Table configuration
TABLE_NAME = "scaleda1a1c9"
TABLE_VERSION = 1
ONLINE_STORE_NAME = f"{MLPAB_DATABRICKS_PREFIX}_online_store"
ONLINE_TABLE_NAME = f"{MLPAB_DATABRICKS_PREFIX}_{TABLE_NAME}_v{TABLE_VERSION}"
FULL_TABLE_NAME = f"{catalog}.{schema}.{TABLE_NAME}"

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Table name: {TABLE_NAME}")
print(f"Full table name: {FULL_TABLE_NAME}")
print(f"Online store name: {ONLINE_STORE_NAME}")
print(f"Online table name: {ONLINE_TABLE_NAME}")

# Initialize workspace client
ws = WorkspaceClient()

# Step 1: Read and process the CSV files
print("\n=== Step 1: Reading CSV files ===")

def read_csv(file_path):
    """Read CSV file and return list of dicts."""
    data = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields to float
            for key in ['f1', 'f2', 'f3', 'f4']:
                row[key] = float(row[key])
            data.append(row)
    return data

# Read training data
train_data = read_csv("data/features_train.csv")
print(f"Read {len(train_data)} training rows")

# Read serving data
serve_data = read_csv("data/features_serve.csv")
print(f"Read {len(serve_data)} serving rows")

# Step 2: Compute statistics from training data only
print("\n=== Step 2: Computing statistics from training data ===")

features = ['f1', 'f2', 'f3', 'f4']
stats = {}

for feature in features:
    values = [row[feature] for row in train_data]
    n = len(values)
    mean = sum(values) / n
    # Population standard deviation (no Bessel's correction)
    variance = sum((x - mean) ** 2 for x in values) / n
    std = variance ** 0.5
    stats[feature] = {'mean': mean, 'std': std}
    print(f"{feature}: mean={mean:.6f}, std={std:.6f}")

# Step 3: Standardize both splits
print("\n=== Step 3: Standardizing data ===")

def standardize_value(value, mean, std):
    """Standardize a value: (x - mean) / std, rounded to 6 decimals."""
    if std == 0:
        return 0.0  # Avoid division by zero
    standardized = (value - mean) / std
    return round(standardized, 6)

# Process training data
standardized_train = []
for row in train_data:
    std_row = {
        'row_id': row['row_id'],
        'split': 'train',
        'f1': standardize_value(row['f1'], stats['f1']['mean'], stats['f1']['std']),
        'f2': standardize_value(row['f2'], stats['f2']['mean'], stats['f2']['std']),
        'f3': standardize_value(row['f3'], stats['f3']['mean'], stats['f3']['std']),
        'f4': standardize_value(row['f4'], stats['f4']['mean'], stats['f4']['std'])
    }
    standardized_train.append(std_row)

# Process serving data
standardized_serve = []
for row in serve_data:
    std_row = {
        'row_id': row['row_id'],
        'split': 'serve',
        'f1': standardize_value(row['f1'], stats['f1']['mean'], stats['f1']['std']),
        'f2': standardize_value(row['f2'], stats['f2']['mean'], stats['f2']['std']),
        'f3': standardize_value(row['f3'], stats['f3']['mean'], stats['f3']['std']),
        'f4': standardize_value(row['f4'], stats['f4']['mean'], stats['f4']['std'])
    }
    standardized_serve.append(std_row)

# Combine all data
all_data = standardized_train + standardized_serve
print(f"Total rows: {len(all_data)}")

# Step 4: Upload data to DBFS
print("\n=== Step 4: Uploading data to DBFS ===")

# Create a temporary CSV file
temp_csv_path = "/tmp/standardized_features.csv"
with open(temp_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['row_id', 'split', 'f1', 'f2', 'f3', 'f4'])
    writer.writeheader()
    for row in all_data:
        writer.writerow(row)

print(f"Created temporary CSV: {temp_csv_path}")

# Upload to DBFS
dbfs_path = f"dbfs:/tmp/{MLPAB_DATABRICKS_PREFIX}_standardized_features.csv"
with open(temp_csv_path, 'r') as f:
    ws.dbfs.upload(dbfs_path, f, overwrite=True)
print(f"Uploaded to DBFS: {dbfs_path}")

# Clean up temp file
os.remove(temp_csv_path)

# Step 5: Create the table in Unity Catalog using SQL
print("\n=== Step 5: Creating table in Unity Catalog ===")

# First, ensure the schema exists
create_schema_sql = f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}"
print(f"Executing: {create_schema_sql}")
response = ws.statement_execution.execute_statement(
    statement=create_schema_sql,
    warehouse_id="4dfab06c923fe3cc",
    wait_timeout="10s"
)
print(f"Schema creation response: {response.status.state}")

# Create the table using COPY INTO or CREATE TABLE AS SELECT
# We'll use CREATE TABLE AS SELECT from the uploaded CSV
create_table_sql = f"""
CREATE OR REPLACE TABLE {catalog}.{schema}.{TABLE_NAME} (
    row_id STRING,
    split STRING,
    f1 DOUBLE,
    f2 DOUBLE,
    f3 DOUBLE,
    f4 DOUBLE
) USING DELTA
"""
print(f"Executing: {create_table_sql}")
response = ws.statement_execution.execute_statement(
    statement=create_table_sql,
    warehouse_id="4dfab06c923fe3cc",
    wait_timeout="10s"
)
print(f"Table creation response: {response.status.state}")

# Load data from CSV
# First copy to a temp location, then insert
copy_sql = f"""
COPY INTO {catalog}.{schema}.{TABLE_NAME}
FROM '/tmp/{MLPAB_DATABRICKS_PREFIX}_standardized_features.csv'
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'false', 'delimiter' = ',')
"""
print(f"Executing: {copy_sql}")
response = ws.statement_execution.execute_statement(
    statement=copy_sql,
    warehouse_id="4dfab06c923fe3cc",
    wait_timeout="15s"
)
print(f"COPY INTO response: {response.status.state}")

# Verify the table
select_sql = f"SELECT COUNT(*) as count, split, AVG(f1) as avg_f1 FROM {catalog}.{schema}.{TABLE_NAME} GROUP BY split"
print(f"\nExecuting verification query: {select_sql}")
response = ws.statement_execution.execute_statement(
    statement=select_sql,
    warehouse_id="4dfab06c923fe3cc",
    wait_timeout="10s",
    format="JSON_ARRAY"
)
print(f"Verification response state: {response.status.state}")
if response.result:
    print(f"Result: {response.result.data_array}")

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
