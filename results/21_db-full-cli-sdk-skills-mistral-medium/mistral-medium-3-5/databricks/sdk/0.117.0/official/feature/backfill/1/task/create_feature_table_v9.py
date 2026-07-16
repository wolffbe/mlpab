#!/usr/bin/env python3
"""
Create feature table accounts9ad208 with latest revisions from batch files.
"""
import os
import csv
import io

# Environment variables
SCHEMA = os.getenv('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpabc1ee89')
PREFIX = os.getenv('MLPAB_DATABRICKS_PREFIX', 'mlpabc1ee89')

# Parse schema - it's in format catalog.schema
catalog_name, schema_name = SCHEMA.split('.')

print(f"Catalog: {catalog_name}, Schema: {schema_name}")
print(f"Prefix: {PREFIX}")

from databricks.sdk import WorkspaceClient

# Initialize workspace client
w = WorkspaceClient()

# Step 1: Ensure schema exists - just check, don't create
try:
    w.schemas.get(full_name=SCHEMA)
    print(f"Schema {SCHEMA} exists")
except Exception as e:
    print(f"Schema check failed: {e}")

# Step 2: Read all CSV files and combine them in memory
print("\nReading CSV files...")
all_rows = []
for batch_file in ['batch_1.csv', 'batch_2.csv', 'batch_3.csv']:
    local_path = f"data/{batch_file}"
    with open(local_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows.append(row)
    print(f"  Read {len(all_rows)} rows from {batch_file}")

# Step 3: Deduplicate in Python - keep latest updated_at per row_id
print("\nDeduplicating...")
latest_rows = {}
for row in all_rows:
    row_id = row['row_id']
    updated_at = int(row['updated_at'])
    
    if row_id not in latest_rows or updated_at > latest_rows[row_id]['updated_at']:
        latest_rows[row_id] = row

print(f"  Kept {len(latest_rows)} unique row_ids")

# Step 4: Create a Delta table
feature_table_name = "accounts9ad208"
full_table_name = f"{catalog_name}.{schema_name}.{feature_table_name}"

print(f"\nCreating Delta table: {full_table_name}")

# Create the table
w.sql.execute(
    warehouse_id="default",
    statement=f"""
    CREATE TABLE IF NOT EXISTS {full_table_name} (
      row_id STRING,
      status STRING,
      balance DOUBLE,
      updated_at BIGINT
    )
    USING DELTA
    COMMENT 'Feature table with latest account revisions'
    """
)
print(f"Table {full_table_name} created")

# Step 5: Insert data using VALUES clause
# We need to build a large INSERT statement
print("\nInserting data...")

# Build VALUES clauses in batches to avoid SQL length limits
batch_size = 50
values_clauses = []
for row_id, row in latest_rows.items():
    values_clauses.append(
        f"('{row_id}', '{row['status']}', {row['balance']}, {row['updated_at']})"
    )

# Insert in batches
for i in range(0, len(values_clauses), batch_size):
    batch = values_clauses[i:i+batch_size]
    values_str = ", ".join(batch)
    w.sql.execute(
        warehouse_id="default",
        statement=f"""
        INSERT INTO TABLE {full_table_name}
        VALUES {values_str}
        """
    )
    print(f"  Inserted batch {i//batch_size + 1}")

print("Data loaded")

# Verify the data
print("\nVerifying data...")
result = w.sql.execute(
    warehouse_id="default",
    statement=f"SELECT COUNT(*) as cnt, COUNT(DISTINCT row_id) as distinct_ids FROM {full_table_name}"
)

for row in result:
    print(f"Total rows: {row.cnt}, Distinct row_ids: {row.distinct_ids}")

# Step 6: Register as a feature table and enable online access
# In Databricks, we publish the table to an online store for low-latency access

print("\nSetting up online access...")

# First, check if we need to create an online store
# List existing online stores
try:
    online_stores = w.feature_store.list_online_stores()
    print(f"Existing online stores: {[s.name for s in online_stores]}")
    online_store_name = online_stores[0].name if online_stores else None
except Exception as e:
    print(f"Error listing online stores: {e}")
    online_store_name = None

# If no online store exists, create one
if not online_store_name:
    online_store_name = f"{PREFIX}_online_store"
    print(f"Creating online store: {online_store_name}")
    try:
        w.feature_store.create_online_store(
            name=online_store_name,
            capacity="SMALL"
        )
        print(f"Online store {online_store_name} created")
    except Exception as e:
        print(f"Error creating online store: {e}")
        # Try to use default
        online_store_name = "default"

# Now publish the table to the online store
# This makes it available for low-latency lookup
online_table_name = f"{PREFIX}_accounts9ad208_online"

print(f"\nPublishing to online store...")
try:
    from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode
    
    publish_spec = PublishSpec(
        online_store=online_store_name,
        online_table_name=online_table_name,
        publish_mode=PublishSpecPublishMode.SNAPSHOT
    )
    
    result = w.feature_store.publish_table(
        source_table_name=full_table_name,
        publish_spec=publish_spec
    )
    print(f"Published to online store: {result}")
except Exception as e:
    print(f"Error publishing: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
