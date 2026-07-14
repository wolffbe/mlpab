#!/usr/bin/env python3
"""
Create feature table accounts9ad208 with latest revisions from batch files.
"""
import os
import uuid
import base64

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

# Step 2: Upload CSV files to DBFS
# Create a unique temp directory in DBFS
temp_dbfs_path = f"FileStore/mlpab_temp_{uuid.uuid4().hex[:8]}/"

print(f"\nUploading files to dbfs:/{temp_dbfs_path}")
for batch_file in ['batch_1.csv', 'batch_2.csv', 'batch_3.csv']:
    local_path = f"data/{batch_file}"
    dbfs_dest = f"{temp_dbfs_path}{batch_file}"
    
    # Read local file
    with open(local_path, 'r') as f:
        content = f.read()
    
    # Write to DBFS - contents should be a string
    w.dbfs.put(f"dbfs:/{dbfs_dest}", contents=content, overwrite=True)
    print(f"  Uploaded {batch_file} to dbfs:/{dbfs_dest}")

# Step 3: Create a Delta table and load deduplicated data
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

# Load deduplicated data
# We need to read from DBFS CSV files, deduplicate by row_id keeping latest updated_at
print("\nLoading and deduplicating data...")
w.sql.execute(
    warehouse_id="default",
    statement=f"""
    INSERT OVERWRITE TABLE {full_table_name}
    WITH all_data AS (
      SELECT * FROM csv.`/FileStore/{temp_dbfs_path}batch_1.csv`
      UNION ALL
      SELECT * FROM csv.`/FileStore/{temp_dbfs_path}batch_2.csv`
      UNION ALL
      SELECT * FROM csv.`/FileStore/{temp_dbfs_path}batch_3.csv`
    ),
    ranked AS (
      SELECT 
        row_id,
        status,
        balance,
        updated_at,
        ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY updated_at DESC) as rn
      FROM all_data
    )
    SELECT row_id, status, balance, updated_at
    FROM ranked
    WHERE rn = 1
    """
)
print("Data loaded and deduplicated")

# Verify the data
print("\nVerifying data...")
result = w.sql.execute(
    warehouse_id="default",
    statement=f"SELECT COUNT(*) as cnt, COUNT(DISTINCT row_id) as distinct_ids FROM {full_table_name}"
)

for row in result:
    print(f"Total rows: {row.cnt}, Distinct row_ids: {row.distinct_ids}")

# Step 4: Register as a feature table and enable online access
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
