#!/usr/bin/env python3
"""
Create feature table accounts9ad208 with latest revisions from batch files.
"""
import os
import sys

# Set environment variables for Databricks SDK
os.environ['DATABRICKS_HOST'] = os.getenv('DATABRICKS_HOST', '')
os.environ['DATABRICKS_TOKEN'] = os.getenv('DATABRICKS_TOKEN', '')

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog, feature_store

# Get environment variables
SCHEMA = os.getenv('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpabc1ee89')
PREFIX = os.getenv('MLPABRICKS_PREFIX', 'mlpabc1ee89')

# Parse schema - it's in format catalog.schema or workspace.schema
catalog_name, schema_name = SCHEMA.split('.')

print(f"Catalog: {catalog_name}, Schema: {schema_name}")
print(f"Prefix: {PREFIX}")

# Initialize workspace client
w = WorkspaceClient()

# First, let's check if the schema exists, create if not
try:
    w.catalog.get_schema(name=SCHEMA)
    print(f"Schema {SCHEMA} already exists")
except Exception as e:
    print(f"Creating schema {SCHEMA}: {e}")
    w.catalog.create_schema(
        name=schema_name,
        catalog_name=catalog_name,
        comment="MLPAB schema for feature tables"
    )
    print(f"Schema {SCHEMA} created")

# Upload the CSV files to DBFS so we can read them
# We'll use dbfs:/FileStore/tables/ as a temporary location
import tempfile
import uuid

# Create a unique temp directory in DBFS
temp_dbfs_path = f"dbfs:/FileStore/mlpab_temp_{uuid.uuid4().hex[:8]}/"

# Upload files to DBFS
for batch_file in ['batch_1.csv', 'batch_2.csv', 'batch_3.csv']:
    local_path = f"data/{batch_file}"
    dbfs_dest = f"{temp_dbfs_path}{batch_file}"
    
    # Read local file
    with open(local_path, 'r') as f:
        content = f.read()
    
    # Write to DBFS
    w.dbfs.put(dbfs_dest, content.encode('utf-8'), overwrite=True)
    print(f"Uploaded {batch_file} to {dbfs_dest}")

# Now create a SQL query that:
# 1. Reads all three CSV files
# 2. Combines them
# 3. For each row_id, keeps only the row with the latest updated_at

query = f"""
WITH all_data AS (
  SELECT *, 'batch_1' as source FROM csv.`{temp_dbfs_path}batch_1.csv`
  UNION ALL
  SELECT *, 'batch_2' as source FROM csv.`{temp_dbfs_path}batch_2.csv`
  UNION ALL
  SELECT *, 'batch_3' as source FROM csv.`{temp_dbfs_path}batch_3.csv`
),
ranked AS (
  SELECT 
    row_id,
    status,
    balance,
    updated_at,
    source,
    ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY updated_at DESC) as rn
  FROM all_data
)
SELECT row_id, status, balance, updated_at
FROM ranked
WHERE rn = 1
"""

print("Query:")
print(query)

# Create the feature table
feature_table_name = "accounts9ad208"
full_table_name = f"{SCHEMA}.{feature_table_name}"

print(f"\nCreating feature table: {full_table_name}")

# First, create a regular Delta table with the deduplicated data
w.sql.execute(
    warehouse_id="default",
    catalog=catalog_name,
    schema=schema_name,
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

# Now load the data using the query
# We need to write the deduplicated data to the table
w.sql.execute(
    warehouse_id="default",
    catalog=catalog_name,
    schema=schema_name,
    statement=f"""
    INSERT OVERWRITE TABLE {full_table_name}
    {query}
    """
)

print(f"Data loaded into {full_table_name}")

# Verify the data
result = w.sql.execute(
    warehouse_id="default",
    catalog=catalog_name,
    schema=schema_name,
    statement=f"SELECT COUNT(*) as cnt, COUNT(DISTINCT row_id) as distinct_ids FROM {full_table_name}"
)

for row in result:
    print(f"Total rows: {row.cnt}, Distinct row_ids: {row.distinct_ids}")

# Now register this as a feature table
# We need to use the Feature Store API
# The table must exist first, then we register it as a feature table

# Check if feature table already exists
try:
    ft = w.feature_store.get_feature_table(
        catalog_name=catalog_name,
        schema_name=schema_name,
        name=feature_table_name,
        version=1
    )
    print(f"Feature table already exists: {ft}")
except Exception as e:
    print(f"Feature table doesn't exist yet: {e}")
    
    # Create the feature table
    # We need to specify the primary keys and timestamp column
    ft = w.feature_store.create_feature_table(
        catalog_name=catalog_name,
        schema_name=schema_name,
        name=feature_table_name,
        primary_keys=["row_id"],
        timestamp_columns=["updated_at"],
        df_pair=None,  # We already have the table
        description="Accounts feature table with latest revisions"
    )
    print(f"Feature table created: {ft}")

# Now we need to enable it for online access
# This means creating an online store endpoint
# But first, let's check what's available

print("\nChecking online store endpoints...")
try:
    endpoints = w.online_tables.list()
    print(f"Existing online tables: {endpoints}")
except Exception as e:
    print(f"Error listing online tables: {e}")

# Create online table for low-latency lookup
# The online table name should be prefixed
online_table_name = f"{PREFIX}_accounts9ad208_online"

try:
    # Create online table from the feature table
    online_table = w.online_tables.create(
        name=online_table_name,
        source_table_fullname=full_table_name,
        comment="Online access for accounts9ad208 feature table"
    )
    print(f"Online table created: {online_table}")
except Exception as e:
    print(f"Error creating online table: {e}")
    # Try alternative approach - maybe we need to use the feature store API
    try:
        # Publish the feature table to online store
        w.feature_store.publish(
            catalog_name=catalog_name,
            schema_name=schema_name,
            name=feature_table_name,
            version=1
        )
        print("Feature table published to online store")
    except Exception as e2:
        print(f"Error publishing: {e2}")

print("\nDone!")
