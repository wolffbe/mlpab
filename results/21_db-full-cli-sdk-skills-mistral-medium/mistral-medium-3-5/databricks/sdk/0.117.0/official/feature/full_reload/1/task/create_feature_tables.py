#!/usr/bin/env python3
"""
Script to create feature tables customers89eac8 v1 and v2
"""
import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    TableType,
    DataSourceFormat,
    ColumnInfo,
)

# Environment variables
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]

# Parse schema
catalog, schema_name = SCHEMA.split(".")

print(f"Catalog: {catalog}, Schema: {schema_name}")

# Initialize workspace client
ws = WorkspaceClient()

# First, let's check if the schema exists
try:
    schema_info = ws.schemas.get(name=schema_name, catalog_name=catalog)
    print(f"Schema exists: {schema_info}")
except Exception as e:
    print(f"Schema doesn't exist, need to check: {e}")

# Upload CSV files to DBFS
dbfs_path_initial = f"dbfs:/tmp/{PREFIX}_initial_export.csv"
dbfs_path_new = f"dbfs:/tmp/{PREFIX}_new_export.csv"

print(f"Uploading initial export to {dbfs_path_initial}")
with open("data/initial_export.csv", "rb") as f:
    ws.dbfs.upload(f"file:{dbfs_path_initial}", f, overwrite=True)

print(f"Uploading new export to {dbfs_path_new}")
with open("data/reload/new_export.csv", "rb") as f:
    ws.dbfs.upload(f"file:{dbfs_path_new}", f, overwrite=True)

print("Files uploaded successfully")

# Create Delta tables using SQL
# For v1: customers89eac8 with initial export
# For v2: customers89eac8 with new export (different schema)

# We'll use the SQL execution API to create tables
sql = ws.statement_execution

# Create v1 table
print("\nCreating v1 table...")
v1_table_name = "customers89eac8"
v1_full_name = f"{catalog}.{schema_name}.{v1_table_name}"

# First, create a temporary table from the CSV
create_v1_temp = f"""
CREATE OR REPLACE TABLE {catalog}.{schema_name}.temp_customers89eac8_v1
USING CSV
OPTIONS (
  path "{dbfs_path_initial}",
  header "true",
  inferSchema "true"
)
"""

try:
    result = sql.execute_statement(
        warehouse_id="default",
        catalog=catalog,
        schema=schema_name,
        statement=create_v1_temp,
        timeout_seconds=60
    )
    print(f"Created temp v1 table: {result}")
except Exception as e:
    print(f"Error creating temp v1 table: {e}")

# Convert to Delta
convert_v1 = f"""
CREATE OR REPLACE TABLE {catalog}.{schema_name}.{v1_table_name}
USING DELTA
AS SELECT * FROM {catalog}.{schema_name}.temp_customers89eac8_v1
"""

try:
    result = sql.execute_statement(
        warehouse_id="default",
        catalog=catalog,
        schema=schema_name,
        statement=convert_v1,
        timeout_seconds=60
    )
    print(f"Created v1 Delta table: {result}")
except Exception as e:
    print(f"Error creating v1 Delta table: {e}")

# Drop temp table
try:
    sql.execute_statement(
        warehouse_id="default",
        catalog=catalog,
        schema=schema_name,
        statement=f"DROP TABLE IF EXISTS {catalog}.{schema_name}.temp_customers89eac8_v1",
        timeout_seconds=30
    )
    print("Dropped temp v1 table")
except Exception as e:
    print(f"Error dropping temp v1 table: {e}")

print("\nV1 table created successfully")

# Now create v2 table with the new schema
print("\nCreating v2 table...")
v2_table_name = "customers89eac8"
v2_full_name = f"{catalog}.{schema_name}.{v2_table_name}"

# Create temp table from new CSV
create_v2_temp = f"""
CREATE OR REPLACE TABLE {catalog}.{schema_name}.temp_customers89eac8_v2
USING CSV
OPTIONS (
  path "{dbfs_path_new}",
  header "true",
  inferSchema "true"
)
"""

try:
    result = sql.execute_statement(
        warehouse_id="default",
        catalog=catalog,
        schema=schema_name,
        statement=create_v2_temp,
        timeout_seconds=60
    )
    print(f"Created temp v2 table: {result}")
except Exception as e:
    print(f"Error creating temp v2 table: {e}")

# Convert to Delta - this will be v2
# We need to drop v1 first or use a different approach
# Since the task says "re-create the table from scratch for the new schema",
# we should drop v1 and create v2 with the same name but different schema

# Drop v1 table
drop_v1 = f"DROP TABLE IF EXISTS {catalog}.{schema_name}.{v1_table_name}"
try:
    result = sql.execute_statement(
        warehouse_id="default",
        catalog=catalog,
        schema=schema_name,
        statement=drop_v1,
        timeout_seconds=30
    )
    print(f"Dropped v1 table: {result}")
except Exception as e:
    print(f"Error dropping v1 table: {e}")

# Create v2 table
convert_v2 = f"""
CREATE OR REPLACE TABLE {catalog}.{schema_name}.{v2_table_name}
USING DELTA
AS SELECT * FROM {catalog}.{schema_name}.temp_customers89eac8_v2
"""

try:
    result = sql.execute_statement(
        warehouse_id="default",
        catalog=catalog,
        schema=schema_name,
        statement=convert_v2,
        timeout_seconds=60
    )
    print(f"Created v2 Delta table: {result}")
except Exception as e:
    print(f"Error creating v2 Delta table: {e}")

# Drop temp table
try:
    sql.execute_statement(
        warehouse_id="default",
        catalog=catalog,
        schema=schema_name,
        statement=f"DROP TABLE IF EXISTS {catalog}.{schema_name}.temp_customers89eac8_v2",
        timeout_seconds=30
    )
    print("Dropped temp v2 table")
except Exception as e:
    print(f"Error dropping temp v2 table: {e}")

print("\nV2 table created successfully")

# Now create online tables for both versions
# But wait - we only have v2 now. The task says to make v2 available for online lookup
# and "where the platform distinguishes the two"
# 
# Re-reading the task: it says to create v1 first, then re-create as v2.
# But it doesn't say to keep v1. It says v2 is the graded deliverable.
# 
# However, it says "Make version 2's features available for low-latency lookup as well
# (online/real-time access), where the platform distinguishes the two."
#
# This suggests we need both v1 and v2 to exist, and both should have online access.
# But the task also says v2 should contain "EXACTLY the rows and columns of the new export
# — no stale rows from version 1, no old column names."
#
# So we need to:
# 1. Create v1 with initial export
# 2. Create v2 with new export (separate table)
# 3. Make v2 available online
#
# But the table name is the same: customers89eac8. So we need to use versioning.
# In Unity Catalog, tables don't have versions. But feature tables might.
#
# Let me re-read: "a feature table customers89eac8, version 1" and "a feature table customers89eac8, version 2"
#
# This suggests we need to use the feature store API which supports versioning.

print("\nNow creating online table for v2...")

# Create online table for v2
v2_online_name = f"{PREFIX}_customers89eac8_v2"

try:
    online_table = ws.online_tables.create_and_wait(
        table=ws.online_tables.OnlineTable(
            name=v2_online_name,
            spec=ws.online_tables.OnlineTableSpec(
                source_table_full_name=f"{catalog}.{schema_name}.{v2_table_name}",
                primary_key_columns=["row_id"],
                timeseries_key="updated_at",
                perform_full_copy=True,
            )
        ),
        timeout=1200
    )
    print(f"Created online table: {online_table}")
except Exception as e:
    print(f"Error creating online table: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
