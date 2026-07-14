#!/usr/bin/env python3
"""
Create feature tables customers89eac8 v1 and v2
"""
import os
from databricks.sdk import WorkspaceClient

# Environment variables
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]

# Parse schema
catalog, schema_name = SCHEMA.split(".")

print(f"Catalog: {catalog}, Schema: {schema_name}, Prefix: {PREFIX}")

# Initialize workspace client
ws = WorkspaceClient()
user = ws.current_user.me().user_name
workspace_base = f"/Users/{user}/{PREFIX}"

print(f"Workspace base: {workspace_base}")

# Create directory in workspace
try:
    ws.workspace.mkdirs(workspace_base)
    print(f"Created directory: {workspace_base}")
except Exception as e:
    print(f"Directory may already exist: {e}")

# Upload CSV files to workspace
workspace_path_initial = f"{workspace_base}/initial_export.csv"
workspace_path_new = f"{workspace_base}/new_export.csv"

print(f"Uploading initial export to workspace:{workspace_path_initial}")
with open("data/initial_export.csv", "rb") as f:
    ws.workspace.upload(workspace_path_initial, f, overwrite=True)

print(f"Uploading new export to workspace:{workspace_path_new}")
with open("data/reload/new_export.csv", "rb") as f:
    ws.workspace.upload(workspace_path_new, f, overwrite=True)

print("Files uploaded successfully\n")

# In Databricks SQL, workspace files can be accessed via file:/Workspace/...
# But the path format might be different. Let's try using the workspace path directly
# Actually, in SQL, we can use: file:/Workspace/Users/.../file.csv
# But the workspace path is /Users/..., so the SQL path would be file:/Workspace/Users/.../file.csv

# Convert workspace path to SQL-accessible path
# workspace: /Users/benedict@logicalclocks.com/mlpab5d9f50/initial_export.csv
# SQL: file:/Workspace/Users/benedict@logicalclocks.com/mlpab5d9f50/initial_export.csv

def workspace_to_sql_path(workspace_path):
    # workspace_path is like /Users/.../file.csv
    # SQL path should be file:/Workspace/.../file.csv
    return f"file:/Workspace{workspace_path}"

dbfs_path_initial = workspace_to_sql_path(workspace_path_initial)
dbfs_path_new = workspace_to_sql_path(workspace_path_new)

print(f"Using SQL paths:")
print(f"  Initial: {dbfs_path_initial}")
print(f"  New: {dbfs_path_new}\n")

sql = ws.statement_execution

# Step 1: Create v1 table
print("Step 1: Creating v1 table (customers89eac8 with initial export)")
table_name = "customers89eac8"
full_table_name = f"{catalog}.{schema_name}.{table_name}"

# Create temp table from CSV
create_temp_v1 = f"""
CREATE OR REPLACE TABLE {catalog}.{schema_name}.temp_v1
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
        statement=create_temp_v1,
        timeout_seconds=120
    )
    print(f"  Created temp v1 table")
except Exception as e:
    print(f"  ERROR creating temp v1 table: {e}")
    raise

# Create main v1 table as Delta
create_v1 = f"""
CREATE OR REPLACE TABLE {full_table_name}
USING DELTA
AS SELECT * FROM {catalog}.{schema_name}.temp_v1
"""

try:
    result = sql.execute_statement(
        warehouse_id="default",
        catalog=catalog,
        schema=schema_name,
        statement=create_v1,
        timeout_seconds=120
    )
    print(f"  Created v1 Delta table: {full_table_name}")
except Exception as e:
    print(f"  ERROR creating v1 Delta table: {e}")
    raise

# Drop temp table
try:
    sql.execute_statement(
        warehouse_id="default",
        catalog=catalog,
        schema=schema_name,
        statement=f"DROP TABLE IF EXISTS {catalog}.{schema_name}.temp_v1",
        timeout_seconds=60
    )
    print(f"  Dropped temp v1 table")
except Exception as e:
    print(f"  ERROR dropping temp v1 table: {e}")
    raise

print("  V1 table created successfully\n")

# Step 2: Create online table for v1
print("Step 2: Creating online table for v1")
v1_online_name = f"{PREFIX}_customers89eac8_v1"

try:
    v1_online = ws.online_tables.create_and_wait(
        table=ws.online_tables.OnlineTable(
            name=v1_online_name,
            spec=ws.online_tables.OnlineTableSpec(
                source_table_full_name=full_table_name,
                primary_key_columns=["row_id"],
                timeseries_key="updated_at",
                perform_full_copy=True,
            )
        ),
        timeout=1200
    )
    print(f"  Created online table: {v1_online_name}")
    print(f"  Status: {v1_online.status}")
except Exception as e:
    print(f"  ERROR creating online table v1: {e}")
    import traceback
    traceback.print_exc()
    raise

print("  V1 online table created successfully\n")

# Step 3: Drop v1 table and recreate as v2
print("Step 3: Dropping v1 table and creating v2 table")

# Drop v1 table
drop_v1 = f"DROP TABLE IF EXISTS {full_table_name}"
try:
    result = sql.execute_statement(
        warehouse_id="default",
        catalog=catalog,
        schema=schema_name,
        statement=drop_v1,
        timeout_seconds=60
    )
    print(f"  Dropped v1 table")
except Exception as e:
    print(f"  ERROR dropping v1 table: {e}")
    raise

# Create temp table from new CSV
create_temp_v2 = f"""
CREATE OR REPLACE TABLE {catalog}.{schema_name}.temp_v2
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
        statement=create_temp_v2,
        timeout_seconds=120
    )
    print(f"  Created temp v2 table")
except Exception as e:
    print(f"  ERROR creating temp v2 table: {e}")
    raise

# Create v2 table (same name as v1, but with new schema)
create_v2 = f"""
CREATE OR REPLACE TABLE {full_table_name}
USING DELTA
AS SELECT * FROM {catalog}.{schema_name}.temp_v2
"""

try:
    result = sql.execute_statement(
        warehouse_id="default",
        catalog=catalog,
        schema=schema_name,
        statement=create_v2,
        timeout_seconds=120
    )
    print(f"  Created v2 Delta table: {full_table_name}")
except Exception as e:
    print(f"  ERROR creating v2 Delta table: {e}")
    raise

# Drop temp table
try:
    sql.execute_statement(
        warehouse_id="default",
        catalog=catalog,
        schema=schema_name,
        statement=f"DROP TABLE IF EXISTS {catalog}.{schema_name}.temp_v2",
        timeout_seconds=60
    )
    print(f"  Dropped temp v2 table")
except Exception as e:
    print(f"  ERROR dropping temp v2 table: {e}")
    raise

print("  V2 table created successfully\n")

# Step 4: Create online table for v2
print("Step 4: Creating online table for v2")
v2_online_name = f"{PREFIX}_customers89eac8_v2"

try:
    v2_online = ws.online_tables.create_and_wait(
        table=ws.online_tables.OnlineTable(
            name=v2_online_name,
            spec=ws.online_tables.OnlineTableSpec(
                source_table_full_name=full_table_name,
                primary_key_columns=["row_id"],
                timeseries_key="updated_at",
                perform_full_copy=True,
            )
        ),
        timeout=1200
    )
    print(f"  Created online table: {v2_online_name}")
    print(f"  Status: {v2_online.status}")
except Exception as e:
    print(f"  ERROR creating online table v2: {e}")
    import traceback
    traceback.print_exc()
    raise

print("  V2 online table created successfully\n")

# Verify the tables
print("Verification:")
print(f"  Delta table: {full_table_name}")
try:
    table_info = ws.tables.get(name=table_name, catalog_name=catalog, schema_name=schema_name)
    print(f"    Exists: Yes")
    print(f"    Type: {table_info.table_type}")
except Exception as e:
    print(f"    ERROR: {e}")

print(f"  Online table v1: {v1_online_name}")
try:
    ot_v1 = ws.online_tables.get(name=v1_online_name)
    print(f"    Exists: Yes")
    print(f"    Status: {ot_v1.status}")
except Exception as e:
    print(f"    ERROR: {e}")

print(f"  Online table v2: {v2_online_name}")
try:
    ot_v2 = ws.online_tables.get(name=v2_online_name)
    print(f"    Exists: Yes")
    print(f"    Status: {ot_v2.status}")
except Exception as e:
    print(f"    ERROR: {e}")

print("\nAll done!")
