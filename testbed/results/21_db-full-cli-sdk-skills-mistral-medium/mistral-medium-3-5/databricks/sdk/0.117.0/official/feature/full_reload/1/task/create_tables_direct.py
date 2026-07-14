#!/usr/bin/env python3
"""
Create feature tables customers89eac8 v1 and v2 by inserting data directly
"""
import os
import csv
from databricks.sdk import WorkspaceClient

# Environment variables
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]

# Parse schema
catalog, schema_name = SCHEMA.split(".")

print(f"Catalog: {catalog}, Schema: {schema_name}, Prefix: {PREFIX}")

# Initialize workspace client
ws = WorkspaceClient()
sql = ws.statement_execution

table_name = "customers89eac8"
full_table_name = f"{catalog}.{schema_name}.{table_name}"

# Step 1: Create v1 table and load initial export
print("Step 1: Creating v1 table (customers89eac8 with initial export)")

# Create v1 table with the schema from initial_export.csv
create_v1 = f"""
CREATE OR REPLACE TABLE {full_table_name} (
  row_id STRING,
  name STRING,
  balance_eur DOUBLE,
  updated_at BIGINT
)
USING DELTA
"""

try:
    result = sql.execute_statement(
        warehouse_id="default",
        catalog=catalog,
        schema=schema_name,
        statement=create_v1,
        wait_timeout="60s"
    )
    print(f"  Created v1 Delta table: {full_table_name}")
except Exception as e:
    print(f"  ERROR creating v1 Delta table: {e}")
    raise

# Read initial export CSV and insert data
with open("data/initial_export.csv", "r") as f:
    reader = csv.DictReader(f)
    batch = []
    batch_size = 50
    
    for i, row in enumerate(reader):
        batch.append(row)
        if len(batch) >= batch_size or i == 0:
            # Build INSERT statement
            values = ", ".join([
                f"('{r['row_id']}', '{r['name']}', {r['balance_eur']}, {r['updated_at']})"
                for r in batch
            ])
            insert_stmt = f"""
            INSERT INTO {full_table_name} (row_id, name, balance_eur, updated_at)
            VALUES {values}
            """
            try:
                sql.execute_statement(
                    warehouse_id="default",
                    catalog=catalog,
                    schema=schema_name,
                    statement=insert_stmt,
                    wait_timeout="60s"
                )
                print(f"  Inserted batch {i//batch_size + 1}")
            except Exception as e:
                print(f"  ERROR inserting batch: {e}")
                raise
            batch = []

print("  V1 table created and loaded successfully\n")

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
        wait_timeout="60s"
    )
    print(f"  Dropped v1 table")
except Exception as e:
    print(f"  ERROR dropping v1 table: {e}")
    raise

# Create v2 table with the new schema
create_v2 = f"""
CREATE OR REPLACE TABLE {full_table_name} (
  row_id STRING,
  full_name STRING,
  balance DOUBLE,
  currency STRING,
  updated_at BIGINT
)
USING DELTA
"""

try:
    result = sql.execute_statement(
        warehouse_id="default",
        catalog=catalog,
        schema=schema_name,
        statement=create_v2,
        wait_timeout="60s"
    )
    print(f"  Created v2 Delta table: {full_table_name}")
except Exception as e:
    print(f"  ERROR creating v2 Delta table: {e}")
    raise

# Read new export CSV and insert data
with open("data/reload/new_export.csv", "r") as f:
    reader = csv.DictReader(f)
    batch = []
    batch_size = 50
    
    for i, row in enumerate(reader):
        batch.append(row)
        if len(batch) >= batch_size or i == 0:
            # Build INSERT statement
            values = ", ".join([
                f"('{r['row_id']}', '{r['full_name']}', {r['balance']}, '{r['currency']}', {r['updated_at']})"
                for r in batch
            ])
            insert_stmt = f"""
            INSERT INTO {full_table_name} (row_id, full_name, balance, currency, updated_at)
            VALUES {values}
            """
            try:
                sql.execute_statement(
                    warehouse_id="default",
                    catalog=catalog,
                    schema=schema_name,
                    statement=insert_stmt,
                    wait_timeout="60s"
                )
                print(f"  Inserted batch {i//batch_size + 1}")
            except Exception as e:
                print(f"  ERROR inserting batch: {e}")
                raise
            batch = []

print("  V2 table created and loaded successfully\n")

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
    
    # Check schema
    columns = ws.tables.list(catalog_name=catalog, schema_name=schema_name, name=table_name)
    print(f"    Columns: {[c.name for c in columns]}")
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
