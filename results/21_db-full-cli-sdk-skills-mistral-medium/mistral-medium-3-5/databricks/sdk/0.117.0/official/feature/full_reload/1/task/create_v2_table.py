#!/usr/bin/env python3
"""
Create feature table customers89eac8 v2
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

# Find warehouse
warehouses = list(ws.warehouses.list())
warehouse_id = warehouses[0].id if warehouses else None
print(f"Using warehouse: {warehouse_id}\n")

if not warehouse_id:
    raise Exception("No warehouse available")

table_name = "customers89eac8"
full_table_name = f"{catalog}.{schema_name}.{table_name}"

# Step 1: Create v1 table and load initial export
print("Step 1: Creating v1 table (customers89eac8 with initial export)")

create_v1 = f"""
CREATE OR REPLACE TABLE {full_table_name} (
  row_id STRING,
  name STRING,
  balance_eur DOUBLE,
  updated_at BIGINT
)
USING DELTA
"""

sql.execute_statement(
    warehouse_id=warehouse_id,
    catalog=catalog,
    schema=schema_name,
    statement=create_v1,
    wait_timeout="50s"
)
print(f"  Created v1 Delta table")

# Insert v1 data
with open("data/initial_export.csv", "r") as f:
    reader = csv.DictReader(f)
    batch = []
    batch_size = 50
    
    for i, row in enumerate(reader):
        batch.append(row)
        if len(batch) >= batch_size:
            values = ", ".join([
                f"('{r['row_id']}', '{r['name']}', {r['balance_eur']}, {r['updated_at']})"
                for r in batch
            ])
            insert_stmt = f"INSERT INTO {full_table_name} (row_id, name, balance_eur, updated_at) VALUES {values}"
            sql.execute_statement(
                warehouse_id=warehouse_id,
                catalog=catalog,
                schema=schema_name,
                statement=insert_stmt,
                wait_timeout="50s"
            )
            batch = []
    
    if batch:
        values = ", ".join([
            f"('{r['row_id']}', '{r['name']}', {r['balance_eur']}, {r['updated_at']})"
            for r in batch
        ])
        insert_stmt = f"INSERT INTO {full_table_name} (row_id, name, balance_eur, updated_at) VALUES {values}"
        sql.execute_statement(
            warehouse_id=warehouse_id,
            catalog=catalog,
            schema=schema_name,
            statement=insert_stmt,
            wait_timeout="50s"
        )

print("  V1 table created and loaded\n")

# Step 2: Drop v1 and create v2
print("Step 2: Dropping v1 table and creating v2 table")

# Drop v1 table
sql.execute_statement(
    warehouse_id=warehouse_id,
    catalog=catalog,
    schema=schema_name,
    statement=f"DROP TABLE IF EXISTS {full_table_name}",
    wait_timeout="50s"
)
print("  Dropped v1 table")

# Create v2 table with new schema
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

sql.execute_statement(
    warehouse_id=warehouse_id,
    catalog=catalog,
    schema=schema_name,
    statement=create_v2,
    wait_timeout="50s"
)
print("  Created v2 Delta table")

# Insert v2 data
with open("data/reload/new_export.csv", "r") as f:
    reader = csv.DictReader(f)
    batch = []
    batch_size = 50
    
    for i, row in enumerate(reader):
        batch.append(row)
        if len(batch) >= batch_size:
            values = ", ".join([
                f"('{r['row_id']}', '{r['full_name']}', {r['balance']}, '{r['currency']}', {r['updated_at']})"
                for r in batch
            ])
            insert_stmt = f"INSERT INTO {full_table_name} (row_id, full_name, balance, currency, updated_at) VALUES {values}"
            sql.execute_statement(
                warehouse_id=warehouse_id,
                catalog=catalog,
                schema=schema_name,
                statement=insert_stmt,
                wait_timeout="50s"
            )
            batch = []
    
    if batch:
        values = ", ".join([
            f"('{r['row_id']}', '{r['full_name']}', {r['balance']}, '{r['currency']}', {r['updated_at']})"
            for r in batch
        ])
        insert_stmt = f"INSERT INTO {full_table_name} (row_id, full_name, balance, currency, updated_at) VALUES {values}"
        sql.execute_statement(
            warehouse_id=warehouse_id,
            catalog=catalog,
            schema=schema_name,
            statement=insert_stmt,
            wait_timeout="50s"
        )

print("  V2 table created and loaded\n")

# Step 3: Try to make it available for online access
# Since Online Tables are deprecated, we'll try to use the feature store
print("Step 3: Making v2 available for online access")

# Create online store
from databricks.sdk.service.ml import OnlineStore, PublishSpec, PublishSpecPublishMode

store_name = f"{PREFIX}-online-store"

try:
    # Check if store exists
    try:
        store = ws.feature_store.get_online_store(name=store_name)
        print(f"  Online store already exists: {store.name}")
    except:
        # Create store
        store = ws.feature_store.create_online_store(
            online_store=OnlineStore(name=store_name, capacity='CU_1')
        )
        print(f"  Created online store: {store.name}")
        
        # Wait for it to be ready
        import time
        for i in range(10):
            store = ws.feature_store.get_online_store(name=store_name)
            if store.state in ['ONLINE', 'AVAILABLE']:
                break
            time.sleep(10)
    
    # Publish v2 table to online store
    publish_spec = PublishSpec(
        online_store=store_name,
        online_table_name=f"{PREFIX}_customers89eac8_v2",
        publish_mode=PublishSpecPublishMode.SNAPSHOT
    )
    
    result = ws.feature_store.publish_table(
        source_table_name=full_table_name,
        publish_spec=publish_spec
    )
    print(f"  Published v2 table to online store")
    print(f"  Result: {result}")
    
except Exception as e:
    print(f"  WARNING: Could not publish to online store: {e}")
    print("  V2 table is still available as a Delta table for queries")

print("\nVerification:")
print(f"  Delta table: {full_table_name}")

# Query to verify
result = sql.execute_statement(
    warehouse_id=warehouse_id,
    catalog=catalog,
    schema=schema_name,
    statement=f'SELECT COUNT(*) as cnt FROM {full_table_name}',
    wait_timeout="50s"
)
print(f"  Row count: {result}")

result = sql.execute_statement(
    warehouse_id=warehouse_id,
    catalog=catalog,
    schema=schema_name,
    statement=f'SELECT * FROM {full_table_name} LIMIT 3',
    wait_timeout="50s"
)
print(f"  Sample data retrieved successfully")

print("\nAll done! V2 table customers89eac8 is ready in {SCHEMA}")
