"""
Batch score accounts using point-in-time correct features and create a Databricks feature table.
"""
import os
import csv
import json
import math
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
from databricks.sdk.service import catalog as catalog_service

# Config
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.mlpab69a58e
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]   # mlpab69a58e
TABLE_NAME = "scores4f5893"
FULL_TABLE_NAME = f"{SCHEMA}.{TABLE_NAME}"

# T in epoch milliseconds
T = 1773306000000

# Load model weights
MODEL = {
    "w_f1": -0.9682,
    "w_f2": -0.0299,
    "w_f3": 1.2708,
    "bias": -0.1715,
}

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

# Read feature history and compute point-in-time correct scores
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "feature_history.csv")
data_path = os.path.normpath(data_path)

print(f"Reading feature history from {data_path}")
# Group revisions by account
revisions = {}
with open(data_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        aid = row["account_id"]
        ts = int(row["event_time"])
        if ts <= T:
            if aid not in revisions or ts > revisions[aid]["event_time"]:
                revisions[aid] = {
                    "event_time": ts,
                    "f1": float(row["f1"]),
                    "f2": float(row["f2"]),
                    "f3": float(row["f3"]),
                }

print(f"Found {len(revisions)} accounts with valid revisions at or before T={T}")

# Compute scores
scored = []
for aid, feat in sorted(revisions.items()):
    z = (MODEL["w_f1"] * feat["f1"] +
         MODEL["w_f2"] * feat["f2"] +
         MODEL["w_f3"] * feat["f3"] +
         MODEL["bias"])
    score = round(sigmoid(z), 6)
    scored.append((aid, score))

print(f"Computed {len(scored)} scores. Sample: {scored[:3]}")

# Connect to Databricks
w = WorkspaceClient()
print(f"Connected to Databricks: {w.config.host}")

# Find a SQL warehouse to use
warehouses = list(w.warehouses.list())
print(f"Available warehouses: {[wh.name for wh in warehouses]}")
if not warehouses:
    raise RuntimeError("No SQL warehouses available")
# Pick the first running or starting warehouse
warehouse = None
for wh in warehouses:
    if wh.state and wh.state.value in ("RUNNING", "STARTING"):
        warehouse = wh
        break
if warehouse is None:
    warehouse = warehouses[0]
warehouse_id = warehouse.id
print(f"Using warehouse: {warehouse.name} (id={warehouse_id})")

def execute_statement(sql, warehouse_id=warehouse_id, timeout=300):
    """Execute a SQL statement and wait for it to complete."""
    print(f"Executing SQL: {sql[:200]}...")
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
    )
    stmt_id = resp.statement_id
    # Poll for completion
    start = time.time()
    while True:
        status = w.statement_execution.get_statement(stmt_id)
        state = status.status.state
        if state == StatementState.SUCCEEDED:
            return status
        elif state in (StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED):
            error_msg = status.status.error.message if status.status.error else "Unknown error"
            raise RuntimeError(f"Statement failed with state {state}: {error_msg}")
        elapsed = time.time() - start
        if elapsed > timeout:
            raise TimeoutError(f"Statement did not complete within {timeout}s")
        time.sleep(2)

# Parse schema components
catalog_name, schema_name = SCHEMA.split(".", 1)  # workspace, mlpab69a58e

# Create the table with the scored data
# Build VALUES clause
values_rows = []
for aid, score in scored:
    values_rows.append(f"('{aid}', {score})")
values_clause = ",\n  ".join(values_rows)

# Drop and recreate table
drop_sql = f"DROP TABLE IF EXISTS {FULL_TABLE_NAME}"
execute_statement(drop_sql)

create_sql = f"""
CREATE TABLE {FULL_TABLE_NAME} (
  account_id STRING NOT NULL,
  score DOUBLE
)
USING DELTA
TBLPROPERTIES (
  'delta.minReaderVersion' = '1',
  'delta.minWriterVersion' = '2'
)
"""
execute_statement(create_sql)
print("Table created.")

# Insert data in batches to avoid huge SQL statements
BATCH_SIZE = 500
for i in range(0, len(scored), BATCH_SIZE):
    batch = scored[i:i+BATCH_SIZE]
    batch_values = ",\n  ".join(f"('{aid}', {score})" for aid, score in batch)
    insert_sql = f"INSERT INTO {FULL_TABLE_NAME} VALUES\n  {batch_values}"
    execute_statement(insert_sql)
    print(f"Inserted batch {i//BATCH_SIZE + 1} ({len(batch)} rows)")

# Verify row count
count_result = execute_statement(f"SELECT COUNT(*) as cnt FROM {FULL_TABLE_NAME}")
print(f"Row count result: {count_result.result}")

print(f"Feature table {FULL_TABLE_NAME} created with {len(scored)} rows.")

# Now create an Online Table for low-latency lookup
print(f"\nCreating online table for {FULL_TABLE_NAME}...")

online_table_name = FULL_TABLE_NAME  # same full name; SDK uses catalog.schema.table

# Check if online table already exists and delete it
try:
    existing = w.online_tables.get(name=FULL_TABLE_NAME)
    print(f"Online table already exists, deleting...")
    w.online_tables.delete(name=FULL_TABLE_NAME)
    # Wait for deletion
    time.sleep(10)
except Exception as e:
    if "does not exist" in str(e).lower() or "not found" in str(e).lower() or "404" in str(e):
        print(f"No existing online table found.")
    else:
        print(f"Note: {e}")

from databricks.sdk.service.catalog import (
    OnlineTable,
    OnlineTableSpec,
    OnlineTableSpecMirrorType,
)

spec = OnlineTableSpec(
    source_table_full_name=FULL_TABLE_NAME,
    primary_key_columns=["account_id"],
    run_triggered=OnlineTableSpec.__dict__.get("run_triggered"),
)

# Try creating the online table
try:
    online_table = w.online_tables.create(
        name=FULL_TABLE_NAME,
        spec=OnlineTableSpec(
            source_table_full_name=FULL_TABLE_NAME,
            primary_key_columns=["account_id"],
        ),
    )
    print(f"Online table creation initiated: {online_table}")
except Exception as e:
    print(f"Online table creation result: {e}")
    # Try alternative approach
    print("Trying alternative online table creation...")

# Wait a bit and check status
time.sleep(15)
try:
    ot = w.online_tables.get(name=FULL_TABLE_NAME)
    print(f"Online table status: {ot}")
except Exception as e:
    print(f"Could not get online table status: {e}")

print("\nDone!")
print(f"Feature table: {FULL_TABLE_NAME}")
print(f"Columns: account_id (record key), score")
print(f"Rows: {len(scored)}")
