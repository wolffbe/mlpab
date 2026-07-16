#!/usr/bin/env python3
"""Add primary key to feature table and publish to online store."""

import os
import time
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
CATALOG, SCHEMA_NAME = SCHEMA.split(".", 1)
TABLE_NAME = "eventsd693d3"
FULL_TABLE = f"{CATALOG}.{SCHEMA_NAME}.{TABLE_NAME}"
ONLINE_STORE_NAME = PREFIX.replace("_", "-") + "-os"
ONLINE_TABLE_FULL = f"{CATALOG}.{SCHEMA_NAME}.{TABLE_NAME}_online"

w = WorkspaceClient()
warehouses = list(w.warehouses.list())
wh_id = warehouses[0].id


def exec_sql(sql, timeout=120):
    print(f"  SQL: {sql[:150]}{'...' if len(sql)>150 else ''}")
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=wh_id,
        wait_timeout="30s",
    )
    if resp.status.state == StatementState.SUCCEEDED:
        return resp
    if resp.status.state in (StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED):
        raise RuntimeError(f"SQL failed: {resp.status.error}")
    # Wait
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = w.statement_execution.get_statement(resp.statement_id)
        if result.status.state == StatementState.SUCCEEDED:
            return result
        if result.status.state in (StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED):
            raise RuntimeError(f"SQL failed: {result.status.error}")
        time.sleep(2)
    raise TimeoutError(f"SQL timed out")


print(f"Table: {FULL_TABLE}")

# Check if table already has constraints
print("\n[1] Checking existing constraints...")
try:
    r = exec_sql(f"DESCRIBE EXTENDED {FULL_TABLE}")
    if r.result and r.result.data_array:
        for row in r.result.data_array:
            if row and any('constraint' in str(v).lower() or 'primary' in str(v).lower() for v in row):
                print(f"  Constraint found: {row}")
except Exception as e:
    print(f"  Error: {e}")

# Try to add primary key constraint - need to recreate with NOT ENFORCED
print("\n[2] Recreating table with PRIMARY KEY...")

# Drop existing
exec_sql(f"DROP TABLE IF EXISTS {FULL_TABLE}")
print("  Dropped old table")

# Read the valid rows from the saved data (we already know them from the CSV)
import csv

VALID_CATEGORIES = {"grocery", "travel", "salary", "rent", "other"}

def validate_row(row):
    amount_str = row.get("amount", "").strip()
    category = row.get("category", "").strip()
    if not amount_str or amount_str.lower() == "null":
        return False
    try:
        amount = float(amount_str)
    except ValueError:
        return False
    if amount < 0 or amount > 10000:
        return False
    if category not in VALID_CATEGORIES:
        return False
    return True

valid_rows = []
with open("data/events.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if validate_row(row):
            valid_rows.append(row)

# Create table with PRIMARY KEY constraint
create_sql = f"""
CREATE TABLE {FULL_TABLE} (
    row_id     STRING       NOT NULL,
    account_id STRING,
    event_time BIGINT,
    amount     DOUBLE,
    category   STRING,
    CONSTRAINT {TABLE_NAME}_pk PRIMARY KEY (row_id)
)
USING DELTA
"""
exec_sql(create_sql)
print("  Table created with PRIMARY KEY")

# Insert valid rows
def escape_str(s):
    if s is None or s == "":
        return "NULL"
    return "'" + s.replace("'", "''") + "'"

BATCH_SIZE = 500
for batch_start in range(0, len(valid_rows), BATCH_SIZE):
    batch = valid_rows[batch_start:batch_start + BATCH_SIZE]
    values = []
    for row in batch:
        values.append(f"({escape_str(row['row_id'])}, {escape_str(row['account_id'])}, {row['event_time'].strip()}, {row['amount'].strip()}, {escape_str(row['category'])})")
    exec_sql(f"INSERT INTO {FULL_TABLE} VALUES {', '.join(values)}")
    print(f"  Inserted batch {batch_start//BATCH_SIZE + 1}: {len(batch)} rows")

# Verify
r = exec_sql(f"SELECT COUNT(*) FROM {FULL_TABLE}")
count = r.result.data_array[0][0] if r.result and r.result.data_array else "?"
print(f"  Row count: {count}")

# Now try publishing
print(f"\n[3] Publishing to online store {ONLINE_STORE_NAME}...")
spec_dict = {
    "online_store": ONLINE_STORE_NAME,
    "publish_mode": "TRIGGERED",
    "online_table_name": ONLINE_TABLE_FULL,
}
print(f"  PublishSpec: {json.dumps(spec_dict)}")

try:
    result = w.api_client.do(
        "POST",
        f"/api/2.0/feature-store/tables/{FULL_TABLE}/publish",
        body={"publish_spec": spec_dict},
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    print(f"  SUCCESS: {result}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

print("\n=== Done ===")
