#!/usr/bin/env python3
"""Final setup: recreate table with PK + CDF, publish to online store."""

import csv
import json
import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

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
    print(f"  SQL: {sql[:160]}{'...' if len(sql)>160 else ''}")
    resp = w.statement_execution.execute_statement(
        statement=sql, warehouse_id=wh_id, wait_timeout="30s",
    )
    if resp.status.state == StatementState.SUCCEEDED:
        return resp
    if resp.status.state in (StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED):
        raise RuntimeError(f"SQL failed: {resp.status.error}")
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = w.statement_execution.get_statement(resp.statement_id)
        if result.status.state == StatementState.SUCCEEDED:
            return result
        if result.status.state in (StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED):
            raise RuntimeError(f"SQL failed: {result.status.error}")
        time.sleep(2)
    raise TimeoutError("SQL timed out")


VALID_CATEGORIES = {"grocery", "travel", "salary", "rent", "other"}

def validate_row(row):
    amount_str = row.get("amount", "").strip()
    category = row.get("category", "").strip()
    if not amount_str or amount_str.lower() == "null":
        return False, "null/empty amount"
    try:
        amount = float(amount_str)
    except ValueError:
        return False, "non-numeric amount"
    if amount < 0 or amount > 10000:
        return False, "amount out of range"
    if category not in VALID_CATEGORIES:
        return False, "invalid category"
    return True, None


def escape_str(s):
    if s is None or s == "":
        return "NULL"
    return "'" + s.replace("'", "''") + "'"


print("=== Final Feature Table Setup ===")
print(f"Table: {FULL_TABLE}")
print(f"Online store: {ONLINE_STORE_NAME}")
print(f"Online table: {ONLINE_TABLE_FULL}")

# Step 1: Read data
print("\n[1] Reading and filtering CSV...")
valid_rows = []
rejected_ids = []
with open("data/events.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ok, _ = validate_row(row)
        if ok:
            valid_rows.append(row)
        else:
            rejected_ids.append(row["row_id"])
print(f"    Valid: {len(valid_rows)}, Rejected: {len(rejected_ids)}")

# Write answers.json
os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"rejected": rejected_ids}, f, indent=2)
print("    Written: submission/answers.json")

# Step 2: Create table with PK + CDF
print("\n[2] Creating table with PRIMARY KEY and Change Data Feed...")
exec_sql(f"DROP TABLE IF EXISTS {FULL_TABLE}")

create_sql = f"""CREATE TABLE {FULL_TABLE} (
    row_id     STRING NOT NULL,
    account_id STRING,
    event_time BIGINT,
    amount     DOUBLE,
    category   STRING,
    CONSTRAINT {TABLE_NAME}_pk PRIMARY KEY (row_id)
)
USING DELTA
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')"""
exec_sql(create_sql)
print("    Table created.")

# Step 3: Insert valid rows
print(f"\n[3] Inserting {len(valid_rows)} rows...")
BATCH_SIZE = 500
for i in range(0, len(valid_rows), BATCH_SIZE):
    batch = valid_rows[i:i + BATCH_SIZE]
    vals = [
        f"({escape_str(r['row_id'])}, {escape_str(r['account_id'])}, {r['event_time'].strip()}, {r['amount'].strip()}, {escape_str(r['category'])})"
        for r in batch
    ]
    exec_sql(f"INSERT INTO {FULL_TABLE} VALUES {', '.join(vals)}")
    print(f"    Batch {i//BATCH_SIZE + 1}: {len(batch)} rows")

r = exec_sql(f"SELECT COUNT(*) FROM {FULL_TABLE}")
print(f"    Total rows: {r.result.data_array[0][0]}")

# Step 4: Publish to online store (TRIGGERED mode)
print(f"\n[4] Publishing to online store...")
spec_dict = {
    "online_store": ONLINE_STORE_NAME,
    "publish_mode": "TRIGGERED",
    "online_table_name": ONLINE_TABLE_FULL,
}
print(f"    Spec: {json.dumps(spec_dict)}")
try:
    result = w.api_client.do(
        "POST",
        f"/api/2.0/feature-store/tables/{FULL_TABLE}/publish",
        body={"publish_spec": spec_dict},
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    print(f"    SUCCESS: {result}")
except Exception as e:
    print(f"    Error: {type(e).__name__}: {e}")
    # Try SNAPSHOT mode as fallback (no CDF needed)
    print(f"\n    Trying SNAPSHOT mode...")
    spec_dict2 = {
        "online_store": ONLINE_STORE_NAME,
        "publish_mode": "SNAPSHOT",
        "online_table_name": ONLINE_TABLE_FULL,
    }
    try:
        result2 = w.api_client.do(
            "POST",
            f"/api/2.0/feature-store/tables/{FULL_TABLE}/publish",
            body={"publish_spec": spec_dict2},
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        print(f"    SNAPSHOT SUCCESS: {result2}")
    except Exception as e2:
        print(f"    SNAPSHOT Error: {type(e2).__name__}: {e2}")

print("\n=== Done ===")
