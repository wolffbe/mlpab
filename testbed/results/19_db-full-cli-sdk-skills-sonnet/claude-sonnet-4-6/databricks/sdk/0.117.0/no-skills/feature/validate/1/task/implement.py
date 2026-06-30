#!/usr/bin/env python3
"""
Complete implementation: feature table creation and data loading.
"""

import csv
import json
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]   # workspace.mlpab394daa
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]    # mlpab394daa
CATALOG, SCHEMA_NAME = SCHEMA.split(".", 1)

TABLE_NAME = "eventsd693d3"
FULL_TABLE = f"{CATALOG}.{SCHEMA_NAME}.{TABLE_NAME}"

DATA_FILE = "data/events.csv"
SUBMISSION_DIR = "submission"

VALID_CATEGORIES = {"grocery", "travel", "salary", "rent", "other"}


def validate_row(row):
    amount_str = row.get("amount", "").strip()
    category = row.get("category", "").strip()
    if not amount_str or amount_str.lower() == "null":
        return False, "null/empty amount"
    try:
        amount = float(amount_str)
    except ValueError:
        return False, f"non-numeric amount: {amount_str}"
    if amount < 0 or amount > 10000:
        return False, f"amount out of range: {amount}"
    if category not in VALID_CATEGORIES:
        return False, f"invalid category: {category}"
    return True, None


def read_and_filter_csv():
    valid_rows = []
    rejected_ids = []
    with open(DATA_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ok, _ = validate_row(row)
            if ok:
                valid_rows.append(row)
            else:
                rejected_ids.append(row["row_id"])
    return valid_rows, rejected_ids


def wait_for_statement(w, statement_id, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = w.statement_execution.get_statement(statement_id)
        state = result.status.state
        if state in (StatementState.SUCCEEDED,):
            return result
        if state in (StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED):
            raise RuntimeError(f"Statement {statement_id} failed: {result.status.error}")
        time.sleep(2)
    raise TimeoutError(f"Statement {statement_id} timed out")


def exec_sql(w, warehouse_id, sql, timeout=300):
    print(f"  SQL: {sql[:120]}{'...' if len(sql)>120 else ''}")
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout="30s",
    )
    if resp.status.state == StatementState.SUCCEEDED:
        return resp
    if resp.status.state in (StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED):
        raise RuntimeError(f"SQL failed: {resp.status.error}\nSQL: {sql}")
    # Still running, wait
    return wait_for_statement(w, resp.statement_id, timeout)


def main():
    print("=== Feature Table Task ===")
    print(f"Catalog: {CATALOG}, Schema: {SCHEMA_NAME}, Table: {TABLE_NAME}")

    # --- Step 1: Filter data ---
    print("\n[1] Filtering CSV data...")
    valid_rows, rejected_ids = read_and_filter_csv()
    print(f"    Valid: {len(valid_rows)}, Rejected: {len(rejected_ids)}")

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    answers_path = os.path.join(SUBMISSION_DIR, "answers.json")
    with open(answers_path, "w") as f:
        json.dump({"rejected": rejected_ids}, f, indent=2)
    print(f"    Written: {answers_path}")

    # --- Step 2: Connect ---
    print("\n[2] Connecting to Databricks...")
    w = WorkspaceClient()
    print(f"    Host: {w.config.host}")

    # --- Step 3: Find warehouse ---
    print("\n[3] Finding warehouse...")
    warehouses = list(w.warehouses.list())
    warehouse = warehouses[0]
    warehouse_id = warehouse.id
    print(f"    Using warehouse: {warehouse.name} ({warehouse_id})")

    # --- Step 4: Drop existing table if any ---
    print("\n[4] Dropping existing table (if any)...")
    try:
        exec_sql(w, warehouse_id, f"DROP TABLE IF EXISTS {FULL_TABLE}")
        print("    Dropped.")
    except Exception as e:
        print(f"    Drop skipped: {e}")

    # --- Step 5: Create Delta table ---
    print("\n[5] Creating Delta table...")
    create_sql = f"""
    CREATE TABLE {FULL_TABLE} (
        row_id     STRING       NOT NULL,
        account_id STRING,
        event_time BIGINT,
        amount     DOUBLE,
        category   STRING
    )
    USING DELTA
    TBLPROPERTIES (
        'delta.feature.allowColumnDefaults' = 'enabled'
    )
    """
    exec_sql(w, warehouse_id, create_sql)
    print("    Table created.")

    # --- Step 6: Insert valid rows in batches ---
    print(f"\n[6] Inserting {len(valid_rows)} valid rows...")
    BATCH_SIZE = 500

    def escape_str(s):
        if s is None or s == "":
            return "NULL"
        return "'" + s.replace("'", "''") + "'"

    for batch_start in range(0, len(valid_rows), BATCH_SIZE):
        batch = valid_rows[batch_start:batch_start + BATCH_SIZE]
        values = []
        for row in batch:
            row_id = escape_str(row["row_id"])
            account_id = escape_str(row["account_id"])
            event_time = row["event_time"].strip()
            amount = row["amount"].strip()
            category = escape_str(row["category"])
            values.append(f"({row_id}, {account_id}, {event_time}, {amount}, {category})")

        insert_sql = f"INSERT INTO {FULL_TABLE} VALUES {', '.join(values)}"
        exec_sql(w, warehouse_id, insert_sql)
        print(f"    Inserted batch {batch_start//BATCH_SIZE + 1} ({len(batch)} rows)")

    # Verify row count
    count_result = exec_sql(w, warehouse_id, f"SELECT COUNT(*) FROM {FULL_TABLE}")
    count = count_result.result.data_array[0][0] if count_result.result and count_result.result.data_array else "unknown"
    print(f"    Total rows in table: {count}")

    # --- Step 7: Create Online Table for low-latency access ---
    print("\n[7] Creating Online Table...")
    online_table_name = f"{FULL_TABLE}_online"
    print(f"    Online table: {online_table_name}")

    # Check if online table already exists and delete it
    try:
        existing = w.online_tables.get(online_table_name)
        print(f"    Deleting existing online table...")
        w.online_tables.delete(online_table_name)
        time.sleep(5)
    except Exception as e:
        print(f"    No existing online table: {e}")

    from databricks.sdk.service.catalog import (
        OnlineTable,
        OnlineTableSpec,
        OnlineTableSpecTriggeredSchedulingPolicy,
        PrimaryKeyConstraint,
    )

    online_table_spec = OnlineTableSpec(
        primary_key_columns=["row_id"],
        timeseries_key="event_time",
        run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
    )
    online_table = OnlineTable(
        name=online_table_name,
        spec=online_table_spec,
    )

    try:
        result = w.online_tables.create_and_wait(table=online_table)
        print(f"    Online table created: {result.name}")
        print(f"    Status: {result.status}")
    except Exception as e:
        print(f"    Online table creation result: {e}")
        # Try to get the current state
        try:
            current = w.online_tables.get(online_table_name)
            print(f"    Current online table state: {current.status}")
        except Exception as e2:
            print(f"    Could not get online table state: {e2}")

    print("\n=== Task Complete ===")
    print(f"Feature table: {FULL_TABLE}")
    print(f"Online table: {online_table_name}")
    print(f"Valid rows loaded: {len(valid_rows)}")
    print(f"Rejected row IDs: {rejected_ids}")
    print(f"Submission: {answers_path}")


if __name__ == "__main__":
    main()
