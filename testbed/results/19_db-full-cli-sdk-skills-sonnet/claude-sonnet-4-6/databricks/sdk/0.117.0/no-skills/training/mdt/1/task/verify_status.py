"""Verify the feature table and online pipeline status."""
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState, ExecuteStatementRequestOnWaitTimeout

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
TABLE_NAME = "scaled7ecfaf"
FULL_TABLE = f"{SCHEMA}.{TABLE_NAME}"
STORE_NAME = PREFIX + "-store"
ONLINE_TABLE = f"{SCHEMA}.{TABLE_NAME}_online"

w = WorkspaceClient()
warehouses = list(w.warehouses.list())
warehouse_id = warehouses[0].id

def run_sql(sql, timeout=120):
    stmt = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout="50s",
        on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE,
    )
    stmt_id = stmt.statement_id
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = w.statement_execution.get_statement(stmt_id)
        state = result.status.state
        if state in (StatementState.SUCCEEDED, StatementState.FAILED,
                     StatementState.CANCELED, StatementState.CLOSED):
            if state != StatementState.SUCCEEDED:
                raise RuntimeError(f"SQL failed ({state}): {result.status.error}")
            return result
        time.sleep(2)
    raise TimeoutError("SQL timed out")

# Check the source Delta table
print("=== Source Delta Table ===")
count = run_sql(f"SELECT COUNT(*) FROM {FULL_TABLE}")
print(f"Row count: {count.result.data_array[0][0]}")

sample = run_sql(f"SELECT * FROM {FULL_TABLE} ORDER BY row_id LIMIT 5")
print("Sample rows:")
for row in sample.result.data_array:
    print(f"  {row}")

# Check table details
desc = run_sql(f"DESCRIBE TABLE EXTENDED {FULL_TABLE}")
print("Table schema:")
for row in desc.result.data_array:
    if row[0] in ["row_id", "split", "f1", "f2", "f3", "f4", "# Primary Key", "Type"]:
        print(f"  {row}")

# Check online store
print("\n=== Online Store ===")
store = w.feature_store.get_online_store(name=STORE_NAME)
print(f"Store: {store.name}, state: {store.state}, capacity: {store.capacity}")

# Check if online table exists
print(f"\n=== Online Table ({ONLINE_TABLE}) ===")
try:
    # Check via get
    result = w.api_client.do("GET", f"/api/2.0/online-tables/{ONLINE_TABLE}")
    print(f"Online table info: {result}")
except Exception as e:
    print(f"Error: {e}")

# Check pipeline status
print("\n=== Pipeline ===")
pipeline_id = "e90e30b2-455c-4a90-aa8d-f00f7fc02ab2"
try:
    pipeline = w.pipelines.get(pipeline_id=pipeline_id)
    print(f"Pipeline: {pipeline.name}, state: {pipeline.state}")
    latest = pipeline.latest_updates
    if latest:
        print(f"Latest update: {latest[0]}")
except Exception as e:
    print(f"Pipeline error: {e}")

print("\n=== Unity Catalog Table Check ===")
try:
    table_info = w.tables.get(full_name=FULL_TABLE)
    print(f"Table: {table_info.full_name}")
    print(f"  Type: {table_info.table_type}")
    print(f"  Columns: {[c.name for c in table_info.columns]}")
except Exception as e:
    print(f"Error: {e}")
