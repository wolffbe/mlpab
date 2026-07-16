"""Verify the feature table and online table state."""
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
import os, time

w = WorkspaceClient()
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
FULL_TABLE = f"{SCHEMA}.scaled7ecfaf"
WAREHOUSE_ID = "4dfab06c923fe3cc"

def exec_sql(statement, timeout=300):
    resp = w.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=WAREHOUSE_ID,
        wait_timeout="50s",
    )
    stmt_id = resp.statement_id
    start = time.time()
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        if time.time() - start > timeout:
            raise TimeoutError(f"SQL timed out after {timeout}s")
        time.sleep(3)
        resp = w.statement_execution.get_statement(stmt_id)
    if resp.status.state not in (StatementState.SUCCEEDED,):
        raise RuntimeError(f"SQL failed ({resp.status.state}): {resp.status.error}")
    return resp

# 1. Verify table exists and has correct data
print("=== Table Verification ===")
resp = exec_sql(f"SELECT split, COUNT(*) as cnt FROM {FULL_TABLE} GROUP BY split ORDER BY split")
print("Row counts by split:", resp.result.data_array)

# 2. Verify sample data
resp = exec_sql(f"SELECT * FROM {FULL_TABLE} ORDER BY row_id LIMIT 5")
print("\nFirst 5 rows:")
print("Columns:", resp.manifest.schema.columns if resp.manifest else "N/A")
for row in (resp.result.data_array or []):
    print(f"  {row}")

# 3. Verify columns
resp = exec_sql(f"DESCRIBE {FULL_TABLE}")
print("\nColumns:")
for row in (resp.result.data_array or []):
    print(f"  {row}")

# 4. Check feature_engineering API
print("\n=== Feature Engineering API ===")
print(dir(w.feature_engineering))

# 5. Check online table status
print("\n=== Online Table Status ===")
online_table_name = f"{SCHEMA}.scaled7ecfaf_online"
try:
    # Check via pipelines if there is a pipeline id
    import databricks.sdk.service.pipelines as pipelines
    # Try to check the pipeline 8df3448a-dc61-4e58-86b0-16ac818fdef4
    pipeline = w.pipelines.get("8df3448a-dc61-4e58-86b0-16ac818fdef4")
    print(f"Pipeline state: {pipeline.state}")
    print(f"Pipeline name: {pipeline.name}")
except Exception as e:
    print(f"Pipeline check error: {e}")

# 6. Check table via UC tables API
print("\n=== UC Table Info ===")
table_info = w.tables.get(FULL_TABLE)
print(f"Table: {table_info.full_name}")
print(f"Type: {table_info.table_type}")
print(f"Data source format: {table_info.data_source_format}")
print(f"Columns: {[c.name for c in (table_info.columns or [])]}")

# 7. Check if the online_table variant is also in UC
try:
    ot_info = w.tables.get(online_table_name)
    print(f"\nOnline table in UC: {ot_info.full_name}")
except Exception as e:
    print(f"\nOnline table not yet in UC: {e}")
