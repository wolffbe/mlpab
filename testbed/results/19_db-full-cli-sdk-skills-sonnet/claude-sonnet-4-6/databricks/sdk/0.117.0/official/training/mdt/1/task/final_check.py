"""Final verification of the feature table and online table."""
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
import os, time

w = WorkspaceClient()
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
FULL_TABLE = f"{SCHEMA}.scaled7ecfaf"
ONLINE_TABLE = f"{SCHEMA}.scaled7ecfaf_online"
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

print("=== FINAL VERIFICATION ===\n")

# 1. Offline table
print(f"1. Offline Delta table: {FULL_TABLE}")
table_info = w.tables.get(FULL_TABLE)
print(f"   Type: {table_info.table_type}")
print(f"   Format: {table_info.data_source_format}")
print(f"   Columns: {[c.name for c in (table_info.columns or [])]}")

resp = exec_sql(f"SELECT split, COUNT(*) FROM {FULL_TABLE} GROUP BY split ORDER BY split")
for row in (resp.result.data_array or []):
    print(f"   {row[0]}: {row[1]} rows")

# Sample a few values to confirm standardization
resp = exec_sql(f"""
SELECT row_id, split, f1, f2, f3, f4 FROM {FULL_TABLE}
WHERE row_id IN ('R00000', 'R00400')
ORDER BY row_id
""")
print("\n   Sample standardized rows:")
for row in (resp.result.data_array or []):
    print(f"     {row}")

# 2. Online table
print(f"\n2. Online table: {ONLINE_TABLE}")
online_table_info = w.tables.get(ONLINE_TABLE)
print(f"   Type: {online_table_info.table_type}")
print(f"   Columns: {[c.name for c in (online_table_info.columns or [])]}")

# Pipeline status
pipeline = w.pipelines.get("8df3448a-dc61-4e58-86b0-16ac818fdef4")
print(f"   Sync pipeline state: {pipeline.state}")

print("\n=== DONE ===")
print(f"Feature table: {FULL_TABLE}")
print(f"Online table: {ONLINE_TABLE}")
print("Both are available on the Databricks platform.")
