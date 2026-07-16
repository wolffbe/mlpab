"""Check if Feature Engineering table registration can be done via table properties."""
import os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog, schema_name = schema.split('.')
WAREHOUSE_ID = "4dfab06c923fe3cc"
table_name = f"{schema}.featuresb1ea93"


def run_sql(statement, timeout_secs=60):
    resp = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=statement,
        catalog=catalog,
        schema=schema_name,
        wait_timeout="0s",
    )
    stmt_id = resp.statement_id
    start = time.time()
    while time.time() - start < timeout_secs:
        result = w.statement_execution.get_statement(stmt_id)
        state = result.status.state
        if state in (StatementState.SUCCEEDED, StatementState.FAILED,
                     StatementState.CANCELED, StatementState.CLOSED):
            if state != StatementState.SUCCEEDED:
                raise RuntimeError(f"SQL failed ({state}): {result.status.error}")
            return result
        time.sleep(2)
    raise TimeoutError("SQL timed out")


# Check current table properties
print("Current table properties:")
r = run_sql(f"SHOW TBLPROPERTIES {table_name}")
if r.result and r.result.data_array:
    for row in r.result.data_array:
        print(f"  {row}")
else:
    print("  (none)")

print()

# Try to set Feature Engineering properties
# These are the properties that the Feature Engineering client would set
print("Setting Feature Engineering table properties...")
try:
    r = run_sql(f"""
    ALTER TABLE {table_name} SET TBLPROPERTIES (
        'feature_store.feature_table' = 'true',
        'feature_store.primary_keys' = 'row_id',
        'feature_store.timestamp_keys' = 'event_time',
        'feature_store.description' = 'Transaction features: amount_usd, is_weekend, amount_7d'
    )
    """)
    print("Properties set!")
except Exception as e:
    print(f"Error setting properties: {e}")

# Check current properties again
print("\nUpdated table properties:")
r = run_sql(f"SHOW TBLPROPERTIES {table_name}")
if r.result and r.result.data_array:
    for row in r.result.data_array:
        print(f"  {row}")

# Check table details
print("\nTable details via SDK:")
try:
    table_info = w.tables.get(full_name=table_name)
    print(f"  Name: {table_info.full_name}")
    print(f"  Type: {table_info.table_type}")
    print(f"  Properties: {table_info.properties}")
    print(f"  Columns: {[(c.name, c.type_text) for c in (table_info.columns or [])]}")
except Exception as e:
    print(f"  Error: {e}")
