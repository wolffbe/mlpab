"""Quick verify the feature table."""
import os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog, schema_name = schema.split('.')
WAREHOUSE_ID = "4dfab06c923fe3cc"


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


r = run_sql(f"SELECT COUNT(*) as cnt FROM {schema}.featuresb1ea93")
count = r.result.data_array[0][0] if r.result and r.result.data_array else "?"
print(f"Row count: {count}")

r = run_sql(f"SELECT * FROM {schema}.featuresb1ea93 LIMIT 3")
print("Sample rows:")
if r.result and r.result.data_array:
    for row in r.result.data_array:
        print(f"  {row}")

print("\nSynced table status:")
st_info = w.postgres.get_synced_table(name=f"synced_tables/{catalog}.{schema_name}.featuresb1ea93_online")
print(f"  State: {st_info.status.detailed_state}")
print(f"  UC state: {st_info.status.unity_catalog_provisioning_state}")
