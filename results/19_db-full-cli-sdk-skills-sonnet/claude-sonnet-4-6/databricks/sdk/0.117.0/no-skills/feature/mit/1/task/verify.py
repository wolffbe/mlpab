import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
schema_full = os.environ["MLPAB_DATABRICKS_SCHEMA"]
catalog, schema_name = schema_full.split(".", 1)
warehouses = list(w.warehouses.list())
wh_id = warehouses[0].id


def run_sql(sql):
    resp = w.statement_execution.execute_statement(
        statement=sql, warehouse_id=wh_id, catalog=catalog, schema=schema_name, wait_timeout="50s"
    )
    stmt_id = resp.statement_id
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(3)
        resp = w.statement_execution.get_statement(stmt_id)
    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed: {resp.status.error}")
    return resp


# Verify feature table
resp = run_sql(f"SELECT COUNT(*) FROM {catalog}.{schema_name}.featuresb1ea93")
print(f"Row count: {resp.result.data_array[0][0]}")

# Sample
resp = run_sql(f"SELECT * FROM {catalog}.{schema_name}.featuresb1ea93 LIMIT 5")
cols = [c.name for c in resp.manifest.schema.columns]
print(f"Columns: {cols}")
for row in resp.result.data_array:
    print(row)

# Check weekend distribution
resp = run_sql(
    f"SELECT is_weekend, COUNT(*) FROM {catalog}.{schema_name}.featuresb1ea93 GROUP BY is_weekend"
)
print("Weekend distribution:", resp.result.data_array)

# Verify synced table
st = w.database.get_synced_database_table(f"{catalog}.{schema_name}.featuresb1ea93_synced")
print(f"\nSynced table: {st.name}")
print(f"UC state: {st.unity_catalog_provisioning_state}")
print(f"DB instance: {st.effective_database_instance_name}")
print(f"Logical DB: {st.effective_logical_database_name}")
if st.data_synchronization_status:
    print(f"Sync state: {st.data_synchronization_status.detailed_state}")
    print(f"Message: {st.data_synchronization_status.message}")
    print(f"Pipeline ID: {st.data_synchronization_status.pipeline_id}")
