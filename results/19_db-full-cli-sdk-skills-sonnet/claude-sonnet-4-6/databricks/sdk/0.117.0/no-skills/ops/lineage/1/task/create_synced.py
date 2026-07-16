import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
from databricks.sdk.service.database import SyncedDatabaseTable, SyncedTableSpec, SyncedTableSchedulingPolicy

w = WorkspaceClient()
schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]
catalog_name, schema_name = schema.split(".", 1)

warehouses = list(w.warehouses.list())
warehouse_id = warehouses[0].id if warehouses else None

def exec_sql(statement):
    resp = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=statement,
        wait_timeout="50s",
    )
    terminal = {StatementState.SUCCEEDED, StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED}
    while resp.status and resp.status.state not in terminal:
        time.sleep(2)
        resp = w.statement_execution.get_statement(resp.statement_id)
    if resp.status and resp.status.state != StatementState.SUCCEEDED:
        err = resp.status.error.message if resp.status.error else "unknown error"
        raise RuntimeError(f"SQL error: {err}")
    return resp

# Enable CDF on derived table
print("Enabling Change Data Feed on derivedd05474...")
exec_sql(f"ALTER TABLE `{catalog_name}`.`{schema_name}`.`derivedd05474` SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print("CDF enabled.")

instance_name = f"{prefix}lakebase"
synced_table_name = f"{catalog_name}.{schema_name}.derivedd05474_synced"

print(f"Creating synced table: {synced_table_name}")
try:
    spec = SyncedTableSpec(
        source_table_full_name=f"{catalog_name}.{schema_name}.derivedd05474",
        primary_key_columns=["row_id"],
        scheduling_policy=SyncedTableSchedulingPolicy.TRIGGERED,
    )
    synced_table = w.database.create_synced_database_table(
        SyncedDatabaseTable(
            name=synced_table_name,
            database_instance_name=instance_name,
            logical_database_name=schema_name,
            spec=spec,
        )
    )
    print(f"Synced table created: {synced_table}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
