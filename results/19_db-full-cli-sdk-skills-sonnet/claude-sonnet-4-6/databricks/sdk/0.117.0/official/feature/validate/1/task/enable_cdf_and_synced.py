import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
from databricks.sdk.service.database import (
    SyncedDatabaseTable,
    SyncedTableSpec,
    SyncedTableSchedulingPolicy,
)

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
CATALOG_NAME, SCHEMA_NAME = SCHEMA.split(".", 1)
FULL_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.eventsd693d3"
WH_ID = "4dfab06c923fe3cc"
DB_INSTANCE = "mlpabdbae68-online"
LOGICAL_DB = "mlpabdbae68_db"

w = WorkspaceClient()


def run_sql(sql, description=""):
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=WH_ID,
        catalog=CATALOG_NAME,
        schema=SCHEMA_NAME,
        wait_timeout="50s",
    )
    stmt_id = resp.statement_id
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(5)
        resp = w.statement_execution.get_statement(stmt_id)
    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed ({resp.status.state}): {resp.status.error}")
    return resp


# Enable Change Data Feed
run_sql(
    f"ALTER TABLE {FULL_TABLE} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')",
    "enable CDF",
)

# Create synced table
synced = w.database.create_synced_database_table(
    synced_table=SyncedDatabaseTable(
        name=f"{FULL_TABLE}_synced",
        database_instance_name=DB_INSTANCE,
        logical_database_name=LOGICAL_DB,
        spec=SyncedTableSpec(
            source_table_full_name=FULL_TABLE,
            primary_key_columns=["row_id"],
            timeseries_key="event_time",
            scheduling_policy=SyncedTableSchedulingPolicy.TRIGGERED,
            create_database_objects_if_missing=True,
        ),
    )
)

print(f"SYNCED_TABLE_NAME={synced.name}")
print(f"SYNCED_TABLE_STATUS={synced.data_synchronization_status}")
