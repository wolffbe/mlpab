import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
CATALOG_NAME, SCHEMA_NAME = SCHEMA.split(".", 1)
FULL_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.eventsd693d3"
WH_ID = "4dfab06c923fe3cc"

w = WorkspaceClient()


def run_sql(sql):
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=WH_ID,
        catalog=CATALOG_NAME,
        schema=SCHEMA_NAME,
        wait_timeout="50s",
    )
    stmt_id = resp.statement_id
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(3)
        resp = w.statement_execution.get_statement(stmt_id)
    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed: {resp.status.error}")
    return resp


resp = run_sql(f"SELECT COUNT(*) AS n FROM {FULL_TABLE}")
count = resp.result.data_array[0][0]
print(f"ROW_COUNT={count}")

resp2 = run_sql(f"DESCRIBE TABLE EXTENDED {FULL_TABLE}")
for row in resp2.result.data_array or []:
    if row and row[0] in ("row_id", "event_time", "amount", "category", "delta.enableChangeDataFeed", "feature_store.record_key"):
        print(f"  {row}")

synced = w.database.get_synced_database_table(f"{FULL_TABLE}_synced")
print(f"SYNCED_TABLE_STATE={synced.data_synchronization_status.detailed_state}")
