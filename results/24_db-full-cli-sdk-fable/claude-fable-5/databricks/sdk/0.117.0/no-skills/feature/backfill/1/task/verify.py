import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
CATALOG, SCHEMA_NAME = SCHEMA.split(".")
TABLE = f"{SCHEMA}.accounts06d84b"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA_NAME}/staging_accounts06d84b"

w = WorkspaceClient()

stmt = f"""
WITH raw AS (
  SELECT row_id, updated_at FROM read_files(
    '{VOLUME_PATH}/', format => 'csv', header => true,
    schema => 'row_id STRING, status STRING, balance DOUBLE, updated_at BIGINT')
),
expected AS (SELECT row_id, max(updated_at) AS max_ts FROM raw GROUP BY row_id)
SELECT
  (SELECT count(*) FROM {TABLE}) AS table_rows,
  (SELECT count(*) FROM expected) AS expected_rows,
  (SELECT count(*) FROM {TABLE} t JOIN expected e
     ON t.row_id = e.row_id AND t.updated_at = e.max_ts) AS matching_latest
"""
r = w.statement_execution.execute_statement(
    warehouse_id="8a93fc195da2ceb1", statement=stmt, wait_timeout="50s"
)
while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
    time.sleep(3)
    r = w.statement_execution.get_statement(r.statement_id)
print(r.status.state, r.result.data_array)

st = w.database.get_synced_database_table(f"{SCHEMA}.accounts06d84b_online")
print("online:", st.name, st.data_synchronization_status.detailed_state)
