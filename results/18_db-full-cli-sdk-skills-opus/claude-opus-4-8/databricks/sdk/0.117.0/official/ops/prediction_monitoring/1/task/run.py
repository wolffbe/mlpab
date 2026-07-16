import time
import json
import databricks.sdk
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
WH = "4dfab06c923fe3cc"
SCHEMA = "workspace.mlpab2ec61c"


def run_sql(stmt):
    r = w.statement_execution.execute_statement(
        warehouse_id=WH, statement=stmt, wait_timeout="50s")
    sid = r.statement_id
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        r = w.statement_execution.get_statement(sid)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed: {r.status.state} {r.status.error}")
    rows = []
    if r.result and r.result.data_array:
        rows = r.result.data_array
    cols = [c.name for c in r.manifest.schema.columns] if r.manifest and r.manifest.schema else []
    return cols, rows


# 1. Load CSV into a managed table
run_sql(f"""CREATE OR REPLACE TABLE {SCHEMA}.prediction_log AS
SELECT to_timestamp(ts) AS ts, CAST(prediction AS DOUBLE) AS prediction
FROM read_files('/Volumes/workspace/mlpab2ec61c/predmon/prediction_log.csv',
  format => 'csv', header => true)""")

cols, rows = run_sql(
    f"SELECT COUNT(*), MIN(ts), MAX(ts), ROUND(MIN(prediction),3), ROUND(MAX(prediction),3) FROM {SCHEMA}.prediction_log")
print("SUMMARY", cols, rows)
