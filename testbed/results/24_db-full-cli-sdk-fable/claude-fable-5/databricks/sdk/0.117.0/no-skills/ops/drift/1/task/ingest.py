import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.<run-id>
cat, sch = schema.split(".")
WAREHOUSES = ["a832b544eb7dc3fe", "8a93fc195da2ceb1"]


def sql(stmt, timeout="50s"):
    last_err = None
    for attempt in range(12):
        wh_id = WAREHOUSES[attempt % len(WAREHOUSES)]
        try:
            r = w.statement_execution.execute_statement(
                statement=stmt, warehouse_id=wh_id, wait_timeout=timeout
            )
            while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
                time.sleep(5)
                r = w.statement_execution.get_statement(r.statement_id)
            if r.status.state == StatementState.SUCCEEDED:
                return r
            last_err = RuntimeError(f"{r.status.state}: {r.status.error}")
        except Exception as e:
            last_err = e
        print(f"attempt {attempt} on {wh_id} failed: {last_err}", flush=True)
        time.sleep(20)
    raise last_err


sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
sql(f"CREATE VOLUME IF NOT EXISTS {schema}.drift_vol")
with open("data/features.csv", "rb") as f:
    w.files.upload(f"/Volumes/{cat}/{sch}/drift_vol/features.csv", f, overwrite=True)
print("uploaded")

sql(
    f"""CREATE OR REPLACE TABLE {schema}.features AS
SELECT entity_id, CAST(event_time AS TIMESTAMP) AS event_time,
       CAST(f1 AS DOUBLE) f1, CAST(f2 AS DOUBLE) f2, CAST(f3 AS DOUBLE) f3,
       CAST(f4 AS DOUBLE) f4, CAST(f5 AS DOUBLE) f5, CAST(f6 AS DOUBLE) f6
FROM read_files('/Volumes/{cat}/{sch}/drift_vol/features.csv',
  format => 'csv', header => true)"""
)
r = sql(f"SELECT COUNT(*), MIN(event_time), MAX(event_time) FROM {schema}.features")
print(r.result.data_array)
