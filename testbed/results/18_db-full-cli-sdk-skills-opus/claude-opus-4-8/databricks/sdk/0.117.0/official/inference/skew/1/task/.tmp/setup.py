import databricks.sdk
from databricks.sdk.service.sql import StatementState
import time, io

w = databricks.sdk.WorkspaceClient()
SCHEMA = "workspace.mlpab9f3ba9"
CATALOG, SCHEMANAME = SCHEMA.split(".")
WH = "4dfab06c923fe3cc"


def run_sql(sql):
    r = w.statement_execution.execute_statement(warehouse_id=WH, statement=sql, wait_timeout="50s")
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed: {r.status.state} {r.status.error}")
    return r


# create volume
run_sql(f"CREATE VOLUME IF NOT EXISTS {SCHEMA}.skewvol")
print("volume created")

# upload CSVs to volume
for fn in ["training_sample.csv", "serving_log.csv"]:
    with open(f"data/{fn}", "rb") as f:
        data = f.read()
    path = f"/Volumes/{CATALOG}/{SCHEMANAME}/skewvol/{fn}"
    w.files.upload(path, io.BytesIO(data), overwrite=True)
    print("uploaded", path)
