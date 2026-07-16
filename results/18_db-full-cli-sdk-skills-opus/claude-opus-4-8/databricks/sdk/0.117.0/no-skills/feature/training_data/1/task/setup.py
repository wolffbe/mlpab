import os, io, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
WH = "4dfab06c923fe3cc"
catalog, schema = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')
print("catalog/schema:", catalog, schema)


def sql(stmt, wait=True):
    r = w.statement_execution.execute_statement(statement=stmt, warehouse_id=WH,
            catalog=catalog, schema=schema, wait_timeout="50s")
    sid = r.statement_id
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        r = w.statement_execution.get_statement(sid)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"FAILED: {r.status.state} {r.status.error}")
    return r


# create volume for raw data
sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.rawdata")
print("volume created")

# upload csvs
vol_base = f"/Volumes/{catalog}/{schema}/rawdata"
for f in ["transactions", "profiles", "activity", "account_health", "transactions_late", "labels"]:
    with open(f"data/{f}.csv", "rb") as fh:
        data = fh.read()
    w.files.upload(f"{vol_base}/{f}.csv", io.BytesIO(data), overwrite=True)
    print("uploaded", f, len(data))
print("DONE")
