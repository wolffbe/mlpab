import os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
WID = "4dfab06c923fe3cc"
CATALOG, SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
print("catalog/schema:", CATALOG, SCHEMA)


def sql(stmt, catalog=CATALOG, schema=SCHEMA):
    r = w.statement_execution.execute_statement(stmt, WID, catalog=catalog, schema=schema, wait_timeout="50s")
    sid = r.statement_id
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        r = w.statement_execution.get_statement(sid)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL FAILED: {r.status.state} :: {r.status.error} :: {stmt[:200]}")
    return r


# 1. create volume
sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.staging")
print("volume created")

vbase = f"/Volumes/{CATALOG}/{SCHEMA}/staging"
for fn in ["requests.csv", "profiles.csv"]:
    with open(f"data/{fn}", "rb") as f:
        w.files.upload(f"{vbase}/{fn}", f, overwrite=True)
    print("uploaded", fn)
print("DONE setup")
