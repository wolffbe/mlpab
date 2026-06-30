import os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
WID = "4dfab06c923fe3cc"
CATALOG, SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
FQ = f"{CATALOG}.{SCHEMA}"
PIPELINE_ID = "a34330d1-5280-40e1-9a18-55750b6a6646"


def sql(stmt):
    r = w.statement_execution.execute_statement(stmt, WID, catalog=CATALOG, schema=SCHEMA, wait_timeout="50s")
    sid = r.statement_id
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        r = w.statement_execution.get_statement(sid)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL FAILED: {r.status.state} :: {r.status.error}")
    return r

# columns of the feature table
r = sql(f"DESCRIBE TABLE {FQ}.scoredbfc4ef")
print("columns:", [(row[0], row[1]) for row in r.result.data_array if row[0] and not row[0].startswith('#')])

# manual cross-check of one row
r = sql(f"""
SELECT r.request_id, r.account_id,
  ROUND(SQRT(POWER(r.request_lat-p.home_lat,2)+POWER(r.request_lon-p.home_lon,2)),6) AS d,
  ROUND(p.base_score - 0.1*ROUND(SQRT(POWER(r.request_lat-p.home_lat,2)+POWER(r.request_lon-p.home_lon,2)),6),6) AS s
FROM {FQ}.stg_requests r JOIN {FQ}.stg_profiles p ON r.account_id=p.account_id
WHERE r.request_id='Q00000'""")
print("recompute Q00000:", r.result.data_array)
r = sql(f"SELECT request_id,account_id,distance_deg,score FROM {FQ}.scoredbfc4ef WHERE request_id='Q00000'")
print("stored   Q00000:", r.result.data_array)

# online sync pipeline status
try:
    p = w.pipelines.get(PIPELINE_ID)
    print("pipeline state:", p.state)
    upd = w.pipelines.list_updates(PIPELINE_ID)
    if upd.updates:
        print("latest update state:", upd.updates[0].state)
except Exception as e:
    print("pipeline check err:", e)
print("DONE verify")
