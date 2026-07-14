import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
wh = next(c for c in w.warehouses.list() if "serverless" in c.name.lower())


def sql(stmt):
    r = w.statement_execution.execute_statement(
        warehouse_id=wh.id, statement=stmt, wait_timeout="50s"
    )
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(3)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed: {r.status.state} {r.status.error}")
    return r


r = sql("""
SELECT
  (SELECT COUNT(*) FROM workspace.mlpab4fd108.scoresbedd56) AS offline_rows,
  (SELECT COUNT(*) FROM workspace.mlpab4fd108.scoresbedd56_online) AS online_rows,
  (SELECT COUNT(*) FROM workspace.mlpab4fd108.scoresbedd56 a
     JOIN workspace.mlpab4fd108.scoresbedd56_online b USING (account_id)
    WHERE a.score = b.score) AS matching
""")
print("offline/online/matching:", r.result.data_array)
