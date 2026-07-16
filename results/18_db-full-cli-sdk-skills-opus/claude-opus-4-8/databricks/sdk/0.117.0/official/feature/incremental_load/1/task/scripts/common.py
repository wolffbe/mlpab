import os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
WH = '4dfab06c923fe3cc'
SCHEMA = os.environ['MLPAB_DATABRICKS_SCHEMA']  # workspace.mlpabb905af
CAT, SCH = SCHEMA.split('.')
PREFIX = os.environ['MLPAB_DATABRICKS_PREFIX']
USER = w.current_user.me().user_name


def run(sql, warehouse=WH):
    r = w.statement_execution.execute_statement(warehouse_id=warehouse, statement=sql, wait_timeout='50s')
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL FAILED: {r.status.state} {r.status.error}\nSQL: {sql[:300]}")
    return r.result
