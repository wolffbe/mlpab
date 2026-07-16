import os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
WH = '4dfab06c923fe3cc'
SCHEMA = os.environ['MLPAB_DATABRICKS_SCHEMA']   # workspace.mlpab6c699a
CAT, SCH = SCHEMA.split('.')

def sql(q):
    r = w.statement_execution.execute_statement(warehouse_id=WH, statement=q, wait_timeout='50s')
    st = r.status.state
    while st in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        r = w.statement_execution.get_statement(r.statement_id)
        st = r.status.state
    if st != StatementState.SUCCEEDED:
        raise RuntimeError(f'{st}: {r.status.error}')
    return r

def rows(r):
    return (r.result.data_array if r.result else None)
