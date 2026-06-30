from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
import time, sys
w = WorkspaceClient(); WH = '4dfab06c923fe3cc'
def run(sql, fetch=False):
    r = w.statement_execution.execute_statement(warehouse_id=WH, statement=sql, wait_timeout='50s')
    sid = r.statement_id
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2); r = w.statement_execution.get_statement(sid)
    if r.status.state != StatementState.SUCCEEDED:
        print('FAIL', r.status.error); sys.exit(1)
    return r.result.data_array if (fetch and r.result and r.result.data_array) else []
C = 'workspace.mlpab6d0586'
print('feature table rows:', run(f"SELECT count(*) FROM {C}.events88b330", fetch=True))
print('constraints:')
for row in run(f"SELECT constraint_name, constraint_type FROM information_schema.table_constraints WHERE table_schema='mlpab6d0586' AND table_name='events88b330'", fetch=True):
    print(' ', row)
print('online synced table rows:', run(f"SELECT count(*) FROM {C}.events88b330_online", fetch=True))
