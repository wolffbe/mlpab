import time
import databricks.sdk as dsdk
from databricks.sdk.service.sql import StatementState

w = dsdk.WorkspaceClient()
WID = '4dfab06c923fe3cc'
CAT, SCH = 'workspace', 'mlpaba2af7c'
TBL = f'{CAT}.{SCH}.incrementalb48074'


def run(sql, wait=600):
    r = w.statement_execution.execute_statement(
        statement=sql, warehouse_id=WID, wait_timeout='50s')
    sid = r.statement_id
    st = r.status.state
    t0 = time.time()
    while st in (StatementState.PENDING, StatementState.RUNNING):
        if time.time() - t0 > wait:
            raise TimeoutError('sql timeout')
        time.sleep(3)
        r = w.statement_execution.get_statement(sid)
        st = r.status.state
    if st != StatementState.SUCCEEDED:
        raise RuntimeError(f'SQL failed: {st} :: {r.status.error}')
    return r
