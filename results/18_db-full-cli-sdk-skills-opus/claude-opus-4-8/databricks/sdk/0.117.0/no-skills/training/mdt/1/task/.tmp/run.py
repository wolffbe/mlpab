from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
WH = '4dfab06c923fe3cc'


def run(sql):
    r = w.statement_execution.execute_statement(warehouse_id=WH, statement=sql, wait_timeout='50s')
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f'{r.status.state}: {r.status.error}')
    return r


base = '/Volumes/workspace/mlpabc1d5e2/taskdata'
r = run(f"SELECT count(*), avg(f1), stddev_pop(f1) FROM read_files('{base}/features_train.csv', format=>'csv', header=>true)")
print('train stats f1:', r.result.data_array)
r = run(f"SELECT count(*) FROM read_files('{base}/features_serve.csv', format=>'csv', header=>true)")
print('serve count:', r.result.data_array)
