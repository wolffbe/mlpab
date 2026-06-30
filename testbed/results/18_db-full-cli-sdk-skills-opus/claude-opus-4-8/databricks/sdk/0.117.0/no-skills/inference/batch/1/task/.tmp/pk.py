from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
WH = '4dfab06c923fe3cc'


def run(sql):
    r = w.statement_execution.execute_statement(warehouse_id=WH, statement=sql, wait_timeout='50s')
    st = r.status.state
    while st in (StatementState.PENDING, StatementState.RUNNING):
        r = w.statement_execution.get_statement(r.statement_id)
        st = r.status.state
    if st != StatementState.SUCCEEDED:
        raise RuntimeError(str(r.status.error))
    return r


t = 'workspace.mlpabc69812.scores3380ed'
run('ALTER TABLE ' + t + ' ALTER COLUMN account_id SET NOT NULL')
run('ALTER TABLE ' + t + ' ADD CONSTRAINT scores3380ed_pk PRIMARY KEY (account_id)')
print('primary key set')
r = run('DESCRIBE TABLE EXTENDED ' + t)
for row in r.result.data_array:
    print(row)
