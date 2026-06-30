from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
WH = '4dfab06c923fe3cc'
FQN = 'workspace.mlpabc1d5e2.scaledd437a3'


def run(sql):
    r = w.statement_execution.execute_statement(warehouse_id=WH, statement=sql, wait_timeout='50s')
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f'{r.status.state}: {r.status.error}')
    return r


# row count + per-split means should be ~0 for train, std ~1 for train
r = run(f"SELECT count(*), count(DISTINCT row_id) FROM {FQN}")
print('total rows / distinct row_id:', r.result.data_array)
r = run(f"SELECT round(avg(f1),6), round(stddev_pop(f1),6) FROM {FQN} WHERE split='train'")
print('train f1 mean/std (expect ~0/~1):', r.result.data_array)
# constraint / PK
r = run(f"DESCRIBE TABLE EXTENDED {FQN}")
rows = r.result.data_array
for row in rows:
    if row[0] and ('Primary Key' in str(row[0]) or 'pk' in str(row[1]).lower()):
        print('constraint:', row)

# online table status
cur = w.database.get_synced_database_table(FQN + '_online')
print('online detailed_state:', cur.data_synchronization_status.detailed_state)
