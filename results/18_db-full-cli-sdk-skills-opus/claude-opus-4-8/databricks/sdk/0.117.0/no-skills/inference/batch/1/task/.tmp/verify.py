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


# feature table columns & counts
r = run("SELECT * FROM workspace.mlpabc69812.scores3380ed ORDER BY account_id")
print('columns:', [c.name for c in r.manifest.schema.columns])
rows = r.result.data_array
print('row count:', len(rows))
print('first 3:', rows[:3])
print('last 3:', rows[-3:])

# synced table final state
cur = w.database.get_synced_database_table('workspace.mlpabc69812.scores3380ed_online')
print('synced state:', cur.data_synchronization_status.detailed_state)

# feature_store / feature_engineering registration check
print('feature_store methods:', [m for m in dir(w.feature_store) if not m.startswith('_')])
