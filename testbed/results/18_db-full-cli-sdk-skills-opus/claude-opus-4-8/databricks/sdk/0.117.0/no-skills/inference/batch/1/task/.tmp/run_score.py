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


vol = '/Volumes/workspace/mlpabc69812/ingest/feature_history.csv'
create_sql = (
    "CREATE OR REPLACE TABLE workspace.mlpabc69812.scores3380ed "
    "TBLPROPERTIES (delta.enableChangeDataFeed = true) AS "
    "WITH raw AS (SELECT account_id, CAST(event_time AS BIGINT) event_time, "
    "CAST(f1 AS DOUBLE) f1, CAST(f2 AS DOUBLE) f2, CAST(f3 AS DOUBLE) f3 "
    "FROM read_files('" + vol + "', " + chr(102) + "ormat => 'csv', header => 'true')), "
    "valid AS (SELECT account_id, f1, f2, f3, "
    "ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) rn "
    "FROM raw WHERE event_time <= 1773489600000) "
    "SELECT account_id, "
    "ROUND(1.0/(1.0+EXP(-(-0.4933*f1 + -0.7922*f2 + 0.23*f3 + -0.051))), 6) AS score "
    "FROM valid WHERE rn = 1"
)
run(create_sql)
print('table created')
r = run('SELECT COUNT(*) c, COUNT(DISTINCT account_id) d FROM workspace.mlpabc69812.scores3380ed')
print('rows/distinct:', r.result.data_array)
r = run('SELECT * FROM workspace.mlpabc69812.scores3380ed ORDER BY account_id LIMIT 5')
print('sample:', r.result.data_array)
