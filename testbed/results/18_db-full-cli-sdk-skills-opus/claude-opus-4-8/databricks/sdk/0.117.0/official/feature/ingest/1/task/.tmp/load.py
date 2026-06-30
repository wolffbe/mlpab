from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
WID = '4dfab06c923fe3cc'
tbl = 'workspace.mlpaba60366.transactions4ebd52'

def run(sql):
    r = w.statement_execution.execute_statement(warehouse_id=WID, statement=sql, wait_timeout='50s')
    if r.status.state != StatementState.SUCCEEDED:
        print('STATE', r.status.state)
        print('ERR', r.status.error)
        raise SystemExit('failed: ' + sql[:80])
    return r

run('DROP TABLE IF EXISTS ' + tbl)
run('''
CREATE TABLE ''' + tbl + ''' (
  row_id STRING NOT NULL,
  account_id STRING,
  event_time BIGINT,
  amount DOUBLE,
  category STRING,
  CONSTRAINT transactions4ebd52_pk PRIMARY KEY (row_id)
)
TBLPROPERTIES (delta.enableChangeDataFeed = true)
''')
print('table created')
run('''
INSERT INTO ''' + tbl + '''
SELECT row_id, account_id, event_time, amount, category FROM (
  SELECT
    CAST(row_id AS STRING) row_id,
    CAST(account_id AS STRING) account_id,
    CAST(event_time AS BIGINT) event_time,
    CAST(amount AS DOUBLE) amount,
    CAST(category AS STRING) category,
    ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY event_time) rn
  FROM read_files('/Volumes/workspace/mlpaba60366/staging/', format => 'csv', header => true)
  WHERE row_id IS NOT NULL
) QUALIFY rn = 1
''')
print('data inserted')
r = run('SELECT COUNT(*) c, COUNT(DISTINCT row_id) d, MIN(event_time) mn, MAX(event_time) mx FROM ' + tbl)
print('counts:', r.result.data_array)
