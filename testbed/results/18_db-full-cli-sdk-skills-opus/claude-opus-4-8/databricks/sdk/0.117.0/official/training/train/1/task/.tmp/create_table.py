import os
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
cat, sch = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')
wh = '4dfab06c923fe3cc'
tbl = f'{cat}.{sch}.predictionsa834e5'
vol_csv = f'/Volumes/{cat}/{sch}/trainvola834e5/predictions.csv'


def run(sql):
    r = w.statement_execution.execute_statement(warehouse_id=wh, statement=sql, wait_timeout='50s')
    st = r.status.state
    print('->', st)
    if str(st) not in ('StatementState.SUCCEEDED',):
        print('  ', r.status.error)
    return r


run(f'DROP TABLE IF EXISTS {tbl}')
run(f'''CREATE TABLE {tbl} (
  row_id STRING NOT NULL,
  score DOUBLE,
  CONSTRAINT predictionsa834e5_pk PRIMARY KEY (row_id)
) TBLPROPERTIES (delta.enableChangeDataFeed = true)''')
run(f'''INSERT INTO {tbl}
SELECT CAST(row_id AS STRING), CAST(score AS DOUBLE)
FROM read_files('{vol_csv}', format => 'csv', header => true)''')
r = run(f'SELECT count(*) AS n, min(score) AS mn, max(score) AS mx FROM {tbl}')
print('table stats:', r.result.data_array)
