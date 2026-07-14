from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
WH = '4dfab06c923fe3cc'
CAT, SCH = 'workspace', 'mlpab35a617'
base = f'/Volumes/{CAT}/{SCH}/ingest'


def sql(q):
    r = w.statement_execution.execute_statement(warehouse_id=WH, statement=q, wait_timeout='50s')
    st = r.status.state
    while st in (StatementState.PENDING, StatementState.RUNNING):
        r = w.statement_execution.get_statement(r.statement_id)
        st = r.status.state
    if st != StatementState.SUCCEEDED:
        raise RuntimeError(f'FAILED: {st} :: {r.status.error} :: {q[:120]}')
    return r


t = f'{CAT}.{SCH}.customers302d18_1'
sql(f'''CREATE OR REPLACE TABLE {t} (
  row_id STRING NOT NULL,
  name STRING,
  balance_eur DOUBLE,
  updated_at BIGINT NOT NULL
) TBLPROPERTIES (delta.enableChangeDataFeed = true)''')
print('created v1')
sql(f'''INSERT INTO {t}
  SELECT row_id, name, balance_eur, updated_at
  FROM read_files('{base}/initial_export.csv', format=>'csv', header=>true,
    schema=>'row_id string, name string, balance_eur double, updated_at bigint')''')
print('loaded v1')
sql(f'ALTER TABLE {t} ADD CONSTRAINT customers302d18_1_pk PRIMARY KEY (row_id, updated_at TIMESERIES)')
print('pk v1')
r = sql(f'SELECT count(*) FROM {t}')
print('v1 rowcount:', r.result.data_array)
