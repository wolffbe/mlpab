from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
WH = '4dfab06c923fe3cc'
cat, sch = 'workspace', 'mlpabbaf12e'
tbl = f'{cat}.{sch}.events88b330'
path = f'/Volumes/{cat}/{sch}/ingest/events.csv'


def run(sql):
    r = w.statement_execution.execute_statement(warehouse_id=WH, statement=sql, wait_timeout='50s')
    st = r.status.state
    if st != StatementState.SUCCEEDED:
        raise RuntimeError(f'{st}: {r.status.error}')
    return r


run(f"""
CREATE OR REPLACE TABLE {tbl} (
  row_id STRING NOT NULL,
  account_id STRING,
  event_time BIGINT,
  amount DOUBLE,
  category STRING,
  CONSTRAINT events88b330_pk PRIMARY KEY (row_id)
)
TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
print('table created')

run(f"""
INSERT INTO {tbl}
SELECT row_id, account_id,
       CAST(event_time AS BIGINT) AS event_time,
       CAST(amount AS DOUBLE) AS amount,
       category
FROM read_files('{path}', format=>'csv', header=>true,
     schema=>'row_id string, account_id string, event_time string, amount string, category string')
WHERE amount IS NOT NULL AND trim(amount) <> ''
  AND CAST(amount AS DOUBLE) IS NOT NULL
  AND CAST(amount AS DOUBLE) BETWEEN 0 AND 10000
  AND category IN ('grocery','travel','salary','rent','other')
""")
print('valid rows inserted')

r = run(f"SELECT count(*) FROM {tbl}")
print('loaded count:', r.result.data_array[0][0])
