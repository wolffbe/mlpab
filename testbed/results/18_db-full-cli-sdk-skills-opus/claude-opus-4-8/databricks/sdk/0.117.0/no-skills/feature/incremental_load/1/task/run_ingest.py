import time
import databricks.sdk as dsdk
import databricks.sdk.service.sql as sqls

w = dsdk.WorkspaceClient()
WID = '4dfab06c923fe3cc'
CAT, SCH = 'workspace', 'mlpaba2af7c'
TBL = f'{CAT}.{SCH}.incrementalb48074'
S = sqls.StatementState


def run(sql, wait=600):
    r = w.statement_execution.execute_statement(
        statement=sql, warehouse_id=WID, wait_timeout='50s')
    sid = r.statement_id
    st = r.status.state
    t0 = time.time()
    while st in (S.PENDING, S.RUNNING):
        if time.time() - t0 > wait:
            raise TimeoutError('sql timeout')
        time.sleep(3)
        r = w.statement_execution.get_statement(sid)
        st = r.status.state
    if st != S.SUCCEEDED:
        raise RuntimeError(f'SQL failed: {st} :: {r.status.error}')
    return r


run(f'DROP TABLE IF EXISTS {TBL}')
ddl = f'''CREATE TABLE {TBL} (
  row_id STRING NOT NULL,
  account_id STRING,
  event_time BIGINT NOT NULL,
  amount DOUBLE,
  category STRING,
  CONSTRAINT incrementalb48074_pk PRIMARY KEY (row_id, event_time TIMESERIES)
) TBLPROPERTIES (delta.enableChangeDataFeed = true)'''
run(ddl)
print('table created with PK row_id + event_time TIMESERIES')

vol = f'/Volumes/{CAT}/{SCH}/inc_raw'
files = "','".join(f'increment_0{i}.csv' for i in range(1, 7))
copy = f'''COPY INTO {TBL}
FROM (SELECT row_id::STRING, account_id::STRING, event_time::BIGINT, amount::DOUBLE, category::STRING
      FROM '{vol}')
FILEFORMAT = CSV
FILES = ('{files}')
FORMAT_OPTIONS ('header'='true', 'inferSchema'='false')'''
run(copy)
print('COPY INTO done')

r = run(f'SELECT count(*) AS c, count(distinct row_id) AS d FROM {TBL}')
print('count rows / distinct row_id:', r.result.data_array)
