import os
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
cat, sch = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')
WH = '8a93fc195da2ceb1'
FQN = f'{cat}.{sch}.scaledf9e607'
VOL = f'/Volumes/{cat}/{sch}/raw'

def sql(stmt):
    r = w.statement_execution.execute_statement(statement=stmt, warehouse_id=WH, wait_timeout='50s')
    if r.status.state.value != 'SUCCEEDED':
        raise RuntimeError(f'{r.status.state}: {r.status.error}')
    return r

sql(f"""
CREATE TABLE {FQN} (
  row_id STRING NOT NULL,
  split STRING,
  f1 DOUBLE, f2 DOUBLE, f3 DOUBLE, f4 DOUBLE,
  CONSTRAINT scaledf9e607_pk PRIMARY KEY (row_id)
) TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
print('table created')

sql(f"""
WITH train AS (
  SELECT row_id, CAST(f1 AS DOUBLE) f1, CAST(f2 AS DOUBLE) f2, CAST(f3 AS DOUBLE) f3, CAST(f4 AS DOUBLE) f4
  FROM read_files('{VOL}/features_train.csv', format => 'csv', header => true)
),
serve AS (
  SELECT row_id, CAST(f1 AS DOUBLE) f1, CAST(f2 AS DOUBLE) f2, CAST(f3 AS DOUBLE) f3, CAST(f4 AS DOUBLE) f4
  FROM read_files('{VOL}/features_serve.csv', format => 'csv', header => true)
),
allrows AS (
  SELECT row_id, 'train' AS split, f1, f2, f3, f4 FROM train
  UNION ALL
  SELECT row_id, 'serve' AS split, f1, f2, f3, f4 FROM serve
),
stats AS (
  SELECT avg(f1) m1, stddev_pop(f1) s1,
         avg(f2) m2, stddev_pop(f2) s2,
         avg(f3) m3, stddev_pop(f3) s3,
         avg(f4) m4, stddev_pop(f4) s4
  FROM train
)
INSERT INTO {FQN}
SELECT row_id, split,
       round((f1 - m1) / s1, 6),
       round((f2 - m2) / s2, 6),
       round((f3 - m3) / s3, 6),
       round((f4 - m4) / s4, 6)
FROM allrows CROSS JOIN stats
""")
print('data inserted')

r = sql(f"SELECT split, count(*) c, round(avg(f1),4), round(stddev_pop(f1),4) FROM {FQN} GROUP BY split ORDER BY split")
for row in r.result.data_array:
    print(row)
r = sql(f"SELECT * FROM {FQN} ORDER BY row_id LIMIT 3")
for row in r.result.data_array:
    print(row)
