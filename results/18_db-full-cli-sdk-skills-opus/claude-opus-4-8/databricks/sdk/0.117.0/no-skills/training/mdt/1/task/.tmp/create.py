from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
WH = '4dfab06c923fe3cc'
CAT = 'workspace'
SCH = 'mlpabc1d5e2'
TBL = 'scaledd437a3'
FQN = f'{CAT}.{SCH}.{TBL}'
base = '/Volumes/workspace/mlpabc1d5e2/taskdata'


def run(sql):
    r = w.statement_execution.execute_statement(warehouse_id=WH, statement=sql, wait_timeout='50s')
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f'{r.status.state}: {r.status.error}\nSQL: {sql[:200]}')
    return r


run(f"DROP TABLE IF EXISTS {FQN}")

feats = ['f1', 'f2', 'f3', 'f4']
stat_cols = ", ".join(
    [f"avg({f}) AS m_{f}, stddev_pop({f}) AS s_{f}" for f in feats]
)
std_cols = ", ".join(
    [f"round((c.{f} - s.m_{f}) / s.s_{f}, 6) AS {f}" for f in feats]
)

ctas = f"""
CREATE TABLE {FQN}
TBLPROPERTIES (delta.enableChangeDataFeed = true)
AS
WITH train AS (
  SELECT CAST(row_id AS STRING) row_id, CAST(f1 AS DOUBLE) f1, CAST(f2 AS DOUBLE) f2,
         CAST(f3 AS DOUBLE) f3, CAST(f4 AS DOUBLE) f4
  FROM read_files('{base}/features_train.csv', format=>'csv', header=>true)
),
serve AS (
  SELECT CAST(row_id AS STRING) row_id, CAST(f1 AS DOUBLE) f1, CAST(f2 AS DOUBLE) f2,
         CAST(f3 AS DOUBLE) f3, CAST(f4 AS DOUBLE) f4
  FROM read_files('{base}/features_serve.csv', format=>'csv', header=>true)
),
stats AS (SELECT {stat_cols} FROM train),
combined AS (
  SELECT row_id, 'train' AS split, f1, f2, f3, f4 FROM train
  UNION ALL
  SELECT row_id, 'serve' AS split, f1, f2, f3, f4 FROM serve
)
SELECT c.row_id, c.split, {std_cols}
FROM combined c CROSS JOIN stats s
"""
run(ctas)
print('table created')

run(f"ALTER TABLE {FQN} ALTER COLUMN row_id SET NOT NULL")
run(f"ALTER TABLE {FQN} ADD CONSTRAINT {TBL}_pk PRIMARY KEY (row_id)")
print('primary key added')

r = run(f"SELECT split, count(*) FROM {FQN} GROUP BY split ORDER BY split")
print('counts:', r.result.data_array)
r = run(f"SELECT * FROM {FQN} WHERE row_id IN ('R00000','R00400') ORDER BY row_id")
print('cols:', [c.name for c in r.manifest.schema.columns])
print('sample:', r.result.data_array)
