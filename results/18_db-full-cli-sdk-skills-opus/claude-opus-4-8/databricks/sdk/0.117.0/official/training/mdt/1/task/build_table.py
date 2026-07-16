from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
import time, sys

w = WorkspaceClient()
WH = "4dfab06c923fe3cc"


def run(sql):
    r = w.statement_execution.execute_statement(statement=sql, warehouse_id=WH, wait_timeout="50s")
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        print("FAILED:", r.status.state, r.status.error)
        sys.exit(1)
    return r.result.data_array if r.result else None


base = "/Volumes/workspace/mlpabf94ccc/ingest"
tbl = "workspace.mlpabf94ccc.scaledd437a3"

sql_create = """
CREATE OR REPLACE TABLE {tbl}
TBLPROPERTIES (delta.enableChangeDataFeed = true) AS
WITH train AS (
  SELECT cast(row_id AS string) AS row_id, cast(f1 AS double) f1, cast(f2 AS double) f2, cast(f3 AS double) f3, cast(f4 AS double) f4
  FROM read_files('{base}/train.csv', format => 'csv', header => true)
),
serve AS (
  SELECT cast(row_id AS string) AS row_id, cast(f1 AS double) f1, cast(f2 AS double) f2, cast(f3 AS double) f3, cast(f4 AS double) f4
  FROM read_files('{base}/serve.csv', format => 'csv', header => true)
),
allrows AS (
  SELECT row_id, 'train' AS split, f1, f2, f3, f4 FROM train
  UNION ALL
  SELECT row_id, 'serve' AS split, f1, f2, f3, f4 FROM serve
),
stats AS (
  SELECT avg(f1) m1, stddev_pop(f1) s1, avg(f2) m2, stddev_pop(f2) s2,
         avg(f3) m3, stddev_pop(f3) s3, avg(f4) m4, stddev_pop(f4) s4
  FROM train
)
SELECT a.row_id, a.split,
  round((a.f1 - s.m1)/s.s1, 6) AS f1,
  round((a.f2 - s.m2)/s.s2, 6) AS f2,
  round((a.f3 - s.m3)/s.s3, 6) AS f3,
  round((a.f4 - s.m4)/s.s4, 6) AS f4
FROM allrows a CROSS JOIN stats s
""".format(tbl=tbl, base=base)

run(sql_create)
print("table created")
run("ALTER TABLE {} ALTER COLUMN row_id SET NOT NULL".format(tbl))
run("ALTER TABLE {} ADD CONSTRAINT pk_scaledd437a3 PRIMARY KEY (row_id)".format(tbl))
print("PK added")
print("counts:", run("SELECT split, count(*) FROM {} GROUP BY split ORDER BY split".format(tbl)))
print("sample:", run("SELECT * FROM {} ORDER BY row_id LIMIT 3".format(tbl)))
