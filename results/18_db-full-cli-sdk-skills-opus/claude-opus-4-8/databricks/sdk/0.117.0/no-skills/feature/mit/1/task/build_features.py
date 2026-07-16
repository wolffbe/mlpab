import sys, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
WH = "4dfab06c923fe3cc"


def run(sql):
    r = w.statement_execution.execute_statement(statement=sql, warehouse_id=WH, wait_timeout="50s")
    sid = r.statement_id
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        r = w.statement_execution.get_statement(sid)
    if r.status.state != StatementState.SUCCEEDED:
        print("FAILED:", r.status.state, getattr(r.status, "error", None))
        sys.exit(1)
    return r


sql = """
CREATE OR REPLACE TABLE workspace.mlpab4bb10d.features74f1ef AS
WITH tx AS (
  SELECT row_id, account_id, CAST(event_time AS BIGINT) AS event_time,
         CAST(amount AS DOUBLE) AS amount, currency
  FROM read_files('/Volumes/workspace/mlpab4bb10d/raw/transactions.csv',
       format => 'csv', header => true)
),
fx AS (
  SELECT currency, CAST(fx_rate AS DOUBLE) AS fx_rate
  FROM read_files('/Volumes/workspace/mlpab4bb10d/raw/fx_rates.csv',
       format => 'csv', header => true)
),
j AS (
  SELECT tx.row_id, tx.account_id, tx.event_time, tx.amount, tx.currency, fx.fx_rate
  FROM tx LEFT JOIN fx ON tx.currency = fx.currency
)
SELECT
  row_id,
  account_id,
  event_time,
  amount * fx_rate AS amount_usd,
  CASE WHEN pmod(CAST(floor(event_time/86400000) AS BIGINT), 7) IN (2,3) THEN 1 ELSE 0 END AS is_weekend,
  SUM(amount) OVER (
    PARTITION BY account_id ORDER BY event_time
    RANGE BETWEEN 604800000 PRECEDING AND CURRENT ROW
  ) AS amount_7d
FROM j
"""
run(sql)
print("table created")

run("ALTER TABLE workspace.mlpab4bb10d.features74f1ef ALTER COLUMN row_id SET NOT NULL")
run("ALTER TABLE workspace.mlpab4bb10d.features74f1ef ADD CONSTRAINT pk_features74f1ef PRIMARY KEY (row_id)")
print("pk added")

run("ALTER TABLE workspace.mlpab4bb10d.features74f1ef SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print("cdf enabled")

r = run("SELECT count(*) c FROM workspace.mlpab4bb10d.features74f1ef")
print("rowcount:", r.result.data_array)
r = run("SELECT * FROM workspace.mlpab4bb10d.features74f1ef ORDER BY event_time LIMIT 6")
print("cols:", [c.name for c in r.manifest.schema.columns])
for row in r.result.data_array:
    print(row)
