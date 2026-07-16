import io
import os

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # e.g. workspace.mlpab68f24d
CATALOG, SCHEMA_NAME = SCHEMA.split(".")
WAREHOUSE_ID = "a832b544eb7dc3fe"
VOLUME = "raw"
VOL_ROOT = f"/Volumes/{CATALOG}/{SCHEMA_NAME}/{VOLUME}"
TABLE = f"{SCHEMA}.churntraining9b67c4"

w = WorkspaceClient()


def sql(stmt):
    r = w.statement_execution.execute_statement(
        statement=stmt, warehouse_id=WAREHOUSE_ID, wait_timeout="50s"
    )
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed: {r.status}\n{stmt[:300]}")
    return r


sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA_NAME}.{VOLUME}")

files = [
    "transactions.csv",
    "transactions_late.csv",
    "profiles.csv",
    "activity.csv",
    "account_health.csv",
    "labels.csv",
]
for f in files:
    with open(f"data/{f}", "rb") as fh:
        w.files.upload(f"{VOL_ROOT}/{f}", io.BytesIO(fh.read()), overwrite=True)
    print("uploaded", f)

sql(f"""
CREATE OR REPLACE TABLE {TABLE} (
  account_id STRING,
  label_time BIGINT,
  amount DOUBLE,
  balance DOUBLE,
  credit_score BIGINT,
  tier STRING,
  sessions_7d BIGINT,
  health_score DOUBLE,
  churned BIGINT
)
""")

insert = f"""
INSERT INTO {TABLE}
WITH labels AS (
  SELECT * FROM read_files('{VOL_ROOT}/labels.csv', format => 'csv', header => true,
    schema => 'account_id STRING, label_time BIGINT, churned BIGINT')
),
tx AS (
  SELECT * FROM read_files('{VOL_ROOT}/transactions.csv', format => 'csv', header => true,
    schema => 'account_id STRING, event_time BIGINT, amount DOUBLE, balance DOUBLE')
  UNION ALL
  SELECT * FROM read_files('{VOL_ROOT}/transactions_late.csv', format => 'csv', header => true,
    schema => 'account_id STRING, event_time BIGINT, amount DOUBLE, balance DOUBLE')
),
pr AS (
  SELECT * FROM read_files('{VOL_ROOT}/profiles.csv', format => 'csv', header => true,
    schema => 'account_id STRING, event_time BIGINT, credit_score BIGINT, tier STRING')
),
ac AS (
  SELECT * FROM read_files('{VOL_ROOT}/activity.csv', format => 'csv', header => true,
    schema => 'account_id STRING, event_time BIGINT, sessions_7d BIGINT')
),
hh AS (
  SELECT * FROM read_files('{VOL_ROOT}/account_health.csv', format => 'csv', header => true,
    schema => 'account_id STRING, event_time BIGINT, health_score DOUBLE')
),
tx1 AS (
  SELECT account_id, label_time, amount, balance FROM (
    SELECT l.account_id, l.label_time, t.amount, t.balance,
           ROW_NUMBER() OVER (PARTITION BY l.account_id, l.label_time ORDER BY t.event_time DESC) rn
    FROM labels l JOIN tx t
      ON l.account_id = t.account_id AND t.event_time <= l.label_time
  ) WHERE rn = 1
),
pr1 AS (
  SELECT account_id, label_time, credit_score, tier FROM (
    SELECT l.account_id, l.label_time, p.credit_score, p.tier,
           ROW_NUMBER() OVER (PARTITION BY l.account_id, l.label_time ORDER BY p.event_time DESC) rn
    FROM labels l JOIN pr p
      ON l.account_id = p.account_id AND p.event_time <= l.label_time
  ) WHERE rn = 1
),
ac1 AS (
  SELECT account_id, label_time, sessions_7d FROM (
    SELECT l.account_id, l.label_time, a.sessions_7d,
           ROW_NUMBER() OVER (PARTITION BY l.account_id, l.label_time ORDER BY a.event_time DESC) rn
    FROM labels l JOIN ac a
      ON l.account_id = a.account_id AND a.event_time <= l.label_time
  ) WHERE rn = 1
),
hh1 AS (
  SELECT account_id, label_time, health_score FROM (
    SELECT l.account_id, l.label_time, h.health_score,
           ROW_NUMBER() OVER (PARTITION BY l.account_id, l.label_time ORDER BY h.event_time DESC) rn
    FROM labels l JOIN hh h
      ON l.account_id = h.account_id AND h.event_time <= l.label_time
  ) WHERE rn = 1
)
SELECT l.account_id, l.label_time,
       tx1.amount, tx1.balance,
       pr1.credit_score, pr1.tier,
       ac1.sessions_7d,
       hh1.health_score,
       l.churned
FROM labels l
LEFT JOIN tx1 ON l.account_id = tx1.account_id AND l.label_time = tx1.label_time
LEFT JOIN pr1 ON l.account_id = pr1.account_id AND l.label_time = pr1.label_time
LEFT JOIN ac1 ON l.account_id = ac1.account_id AND l.label_time = ac1.label_time
LEFT JOIN hh1 ON l.account_id = hh1.account_id AND l.label_time = hh1.label_time
"""
sql(insert)
print("insert done")

r = sql(f"SELECT version, operation FROM (DESCRIBE HISTORY {TABLE}) ORDER BY version")
print("history:", r.result.data_array)
r = sql(f"SELECT COUNT(*), COUNT(amount), COUNT(credit_score), COUNT(sessions_7d), COUNT(health_score) FROM {TABLE}")
print("counts:", r.result.data_array)
r = sql(f"SELECT * FROM {TABLE} ORDER BY account_id, label_time LIMIT 5")
print("sample:")
for row in r.result.data_array:
    print(row)
