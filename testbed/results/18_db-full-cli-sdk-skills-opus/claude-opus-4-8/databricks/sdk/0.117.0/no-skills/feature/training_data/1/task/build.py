import os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
WH = "4dfab06c923fe3cc"
catalog, schema = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')
base = f"/Volumes/{catalog}/{schema}/rawdata"


def sql(stmt):
    r = w.statement_execution.execute_statement(statement=stmt, warehouse_id=WH,
            catalog=catalog, schema=schema, wait_timeout="50s")
    sid = r.statement_id
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        r = w.statement_execution.get_statement(sid)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"FAILED: {r.status.state} {r.status.error}")
    return r


def rf(name, cols):
    # read_files CSV with header; cast event_time to bigint
    sel = ", ".join(cols)
    return (f"SELECT account_id, CAST(event_time AS BIGINT) AS event_time, {sel} "
            f"FROM read_files('{base}/{name}.csv', format => 'csv', header => true)")


tx = rf("transactions", ["CAST(amount AS DOUBLE) AS amount", "CAST(balance AS DOUBLE) AS balance"])
tx_late = rf("transactions_late", ["CAST(amount AS DOUBLE) AS amount", "CAST(balance AS DOUBLE) AS balance"])
profiles = rf("profiles", ["CAST(credit_score AS INT) AS credit_score", "CAST(tier AS STRING) AS tier"])
activity = rf("activity", ["CAST(sessions_7d AS INT) AS sessions_7d"])
health = rf("account_health", ["CAST(health_score AS DOUBLE) AS health_score"])

q = f"""
CREATE OR REPLACE TABLE {catalog}.{schema}.churntraining605fb7 AS
WITH labels AS (
  SELECT account_id, CAST(label_time AS BIGINT) AS label_time, CAST(churned AS INT) AS churned
  FROM read_files('{base}/labels.csv', format => 'csv', header => true)
),
tx AS ( {tx} UNION ALL {tx_late} ),
profiles AS ( {profiles} ),
activity AS ( {activity} ),
health AS ( {health} ),
tx_asof AS (
  SELECT account_id, label_time, amount, balance FROM (
    SELECT l.account_id, l.label_time, f.amount, f.balance,
      ROW_NUMBER() OVER (PARTITION BY l.account_id, l.label_time ORDER BY f.event_time DESC) rn
    FROM labels l JOIN tx f ON f.account_id = l.account_id AND f.event_time <= l.label_time
  ) WHERE rn = 1
),
profiles_asof AS (
  SELECT account_id, label_time, credit_score, tier FROM (
    SELECT l.account_id, l.label_time, f.credit_score, f.tier,
      ROW_NUMBER() OVER (PARTITION BY l.account_id, l.label_time ORDER BY f.event_time DESC) rn
    FROM labels l JOIN profiles f ON f.account_id = l.account_id AND f.event_time <= l.label_time
  ) WHERE rn = 1
),
activity_asof AS (
  SELECT account_id, label_time, sessions_7d FROM (
    SELECT l.account_id, l.label_time, f.sessions_7d,
      ROW_NUMBER() OVER (PARTITION BY l.account_id, l.label_time ORDER BY f.event_time DESC) rn
    FROM labels l JOIN activity f ON f.account_id = l.account_id AND f.event_time <= l.label_time
  ) WHERE rn = 1
),
health_asof AS (
  SELECT account_id, label_time, health_score FROM (
    SELECT l.account_id, l.label_time, f.health_score,
      ROW_NUMBER() OVER (PARTITION BY l.account_id, l.label_time ORDER BY f.event_time DESC) rn
    FROM labels l JOIN health f ON f.account_id = l.account_id AND f.event_time <= l.label_time
  ) WHERE rn = 1
)
SELECT l.account_id, l.label_time,
       tx.amount, tx.balance,
       p.credit_score, p.tier,
       a.sessions_7d,
       h.health_score,
       l.churned
FROM labels l
LEFT JOIN tx_asof tx USING (account_id, label_time)
LEFT JOIN profiles_asof p USING (account_id, label_time)
LEFT JOIN activity_asof a USING (account_id, label_time)
LEFT JOIN health_asof h USING (account_id, label_time)
"""

sql(q)
print("table created")

# validate
r = sql(f"SELECT COUNT(*) AS n, COUNT(amount) AS namt, COUNT(credit_score) nc, COUNT(sessions_7d) ns, COUNT(health_score) nh FROM {catalog}.{schema}.churntraining605fb7")
print("counts:", r.result.data_array)
r = sql(f"SELECT * FROM {catalog}.{schema}.churntraining605fb7 ORDER BY account_id LIMIT 5")
print("cols:", [c.name for c in r.manifest.schema.columns])
for row in r.result.data_array:
    print(row)
