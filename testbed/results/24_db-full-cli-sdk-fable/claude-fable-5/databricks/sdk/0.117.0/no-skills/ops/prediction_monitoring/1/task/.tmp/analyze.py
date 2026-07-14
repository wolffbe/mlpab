import os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]
catalog, sch = schema.split(".")

# pick a warehouse
wh = None
for x in w.warehouses.list():
    if x.name == "mlpab-grader":
        wh = x
        break
    wh = wh or x
print("using warehouse:", wh.name, wh.id)

def run(sql):
    r = w.statement_execution.execute_statement(
        statement=sql, warehouse_id=wh.id, wait_timeout="50s")
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(3)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"{r.status.state}: {r.status.error}")
    cols = [c.name for c in r.manifest.schema.columns] if r.manifest else []
    rows = r.result.data_array if r.result and r.result.data_array else []
    return cols, rows

# 1. load CSV into a Delta table
run(f"""
CREATE OR REPLACE TABLE {schema}.prediction_log AS
SELECT to_timestamp(ts) AS ts, CAST(prediction AS DOUBLE) AS prediction
FROM read_files(
  '/Volumes/{catalog}/{sch}/prediction_logs/prediction_log.csv',
  format => 'csv', header => true)
""")
c, r = run(f"SELECT COUNT(*), MIN(ts), MAX(ts) FROM {schema}.prediction_log")
print("table loaded:", r)

# 2. daily monitoring statistics table
run(f"""
CREATE OR REPLACE TABLE {schema}.prediction_daily_stats AS
SELECT DATE(ts) AS day,
       COUNT(*) AS n,
       AVG(prediction) AS mean_pred,
       STDDEV(prediction) AS std_pred,
       MIN(prediction) AS min_pred,
       MAX(prediction) AS max_pred,
       PERCENTILE(prediction, 0.5) AS median_pred
FROM {schema}.prediction_log
GROUP BY DATE(ts)
""")

cols, rows = run(f"SELECT day, n, ROUND(mean_pred,4), ROUND(std_pred,4), ROUND(median_pred,4) FROM {schema}.prediction_daily_stats ORDER BY day")
for row in rows:
    print(row)

# 3. change-point detection in SQL: for each candidate day d, split the log
# into before/after and score the mean separation (two-sample style).
cols, rows = run(f"""
WITH d AS (SELECT DATE(ts) AS day, prediction FROM {schema}.prediction_log),
days AS (SELECT DISTINCT day FROM d),
scores AS (
  SELECT c.day AS onset,
         ABS(
           (SELECT AVG(prediction) FROM d WHERE d.day >= c.day) -
           (SELECT AVG(prediction) FROM d WHERE d.day < c.day)
         ) AS mean_shift
  FROM days c
  WHERE c.day > (SELECT MIN(day) FROM d)
)
SELECT onset, ROUND(mean_shift,5) FROM scores ORDER BY mean_shift DESC LIMIT 5
""")
print("top change-point candidates (max mean separation):")
for row in rows:
    print(row)
