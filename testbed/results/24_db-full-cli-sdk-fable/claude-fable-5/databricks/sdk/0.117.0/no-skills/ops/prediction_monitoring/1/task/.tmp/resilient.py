import os, time, sys
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]
catalog, sch = schema.split(".")
csv = f"/Volumes/{catalog}/{sch}/prediction_logs/prediction_log.csv"
WHS = ["8a93fc195da2ceb1", "a832b544eb7dc3fe"]

def run_once(sql, wh, budget=1200):
    r = w.statement_execution.execute_statement(
        statement=sql, warehouse_id=wh, wait_timeout="30s")
    t0 = time.time()
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        if time.time() - t0 > budget:
            try:
                w.statement_execution.cancel_execution(r.statement_id)
            except Exception:
                pass
            raise RuntimeError("timeout")
        time.sleep(15)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"{r.status.state}: {r.status.error}")
    return r.result.data_array if r.result and r.result.data_array else []

def run(sql, total_budget=3000):
    t0 = time.time()
    i = 0
    while True:
        wh = WHS[i % len(WHS)]
        i += 1
        try:
            w.warehouses.start(wh)
        except Exception:
            pass
        try:
            return run_once(sql, wh, budget=min(1200, total_budget))
        except Exception as e:
            print(f"attempt on {wh} failed: {str(e)[:150]}", flush=True)
            if time.time() - t0 > total_budget:
                raise
            time.sleep(30)

sql = f"""
WITH raw AS (
  SELECT to_timestamp(ts) AS ts, CAST(prediction AS DOUBLE) AS p
  FROM read_files('{csv}', format => 'csv', header => true)
),
daily AS (
  SELECT DATE(ts) AS day, COUNT(*) AS n, SUM(p) AS s,
         AVG(p) AS m, STDDEV(p) AS sd
  FROM raw GROUP BY DATE(ts)
),
cum AS (
  SELECT day, n, m, sd,
         SUM(s) OVER (ORDER BY day) AS cs,
         SUM(n) OVER (ORDER BY day) AS cn,
         SUM(s) OVER () AS tots,
         SUM(n) OVER () AS totn
  FROM daily
),
scored AS (
  SELECT day, n, m, sd,
         CASE WHEN cn < totn THEN
           ABS((tots - cs) / (totn - cn) - cs / cn) * SQRT(cn * (totn - cn) / totn)
         END AS score,
         LEAD(day) OVER (ORDER BY day) AS next_day
  FROM cum
)
SELECT next_day AS onset, ROUND(score, 5) AS score
FROM scored WHERE score IS NOT NULL
ORDER BY score DESC LIMIT 5
"""
print("submitting change-point query...", flush=True)
rows = run(sql)
print("TOP ONSET CANDIDATES:")
for row in rows:
    print(row)

sql2 = f"""
SELECT DATE(to_timestamp(ts)) AS day, COUNT(*) AS n,
       ROUND(AVG(CAST(prediction AS DOUBLE)), 4) AS mean_p,
       ROUND(STDDEV(CAST(prediction AS DOUBLE)), 4) AS std_p
FROM read_files('{csv}', format => 'csv', header => true)
GROUP BY 1 ORDER BY 1
"""
print("DAILY PROFILE:")
for row in run(sql2, total_budget=1500):
    print(row)
