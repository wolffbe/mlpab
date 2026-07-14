import os, time, datetime
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
from databricks.sdk.service.jobs import SubmitTask, NotebookTask

w = WorkspaceClient()
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]
me = w.current_user.me().user_name
nb_path = f"/Users/{me}/{prefix}/prediction_shift_analysis"

# Attempt 1: serverless job
try:
    waiter = w.jobs.submit(
        run_name=f"{prefix}_prediction_shift_analysis",
        tasks=[SubmitTask(task_key="analyze",
                          notebook_task=NotebookTask(notebook_path=nb_path))])
    print("job submitted, run_id:", waiter.run_id, flush=True)
    run = waiter.result(timeout=datetime.timedelta(minutes=20))
    print("run state:", run.state)
    out = w.jobs.get_run_output(run.tasks[0].run_id)
    print("NOTEBOOK_OUTPUT:")
    print(out.notebook_output.result if out.notebook_output else out.error)
    raise SystemExit(0)
except SystemExit:
    raise
except Exception as e:
    print("job attempt failed:", str(e)[:300], flush=True)

# Attempt 2: queued SQL statement, 20 min patience
schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]
catalog, sch = schema.split(".")
csv = f"/Volumes/{catalog}/{sch}/prediction_logs/prediction_log.csv"
sql = f"""
WITH raw AS (
  SELECT to_timestamp(ts) AS ts, CAST(prediction AS DOUBLE) AS p
  FROM read_files('{csv}', format => 'csv', header => true)
),
daily AS (
  SELECT DATE(ts) AS day, COUNT(*) AS n, SUM(p) AS s FROM raw GROUP BY DATE(ts)
),
cum AS (
  SELECT day, SUM(s) OVER (ORDER BY day) AS cs, SUM(n) OVER (ORDER BY day) AS cn,
         SUM(s) OVER () AS tots, SUM(n) OVER () AS totn
  FROM daily
),
scored AS (
  SELECT CASE WHEN cn < totn THEN
           ABS((tots - cs) / (totn - cn) - cs / cn) * SQRT(cn * (totn - cn) / totn)
         END AS score,
         LEAD(day) OVER (ORDER BY day) AS next_day
  FROM cum
)
SELECT next_day AS onset, ROUND(score, 5) AS score
FROM scored WHERE score IS NOT NULL ORDER BY score DESC LIMIT 5
"""
for wh in ["8a93fc195da2ceb1", "a832b544eb7dc3fe"]:
    try:
        w.warehouses.start(wh)
    except Exception:
        pass
    try:
        r = w.statement_execution.execute_statement(
            statement=sql, warehouse_id=wh, wait_timeout="30s")
        t0 = time.time()
        while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
            if time.time() - t0 > 1100:
                raise RuntimeError("timeout")
            time.sleep(20)
            r = w.statement_execution.get_statement(r.statement_id)
        if r.status.state != StatementState.SUCCEEDED:
            raise RuntimeError(f"{r.status.state}: {r.status.error}")
        print("SQL_RESULT:")
        for row in r.result.data_array:
            print(row)
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as e:
        print(f"sql attempt on {wh} failed:", str(e)[:200], flush=True)
print("ALL_PLATFORM_ATTEMPTS_FAILED")
