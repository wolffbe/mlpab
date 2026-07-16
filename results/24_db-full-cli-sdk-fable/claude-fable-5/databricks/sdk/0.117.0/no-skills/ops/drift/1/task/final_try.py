import json
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]
cat, sch = schema.split(".")
vol = f"/Volumes/{cat}/{sch}/drift_vol"

STMT = f"""
SELECT to_date(event_time) AS d,
       avg(f1) f1, avg(f2) f2, avg(f3) f3, avg(f4) f4, avg(f5) f5, avg(f6) f6
FROM read_files('{vol}/features.csv', format => 'csv', header => true)
GROUP BY 1 ORDER BY 1
"""

res = None
for attempt in range(1):
    for wh in ["a832b544eb7dc3fe", "8a93fc195da2ceb1"]:
        try:
            r = w.statement_execution.execute_statement(
                statement=STMT, warehouse_id=wh, wait_timeout="50s"
            )
            t0 = time.time()
            while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
                if time.time() - t0 > 120:
                    raise TimeoutError("statement stuck")
                time.sleep(5)
                r = w.statement_execution.get_statement(r.statement_id)
            if r.status.state == StatementState.SUCCEEDED:
                res = r
                break
            print(wh, r.status.state, r.status.error and r.status.error.message, flush=True)
        except Exception as e:
            print(wh, "err:", e, flush=True)
    if res:
        break
    time.sleep(15)

if not res:
    raise SystemExit("NO_COMPUTE")

rows = res.result.data_array
dates = [r[0] for r in rows]
feats = ["f1", "f2", "f3", "f4", "f5", "f6"]
best = None
for j, f in enumerate(feats, start=1):
    means = [float(r[j]) for r in rows]
    base = means[:14]
    bmean = sum(base) / len(base)
    bstd = (sum((x - bmean) ** 2 for x in base) / (len(base) - 1)) ** 0.5
    for i in range(14, len(means)):
        z = abs(means[i] - bmean) / (bstd + 1e-12)
        if z > 5 and all(abs(m - bmean) / (bstd + 1e-12) > 3 for m in means[i:i + 5]):
            score = sum(abs(m - bmean) / (bstd + 1e-12) for m in means[i:]) / (len(means) - i)
            if best is None or score > best[2]:
                best = (f, dates[i], score)
            break

print("daily means per feature (first/last 5):")
for j, f in enumerate(feats, start=1):
    ms = [round(float(r[j]), 2) for r in rows]
    print(f, ms[:5], "...", ms[-5:])

ans = {"feature": best[0], "onset": best[1]} if best else {"feature": None, "onset": None}
print("ANSWER:", ans, "score:", best and best[2])
os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as fh:
    json.dump(ans, fh)
import io
w.files.upload(f"{vol}/answers.json", io.BytesIO(json.dumps(ans).encode()), overwrite=True)
print("written")
