import hopsworks
import os

project = hopsworks.login()
ds = project.get_dataset_api()
print("list Resources:")
try:
    res = ds.list("Resources")
    items = res.get("items", []) if isinstance(res, dict) else res
    for it in items:
        try:
            print(" -", it["attributes"]["path"])
        except Exception:
            print(" -", it)
except Exception as e:
    print("list failed:", e)

for p in [
    "Resources/valid_events.csv",
    "/Projects/mlpabd73e67/Resources/valid_events.csv",
    "Resources/platform_ingest.py",
    "/Projects/mlpabd73e67/Resources/platform_ingest.py",
]:
    try:
        print(p, "exists:", ds.exists(p))
    except Exception as e:
        print(p, "exists check failed:", e)

jobs = project.get_job_api()
job = jobs.get_job("ingest_eventsee881b")
ex = job.get_executions()[0]
print("execution:", ex.id, ex.state, ex.final_status)
out, err = ex.download_logs()
if err and os.path.exists(err):
    print(open(err).read()[-6000:])
