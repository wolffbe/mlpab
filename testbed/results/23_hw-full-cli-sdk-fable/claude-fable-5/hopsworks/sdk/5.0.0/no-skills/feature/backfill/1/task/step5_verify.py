import os

for _v in ("NO_PROXY", "no_proxy"):
    os.environ.pop(_v, None)

import hopsworks

project = hopsworks.login()
jobs = project.get_jobs_api()

job = jobs.get_job("accountsd00439_ingest")
execution = job.get_executions()[0]
print("execution:", execution.id, execution.state, execution.final_status)

out_path, err_path = execution.download_logs()
with open(out_path) as f:
    print("---- stdout ----")
    print(f.read())

fs = project.get_feature_store()
fg = fs.get_feature_group("accountsd00439", 1)
print("fg:", fg.name, "v", fg.version, "| online_enabled:", fg.online_enabled,
      "| primary_key:", fg.primary_key, "| event_time:", fg.event_time)

try:
    df = fg.read()
    print("offline read rows:", len(df))
    print(df.sort_values("row_id").head(5))
    print("unique row_ids:", df["row_id"].nunique())
except Exception as e:
    print("offline read failed:", type(e).__name__, str(e)[:500])

try:
    odf = fg.read(online=True)
    print("online read rows:", len(odf))
except Exception as e:
    print("online read failed:", type(e).__name__, str(e)[:500])
