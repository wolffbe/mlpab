import hopsworks
import os

with open(".tmp/valid_events.csv") as f:
    csv_text = f.read()

platform_script = '''import io
import hopsworks
import pandas as pd

CSV_DATA = """%s"""

project = hopsworks.login()
df = pd.read_csv(io.StringIO(CSV_DATA))
df["row_id"] = df["row_id"].astype(str)
df["account_id"] = df["account_id"].astype(str)
df["event_time"] = df["event_time"].astype("int64")
df["amount"] = df["amount"].astype("float64")
df["category"] = df["category"].astype(str)
print("rows to insert:", len(df))
print(df.dtypes)

fs = project.get_feature_store()
fg = fs.get_or_create_feature_group(
    name="eventsee881b",
    version=1,
    description="Contract-valid events export",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
)
fg.insert(df, wait=True)
print("INSERT_COMPLETE fg id:", fg.id)
''' % csv_text

with open(".tmp/platform_ingest2.py", "w") as f:
    f.write(platform_script)

project = hopsworks.login()
ds = project.get_dataset_api()
path = ds.upload(".tmp/platform_ingest2.py", "Resources", overwrite=True)
print("uploaded:", path, "exists:", ds.exists("Resources/platform_ingest2.py"))

jobs = project.get_job_api()
cfg = jobs.get_configuration("PYTHON")
cfg["appPath"] = f"/Projects/{project.name}/Resources/platform_ingest2.py"
cfg["environmentName"] = "python-feature-pipeline"
cfg["resourceConfig"]["memory"] = 4096
job = jobs.create_job("ingest2_eventsee881b", cfg)

execution = job.run(await_termination=True)
print("execution state:", execution.state, "final status:", execution.final_status)
out, err = execution.download_logs()
for p in (out, err):
    if p and os.path.exists(p):
        print(f"===== {p} =====")
        print(open(p).read()[-8000:])
if execution.final_status.lower() != "succeeded":
    raise SystemExit("platform job did not succeed")
