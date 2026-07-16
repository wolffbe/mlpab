import hopsworks
import os

project = hopsworks.login()
ds = project.get_dataset_api()
print("csv exists:", ds.exists("Resources/valid_events.csv"))
ds.upload("platform_ingest.py", "Resources", overwrite=True)

jobs = project.get_job_api()
job = jobs.get_job("ingest_eventsee881b")
execution = job.run(await_termination=True)
print("execution state:", execution.state, "final status:", execution.final_status)
out, err = execution.download_logs()
for p in (out, err):
    if p and os.path.exists(p):
        print(f"===== {p} =====")
        print(open(p).read()[-10000:])
if execution.final_status.lower() != "succeeded":
    raise SystemExit("platform job did not succeed")
