import hopsworks
import os

project = hopsworks.login()
jobs = project.get_job_api()
job = jobs.get_job("ingest_eventsee881b")
ex = job.get_executions()[0]
print("execution:", ex.id, ex.state, ex.final_status)
out, err = ex.download_logs()
for p in (out, err):
    if p and os.path.exists(p):
        print(f"===== {p} =====")
        print(open(p).read()[-10000:])
