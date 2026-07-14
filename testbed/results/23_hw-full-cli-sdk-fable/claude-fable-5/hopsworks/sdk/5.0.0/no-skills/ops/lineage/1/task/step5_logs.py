import os

import hopsworks

project = hopsworks.login()
job_api = project.get_job_api()
job = job_api.get_job("etl8af783")
execs = job.get_executions()
ex = sorted(execs, key=lambda e: e.id)[-1]
print("execution:", ex.id, ex.state, ex.final_status)
os.makedirs(".tmp", exist_ok=True)
out, err = ex.download_logs(path=".tmp")
print("stdout log:", out)
print("stderr log:", err)
