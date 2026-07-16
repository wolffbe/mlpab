import os

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login()
jobs_api = project.get_job_api()
job = jobs_api.get_job("recs48963e_verify")
ex = job.get_executions()[0]
print("state:", ex.state, ex.final_status)
out, err = ex.download_logs()
print("---- stdout ----")
print(open(out).read()[-3000:])
print("---- stderr ----")
print(open(err).read()[-5000:])
