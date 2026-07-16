import os

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login(hostname_verification=False)
jobs_api = project.get_job_api()
job = jobs_api.get_job("leakage_detection")
ex = max(job.get_executions(), key=lambda e: e.id)
print("execution", ex.id, ex.state, ex.final_status)
out, err = ex.download_logs()
with open(out) as fh:
    print("STDOUT:\n", fh.read())
with open(err) as fh:
    print("STDERR:\n", fh.read()[-5000:])
