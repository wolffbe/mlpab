import os

os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)

import hopsworks

project = hopsworks.login()
jobs_api = project.get_jobs_api()
job = jobs_api.get_job("incrementaljob76da9e")
execution = job.get_executions()[0]
print("execution:", execution.id, execution.final_status)
out, err = execution.download_logs()
print("=== STDOUT ===")
print(open(out).read()[-4000:])
print("=== STDERR ===")
print(open(err).read()[-4000:])
