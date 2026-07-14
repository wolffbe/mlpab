import os

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login()
jobs_api = project.get_job_api()
job = jobs_api.get_job("recs48963e_verify")
execution = job.run(await_termination=True)
print("final state:", execution.state, execution.final_status)
out, err = execution.download_logs()
print("---- stdout ----")
print(open(out).read()[-3000:])
print("---- stderr tail ----")
print(open(err).read()[-1200:])
