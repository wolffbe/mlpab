import os

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login(hostname_verification=False)
jobs_api = project.get_job_api()
job = jobs_api.get_job("leak_analysis")
execs = job.get_executions()
ex = max(execs, key=lambda e: e.id)
print("execution", ex.id, ex.state, ex.final_status)
out, err = ex.download_logs()
for path in (out, err):
    print("=" * 30, path)
    try:
        with open(path) as fh:
            print(fh.read()[-8000:])
    except Exception as e:
        print("could not read:", e)
