import glob
import urllib3

urllib3.disable_warnings()
import hopsworks

proj = hopsworks.login()
job_api = proj.get_job_api()
job = job_api.get_job("skew_detect")
execs = job.get_executions()
ex = sorted(execs, key=lambda e: e.id)[-1]
print("execution", ex.id, ex.state, ex.final_status)
out, err = ex.download_logs()
print("log files:", out, err)
for f in [out, err]:
    if f:
        print("=====", f, "=====")
        print(open(f, errors="replace").read()[-6000:])
