import os
os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)
import hopsworks

project = hopsworks.login()
jobs = project.get_job_api()
job = jobs.get_job("scaledaff2b3_ingest")
ex = jobs.last_execution(job)
if isinstance(ex, list):
    ex = ex[0]
print("state:", ex.state, ex.final_status)
out_path, err_path = ex.download_logs()
for p in (out_path, err_path):
    print("=====", p, "=====")
    with open(p) as fh:
        print(fh.read()[-8000:])
