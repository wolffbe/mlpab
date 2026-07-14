import time

import hopsworks

proj = hopsworks.login()
job_api = proj.get_job_api()
job = job_api.get_job("prediction_monitoring")

for _ in range(60):
    execs = job.get_executions()
    ex = sorted(execs, key=lambda e: e.id)[-1]
    print(ex.id, ex.state, ex.final_status, flush=True)
    if ex.state not in ("RUNNING", "INITIALIZING", "PENDING", "ACCEPTED", "AGGREGATING_LOGS"):
        break
    time.sleep(20)

out, err = ex.download_logs()
for name in (out, err):
    if not name:
        continue
    print("=" * 20, name)
    with open(name) as f:
        print(f.read()[-7000:])
