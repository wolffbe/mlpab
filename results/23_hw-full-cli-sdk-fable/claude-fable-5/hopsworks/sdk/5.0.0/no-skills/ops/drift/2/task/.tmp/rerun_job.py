import time

import hw_env  # noqa: F401
import hopsworks

proj = hopsworks.login()
ds = proj.get_dataset_api()
ja = proj.get_job_api()

ds.upload(".tmp/drift_job.py", "Resources/drift_task", overwrite=True)
print("uploaded script")

job = ja.get_job("drift_investigation")
execution = job.run(await_termination=False)
print("execution id:", execution.id)

state = None
for _ in range(240):  # up to ~40 min, resilient to transient proxy errors
    time.sleep(10)
    try:
        cur = job.get_executions()
        cur = [e for e in cur if e.id == execution.id][0]
        state = cur.state
        if state not in ("INITIALIZING", "RUNNING", "ACCEPTED", "NEW", "PENDING",
                         "AGGREGATING_LOGS", "STARTING_APP_MASTER", "SUBMITTED"):
            execution = cur
            break
    except Exception as e:
        print("poll error (retrying):", str(e)[:120])
print("final state:", state)

for attempt in range(5):
    try:
        out_path, err_path = execution.download_logs()
        break
    except Exception as e:
        print("log download failed (retrying):", str(e)[:120])
        time.sleep(10)
for p in (out_path, err_path):
    print("=====", p.split("/")[-1], "=====")
    with open(p) as fh:
        t = fh.read()
    print(t[-10000:])
