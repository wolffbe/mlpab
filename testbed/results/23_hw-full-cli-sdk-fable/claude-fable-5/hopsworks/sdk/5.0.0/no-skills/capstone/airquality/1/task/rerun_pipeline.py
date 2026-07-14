"""Re-upload the fixed job script and re-run the existing pipeline job."""

import time

import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
ds.upload("pipeline_job.py", "Resources/airq754fa9", overwrite=True)
print("script re-uploaded", flush=True)

job_api = project.get_job_api()
job = job_api.get_job("airq_pipeline_754fa9")

try:
    execution = job.run(await_termination=True)
    print("final status:", execution.final_status, "success:", execution.success, flush=True)
except Exception as e:  # noqa: BLE001
    print("await interrupted (will poll):", e, flush=True)
    deadline = time.time() + 2700
    execution = None
    while time.time() < deadline:
        try:
            job = job_api.get_job("airq_pipeline_754fa9")
            execution = max(job.get_executions(), key=lambda x: x.id)
            print(f"execution {execution.id}: state={execution.state} final={execution.final_status}", flush=True)
            if execution.state and execution.state.upper() in ("FINISHED", "FAILED", "KILLED"):
                break
        except Exception as e2:  # noqa: BLE001
            print("poll error (retrying):", e2, flush=True)
        time.sleep(15)

for attempt in range(5):
    try:
        paths = execution.download_logs()
        for p in paths:
            if p is None:
                continue
            print(f"===== {p} =====", flush=True)
            with open(p) as f:
                print(f.read(), flush=True)
        break
    except Exception as e:  # noqa: BLE001
        print("log download failed (retrying):", e, flush=True)
        time.sleep(20)
