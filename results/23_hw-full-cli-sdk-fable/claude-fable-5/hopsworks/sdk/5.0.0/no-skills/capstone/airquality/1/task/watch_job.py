"""Monitor the running pipeline job, tolerating transient 503s, then print its logs."""

import time

import hopsworks

project = hopsworks.login()
job_api = project.get_job_api()

execution = None
final = None
deadline = time.time() + 2700
while time.time() < deadline:
    try:
        job = job_api.get_job("airq_pipeline_754fa9")
        executions = job.get_executions()
        execution = max(executions, key=lambda e: e.id)
        state = execution.state
        final = execution.final_status
        print(f"execution {execution.id}: state={state} final={final}", flush=True)
        if final and final.upper() not in ("UNDEFINED", "NONE", ""):
            break
        if state and state.upper() in ("FINISHED", "FAILED", "KILLED"):
            break
    except Exception as e:  # noqa: BLE001
        print("poll error (retrying):", e, flush=True)
    time.sleep(15)

print("FINAL:", execution.id if execution else None, final, flush=True)

for attempt in range(5):
    try:
        out_path, err_path = execution.download_logs()
        for label, p in [("STDOUT", out_path), ("STDERR", err_path)]:
            print(f"===== {label} =====", flush=True)
            with open(p) as f:
                print(f.read(), flush=True)
        break
    except Exception as e:  # noqa: BLE001
        print("log download failed (retrying):", e, flush=True)
        time.sleep(20)
