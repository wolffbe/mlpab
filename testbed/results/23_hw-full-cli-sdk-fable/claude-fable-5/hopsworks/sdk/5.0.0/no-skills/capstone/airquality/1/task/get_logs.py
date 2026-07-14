import sys
import time

import hopsworks

job_name = sys.argv[1] if len(sys.argv) > 1 else "airq_verify_754fa9"

project = hopsworks.login()
job_api = project.get_job_api()
job = job_api.get_job(job_name)
execution = max(job.get_executions(), key=lambda x: x.id)
print("execution", execution.id, "state:", execution.state, "final:", execution.final_status, flush=True)

for attempt in range(8):
    try:
        paths = execution.download_logs()
        printed = False
        for p in paths:
            if p is None:
                continue
            print(f"===== {p} =====", flush=True)
            with open(p) as f:
                print(f.read(), flush=True)
            printed = True
        if printed:
            break
        print("no log paths yet, retrying", flush=True)
    except Exception as e:  # noqa: BLE001
        print("log download failed (retrying):", e, flush=True)
    time.sleep(20)
