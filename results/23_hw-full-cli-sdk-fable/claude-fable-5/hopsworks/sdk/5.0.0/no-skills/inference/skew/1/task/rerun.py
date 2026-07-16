"""Ensure inputs exist on the platform, re-run the job, read back the answer."""
import json
import os

import urllib3

urllib3.disable_warnings()
import hopsworks

proj = hopsworks.login()
ds = proj.get_dataset_api()

for local, remote in [("data/training_sample.csv", "Resources/skew_task/training_sample.csv"),
                      ("data/serving_log.csv", "Resources/skew_task/serving_log.csv"),
                      ("skew_job.py", "Resources/skew_task/skew_job.py")]:
    for attempt in range(3):
        if ds.exists(remote):
            break
        print("uploading", local, "attempt", attempt + 1)
        try:
            ds.upload(local, "Resources/skew_task", overwrite=True)
        except Exception as e:
            print("  upload error:", e)
    print(remote, "exists:", ds.exists(remote))
    if not ds.exists(remote):
        raise SystemExit("could not upload " + remote)

job_api = proj.get_job_api()
job = job_api.get_job("skew_detect")
execution = job.run(await_termination=True)
print("execution state:", execution.state, "final:", execution.final_status)

out, err = execution.download_logs()
for f in [out, err]:
    if f and os.path.exists(f):
        print("=====", f, "=====")
        print(open(f, errors="replace").read()[-4000:])

os.makedirs("submission", exist_ok=True)
local = ds.download("Resources/submission/answers.json", "submission", overwrite=True)
print("downloaded:", local)
content = open(local).read()
print("platform answers.json:", content)
dst = os.path.join("submission", "answers.json")
if os.path.abspath(local) != os.path.abspath(dst):
    with open(dst, "w") as fh:
        fh.write(content)
