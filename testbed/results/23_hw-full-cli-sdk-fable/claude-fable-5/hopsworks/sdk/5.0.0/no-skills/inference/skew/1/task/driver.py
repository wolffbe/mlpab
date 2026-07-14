"""Client driver: uploads inputs + job script, runs the platform job, reads back the answer."""
import json
import os

import urllib3

urllib3.disable_warnings()

import hopsworks

proj = hopsworks.login()
ds = proj.get_dataset_api()
print("project:", proj.name)

try:
    ds.mkdir("Resources/skew_task")
except Exception as e:
    print("mkdir note:", e)

for f in ["data/training_sample.csv", "data/serving_log.csv", "skew_job.py"]:
    p = ds.upload(f, "Resources/skew_task", overwrite=True)
    print("uploaded:", p)

job_api = proj.get_job_api()
cfg = job_api.get_configuration("PYTHON")
cfg["appPath"] = "/Projects/{}/Resources/skew_task/skew_job.py".format(proj.name)
job = job_api.create_job("skew_detect", cfg)
print("job created:", job.name)

execution = job.run(await_termination=True)
print("execution state:", execution.state, "final:", execution.final_status)

try:
    execution.download_logs()
    for suffix in ["out", "err"]:
        fname = "skew_detect-{}.{}".format(execution.id, suffix)
        if os.path.exists(fname):
            print("--- log", suffix, "---")
            print(open(fname).read()[-3000:])
except Exception as e:
    print("log fetch note:", e)

os.makedirs("submission", exist_ok=True)
local = ds.download("Resources/submission/answers.json", "submission", overwrite=True)
print("downloaded:", local)
content = open(local).read()
print("platform answers.json:", content)
dst = os.path.join("submission", "answers.json")
if os.path.abspath(local) != os.path.abspath(dst):
    with open(dst, "w") as fh:
        fh.write(content)
