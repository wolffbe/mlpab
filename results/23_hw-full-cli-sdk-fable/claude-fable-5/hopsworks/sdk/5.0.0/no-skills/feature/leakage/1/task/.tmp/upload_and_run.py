import json
import os

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login(hostname_verification=False)
ds = project.get_dataset_api()
jobs_api = project.get_job_api()

# upload right before launching so the job reads it before any cleanup sweep
p = ds.upload("data/training_data.csv", "Resources/leakage_task", overwrite=True)
print("uploaded:", p, "exists:", ds.exists("Resources/leakage_task/training_data.csv"))

job = jobs_api.get_job("leakage_detection")
execution = job.run(await_termination=True)
print("final state:", execution.final_status, execution.state)

out, err = execution.download_logs()
with open(out) as fh:
    print("STDOUT:\n", fh.read())
with open(err) as fh:
    print("STDERR tail:\n", fh.read()[-2500:])

if ds.exists("submission/answers.json"):
    local = ds.download("submission/answers.json", "submission", overwrite=True)
    print("downloaded:", local)
    with open(local) as fh:
        print(json.dumps(json.load(fh), indent=2))
else:
    print("submission/answers.json not found on platform")
