import json
import os

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login(hostname_verification=False)
ds = project.get_dataset_api()
jobs_api = project.get_job_api()

p = ds.upload(".tmp/leakage_job_v2.py", "Resources/leakage_task", overwrite=True)
print("uploaded:", p)
if not ds.exists("Resources/leakage_task/training_data.txt"):
    import shutil

    shutil.copy("data/training_data.csv", ".tmp/training_data.txt")
    ds.upload(".tmp/training_data.txt", "Resources/leakage_task", overwrite=True)
    print("re-uploaded training_data.txt")

config = jobs_api.get_configuration("PYTHON")
config["appPath"] = f"/Projects/{project.name}/Resources/leakage_task/leakage_job_v2.py"
job = jobs_api.create_job("leakage_detection_v2", config)

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
