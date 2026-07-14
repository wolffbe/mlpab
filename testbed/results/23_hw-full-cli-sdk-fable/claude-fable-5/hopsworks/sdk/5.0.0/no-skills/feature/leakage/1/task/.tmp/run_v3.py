import json
import os
import shutil

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login(hostname_verification=False)
ds = project.get_dataset_api()
jobs_api = project.get_job_api()

p = ds.upload(".tmp/leakage_job_v3.py", "Resources/leakage_task", overwrite=True)
print("uploaded:", p)
shutil.copy("data/training_data.csv", ".tmp/training_data.txt")
ds.upload(".tmp/training_data.txt", "Resources/leakage_task", overwrite=True)
print("uploaded training_data.txt, exists:", ds.exists("Resources/leakage_task/training_data.txt"))

config = jobs_api.get_configuration("PYTHON")
config["appPath"] = f"/Projects/{project.name}/Resources/leakage_task/leakage_job_v3.py"
job = jobs_api.create_job("leakage_detection_v3", config)

execution = job.run(await_termination=True)
print("final state:", execution.final_status, execution.state)

out, err = execution.download_logs()
with open(out) as fh:
    print("STDOUT:\n", fh.read())
with open(err) as fh:
    print("STDERR tail:\n", fh.read()[-2500:])
