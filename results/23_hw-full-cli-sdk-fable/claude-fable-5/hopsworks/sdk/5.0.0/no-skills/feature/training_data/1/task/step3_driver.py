import os

# Sandbox only allows network via the localhost proxy; NO_PROXY would bypass
# it for the 10.x Hopsworks host, so drop the bypass rules.
for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login()
print("project:", project.name)

dataset_api = project.get_dataset_api()
base = f"Resources/churn30fee3"
if not dataset_api.exists(base):
    dataset_api.mkdir(base)

files = [
    "data/transactions.csv",
    "data/transactions_late.csv",
    "data/profiles.csv",
    "data/activity.csv",
    "data/account_health.csv",
    "data/labels.csv",
    "churn_td_30fee3_job.py",
]
for f in files:
    p = dataset_api.upload(f, base, overwrite=True)
    print("uploaded:", p)

jobs_api = project.get_job_api()
config = jobs_api.get_configuration("PYSPARK")
config["appPath"] = f"/Projects/{project.name}/{base}/churn_td_30fee3_job.py"
job = jobs_api.create_job("churn_td_30fee3", config)
print("job created:", job.name)

execution = job.run(await_termination=True)
print("execution state:", execution.state, "final status:", execution.final_status)

out, err = execution.download_logs()
for path in (out, err):
    if path and os.path.exists(path):
        print(f"===== {path} =====")
        with open(path) as fh:
            print(fh.read()[-8000:])
