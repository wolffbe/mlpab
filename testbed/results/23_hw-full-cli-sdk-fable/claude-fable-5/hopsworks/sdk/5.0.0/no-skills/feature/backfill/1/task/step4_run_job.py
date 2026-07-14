import os

for _v in ("NO_PROXY", "no_proxy"):
    os.environ.pop(_v, None)

import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
jobs = project.get_jobs_api()

# upload batch CSVs
if not ds.exists("Resources/accounts_batches"):
    ds.mkdir("Resources/accounts_batches")
for i in (1, 2, 3):
    p = ds.upload(f"data/batch_{i}.csv", "Resources/accounts_batches", overwrite=True)
    print("uploaded:", p)

# upload job script
app_path = ds.upload("ingest_accounts_job.py", "Resources", overwrite=True)
print("uploaded job script:", app_path)

cfg = jobs.get_configuration("PYSPARK")
cfg["appPath"] = app_path

job = jobs.create_job("accountsd00439_ingest", cfg)
print("job created:", job.name)

execution = job.run(await_termination=True)
print("final state:", execution.state, "| success:", execution.success)
