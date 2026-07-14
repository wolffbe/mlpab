import os

for _v in ("NO_PROXY", "no_proxy"):
    os.environ.pop(_v, None)

import hopsworks

project = hopsworks.login()

ds = project.get_dataset_api()
print("DatasetApi:", [m for m in dir(ds) if not m.startswith("_")])

jobs = project.get_jobs_api()
print("JobsApi:", [m for m in dir(jobs) if not m.startswith("_")])

import inspect
print(inspect.signature(ds.upload))
print(inspect.signature(jobs.create_job))
print(inspect.signature(jobs.get_configuration))
cfg = jobs.get_configuration("PYSPARK")
print("PYSPARK cfg:", cfg)
