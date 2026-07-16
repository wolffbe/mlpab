import os

os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)

import inspect

import hopsworks

project = hopsworks.login()

jobs_api = project.get_jobs_api()
ds_api = project.get_dataset_api()

print("=== JobsApi methods ===")
print([m for m in dir(jobs_api) if not m.startswith("_")])
print(inspect.signature(jobs_api.create_job))
print(inspect.signature(jobs_api.get_configuration))

print("=== DatasetApi methods ===")
print([m for m in dir(ds_api) if not m.startswith("_")])
print(inspect.signature(ds_api.upload))

from hopsworks_common.job import Job

print("=== Job methods ===")
print([m for m in dir(Job) if not m.startswith("_")])
print(inspect.signature(Job.schedule))

print("=== PYTHON job config ===")
print(jobs_api.get_configuration("PYTHON"))
