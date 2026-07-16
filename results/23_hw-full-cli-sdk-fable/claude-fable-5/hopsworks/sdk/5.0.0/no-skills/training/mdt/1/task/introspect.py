import os
os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)
import inspect
import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
print("DATASET API:", [m for m in dir(ds) if not m.startswith("_")])
print("upload sig:", inspect.signature(ds.upload))
jobs = project.get_job_api()
print("JOB API:", [m for m in dir(jobs) if not m.startswith("_")])
print("create_job sig:", inspect.signature(jobs.create_job))
cfg = jobs.get_configuration("PYTHON")
print("PYTHON job config:", cfg)
