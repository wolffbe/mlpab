import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
job_api = project.get_job_api()
print("DATASET_API:", sorted(m for m in dir(ds) if not m.startswith("_")))
print()
import inspect
print("upload sig:", inspect.signature(ds.upload))
print("download sig:", inspect.signature(ds.download))
print()
print("JOB_API:", sorted(m for m in dir(job_api) if not m.startswith("_")))
print("create_job sig:", inspect.signature(job_api.create_job))
print("get_configuration sig:", inspect.signature(job_api.get_configuration))
cfg = job_api.get_configuration("PYTHON")
print("PYTHON cfg:", cfg)
from hopsworks_common.job import Job
print("JOB:", sorted(m for m in dir(Job) if not m.startswith("_")))
print("run sig:", inspect.signature(Job.run))
from hopsworks_common.execution import Execution
print("EXECUTION:", sorted(m for m in dir(Execution) if not m.startswith("_")))
env_api = project.get_environment_api()
print("ENVS:", [ (e.name if hasattr(e,'name') else e) for e in env_api.get_environments()])
