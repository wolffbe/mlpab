import hopsworks
import inspect

project = hopsworks.login()
job_api = project.get_job_api()
ds_api = project.get_dataset_api()

print("JOB API:", [a for a in dir(job_api) if not a.startswith("_")])
print()
print(inspect.signature(job_api.create_job))
print(inspect.signature(job_api.get_configuration))
print()
print("DATASET API:", [a for a in dir(ds_api) if not a.startswith("_")])
print(inspect.signature(ds_api.upload))
print()
cfg = job_api.get_configuration("PYTHON")
print("PYTHON cfg:", cfg)
env_api = project.get_environment_api()
print("ENVS:", [e.name for e in env_api.get_environments()])
