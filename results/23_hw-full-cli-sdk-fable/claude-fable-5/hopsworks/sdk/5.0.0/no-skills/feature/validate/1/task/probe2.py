import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
jobs = project.get_job_api()
print("dataset_api:", [m for m in dir(ds) if not m.startswith("_")])
print("job_api:", [m for m in dir(jobs) if not m.startswith("_")])
cfg = jobs.get_configuration("PYTHON")
print("python job config:", cfg)
envs = project.get_environment_api().get_environments()
print("environments:", [e.name for e in envs])
