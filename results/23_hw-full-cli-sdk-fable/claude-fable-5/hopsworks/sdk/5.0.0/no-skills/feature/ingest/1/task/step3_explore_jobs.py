import hopsworks

project = hopsworks.login()
jobs_api = project.get_job_api()
print(type(jobs_api))
print([m for m in dir(jobs_api) if not m.startswith("_")])
cfg_py = jobs_api.get_configuration("PYTHON")
print("PYTHON config:", cfg_py)
cfg_spark = jobs_api.get_configuration("PYSPARK")
print("PYSPARK config:", cfg_spark)
ds = project.get_dataset_api()
print([m for m in dir(ds) if not m.startswith("_")])
env_api = project.get_environment_api()
print("envs:", [e.name for e in env_api.get_environments()])
