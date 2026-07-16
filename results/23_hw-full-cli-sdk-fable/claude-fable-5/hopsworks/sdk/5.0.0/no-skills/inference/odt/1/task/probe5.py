import hopsworks, inspect
project = hopsworks.login()
ds_api = project.get_dataset_api()
job_api = project.get_job_api()
print("upload:", inspect.signature(ds_api.upload))
print("job cfg:", inspect.signature(job_api.get_configuration))
print("create_job:", inspect.signature(job_api.create_job))
cfg = job_api.get_configuration("PYTHON")
print("PYTHON cfg:", cfg)
