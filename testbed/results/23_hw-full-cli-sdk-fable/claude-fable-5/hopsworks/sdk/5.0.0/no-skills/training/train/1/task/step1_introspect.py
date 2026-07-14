import hopsworks

project = hopsworks.login()
print("project:", project.name)
js = project.get_job_api()
ds = project.get_dataset_api()
print("JOB API:", [m for m in dir(js) if not m.startswith("_")])
print("DS API:", [m for m in dir(ds) if not m.startswith("_")])
