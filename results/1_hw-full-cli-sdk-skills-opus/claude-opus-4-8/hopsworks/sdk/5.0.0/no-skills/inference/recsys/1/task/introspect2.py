import hopsworks, inspect
proj = hopsworks.login()
print("PROJ methods:", [m for m in dir(proj) if not m.startswith('_')])
ja = proj.get_jobs_api()
print("JOBS api:", [m for m in dir(ja) if not m.startswith('_')])
print("create_job sig:", inspect.signature(ja.create_job))
ds = proj.get_dataset_api()
print("DATASET api:", [m for m in dir(ds) if not m.startswith('_')])
