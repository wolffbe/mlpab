import inspect, hopsworks
project = hopsworks.login()
print("PROJECT", project.name, project.id)
ds = project.get_dataset_api()
print("UPLOAD", inspect.signature(ds.upload))
print("DOWNLOAD", inspect.signature(ds.download))
api = project.get_job_api()
print("JOBAPI", [m for m in dir(api) if not m.startswith("_")])
print("GETCONF", inspect.signature(api.get_configuration))
try:
    cfg = api.get_configuration("PYTHON")
    print("PYCONF KEYS", list(cfg.keys()))
    print("PYCONF", cfg)
except Exception as e:
    print("conf err", repr(e))
