import hw_env  # noqa: F401
import hopsworks

proj = hopsworks.login()
ds = proj.get_dataset_api()
print("dataset_api:", [m for m in dir(ds) if not m.startswith("_")])
ja = proj.get_job_api()
print("job_api:", [m for m in dir(ja) if not m.startswith("_")])
import json
print(json.dumps(ja.get_configuration("PYTHON"), indent=1))
