import os

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login(hostname_verification=False)
ds = project.get_dataset_api()
local = ds.download("Resources/leakage_task/leakage_job.py", ".tmp", overwrite=True)
print("downloaded to", local)
with open(local) as fh:
    print(fh.read())
