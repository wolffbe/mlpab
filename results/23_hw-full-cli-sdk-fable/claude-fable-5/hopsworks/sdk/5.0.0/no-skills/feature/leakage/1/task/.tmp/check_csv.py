import os

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login(hostname_verification=False)
ds = project.get_dataset_api()
print("exists:", ds.exists("Resources/leakage_task/training_data.csv"))
print(ds.list("Resources/leakage_task"))
