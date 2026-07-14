import os

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login(hostname_verification=False)
ds = project.get_dataset_api()
print(ds.list("Resources/leakage_task"))
# retry the csv upload and verify immediately
p = ds.upload("data/training_data.csv", "Resources/leakage_task", overwrite=True)
print("uploaded:", p)
print("exists now:", ds.exists("Resources/leakage_task/training_data.csv"))
