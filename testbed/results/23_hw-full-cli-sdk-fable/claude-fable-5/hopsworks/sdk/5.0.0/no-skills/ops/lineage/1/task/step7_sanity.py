import os

import hopsworks

project = hopsworks.login()
dataset_api = project.get_dataset_api()

names = dataset_api.list("Resources/derived8af783_out")
print("export files:", names)
os.makedirs(".tmp/derived_out", exist_ok=True)
for p in names:
    if p.endswith(".csv"):
        local = dataset_api.download(p, ".tmp/derived_out", overwrite=True)
        print("downloaded:", local)
