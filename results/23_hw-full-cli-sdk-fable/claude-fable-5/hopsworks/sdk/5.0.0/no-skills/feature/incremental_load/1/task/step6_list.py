import os

os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)

import json

import hopsworks

project = hopsworks.login()
ds_api = project.get_dataset_api()

res = ds_api.list("Resources/incremental76da9e")
print(type(res))
print(json.dumps(res, indent=2, default=str)[:3000])
