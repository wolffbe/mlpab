import os

os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)

import hopsworks

print("hopsworks version:", hopsworks.__version__)
project = hopsworks.login()
print("project:", project.name)
fs = project.get_feature_store()
print("feature store:", fs.name)
