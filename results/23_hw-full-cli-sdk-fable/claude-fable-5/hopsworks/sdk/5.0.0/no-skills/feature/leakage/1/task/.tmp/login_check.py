import os

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login(hostname_verification=False)
print("project:", project.name)
fs = project.get_feature_store()
print("feature store:", fs.name)
