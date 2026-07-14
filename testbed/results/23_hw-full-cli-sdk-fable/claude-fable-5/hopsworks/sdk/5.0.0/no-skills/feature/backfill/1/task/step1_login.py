import os

# route platform traffic through the localhost proxy (sandbox only allows localhost)
for _v in ("NO_PROXY", "no_proxy"):
    os.environ.pop(_v, None)

import hopsworks

project = hopsworks.login()
print("project:", project.name)
fs = project.get_feature_store()
print("feature store:", fs.name)
