import os

# The sandbox only allows network via the localhost proxy; NO_PROXY would
# bypass it for the 10.x Hopsworks host, so drop the bypass rules.
for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login()
print("project:", project.name)
fs = project.get_feature_store()
print("fs:", fs.name)
