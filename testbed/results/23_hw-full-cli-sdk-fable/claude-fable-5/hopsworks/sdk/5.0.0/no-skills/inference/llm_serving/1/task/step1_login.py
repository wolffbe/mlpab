import os

# The sandbox only allows egress via the localhost proxy; NO_PROXY=10.0.0.0/8
# would bypass it for the Hopsworks host, so clear it for this process.
os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)

import hopsworks

project = hopsworks.login()
print("project:", project.name)
ms = project.get_model_serving()
mr = project.get_model_registry()
print("serving:", type(ms).__name__, "registry:", type(mr).__name__)
print("deployments:", [d.name for d in ms.get_deployments()])
