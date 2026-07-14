import os

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login(hostname_verification=False)
ds = project.get_dataset_api()

p1 = ds.upload("data/training_data.csv", "Resources", overwrite=True)
print("uploaded:", p1)
p2 = ds.upload(".tmp/leak_analysis_job.py", "Resources", overwrite=True)
print("uploaded:", p2)
