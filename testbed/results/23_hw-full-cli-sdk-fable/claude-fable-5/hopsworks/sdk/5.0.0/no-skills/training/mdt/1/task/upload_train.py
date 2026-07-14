import os
os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)
import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
p = ds.upload("data/features_train.csv", "Resources/scaledaff2b3", overwrite=True)
print("uploaded", p)
print("exists:", ds.exists("Resources/scaledaff2b3/features_train.csv"))
