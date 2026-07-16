import os
import shutil
import time

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login(hostname_verification=False)
ds = project.get_dataset_api()

shutil.copy("data/training_data.csv", ".tmp/training_data.txt")

p1 = ds.upload("data/training_data.csv", "Resources/leakage_task", overwrite=True)
p2 = ds.upload(".tmp/training_data.txt", "Resources/leakage_task", overwrite=True)
print("uploaded:", p1, p2)

back = ds.download("Resources/leakage_task/training_data.csv", ".tmp/back.csv", overwrite=True)
print("immediate readback:", back, os.path.getsize(back))

time.sleep(30)
print("after 30s: csv exists:", ds.exists("Resources/leakage_task/training_data.csv"))
print("after 30s: txt exists:", ds.exists("Resources/leakage_task/training_data.txt"))
time.sleep(60)
print("after 90s: csv exists:", ds.exists("Resources/leakage_task/training_data.csv"))
print("after 90s: txt exists:", ds.exists("Resources/leakage_task/training_data.txt"))
