import os
import urllib3

urllib3.disable_warnings()
import hopsworks

proj = hopsworks.login()
ds = proj.get_dataset_api()

for p in ["Resources/submission/answers.json", "Resources/skew_task/answers.json"]:
    print(p, "exists:", ds.exists(p))

os.makedirs("submission", exist_ok=True)
local = ds.download("Resources/submission/answers.json", "submission", overwrite=True)
print("downloaded:", local)
print("platform answers.json:", open(local).read())
