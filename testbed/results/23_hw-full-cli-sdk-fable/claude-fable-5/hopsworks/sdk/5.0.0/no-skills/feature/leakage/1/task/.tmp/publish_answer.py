import json
import os

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login(hostname_verification=False)
ds = project.get_dataset_api()

if not ds.exists("submission"):
    try:
        made = ds.mkdir("submission")
        print("created dataset:", made)
    except Exception as e:
        print("mkdir failed:", e)

up = ds.upload("submission/answers.json", "submission", overwrite=True)
print("uploaded:", up)
print("exists on platform:", ds.exists("submission/answers.json"))

back = ds.download("submission/answers.json", ".tmp/answers_readback.json", overwrite=True)
with open(back) as fh:
    print("readback:", json.load(fh)["feature"])
