import json
import os

import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

fg_d = fs.get_feature_group("derived8af783", 1)
print("derived8af783 online_enabled:", fg_d.online_enabled)
print("columns:", [f.name for f in fg_d.features])
print("primary key:", fg_d.primary_key)

try:
    lineage = fg_d.get_parent_feature_groups()
    parents = lineage.accessible if hasattr(lineage, "accessible") else lineage
    print("parents:", [(p.name, p.version) for p in parents])
except Exception as e:
    print("lineage read failed:", type(e).__name__, str(e)[:300])

# download the derived table export for a sanity check
dataset_api = project.get_dataset_api()
base = "Resources/derived8af783_out"
os.makedirs(".tmp/derived_out", exist_ok=True)
try:
    files, _ = dataset_api.list_files(base, 0, 100)
    names = [f.attributes.path for f in files]
except Exception as e:
    print("list_files failed:", type(e).__name__, str(e)[:200])
    names = []
print("export files:", names)
for p in names:
    if p.endswith(".csv"):
        local = dataset_api.download(p, ".tmp/derived_out", overwrite=True)
        print("downloaded:", local)

os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"derived_from": ["rawa8af783", "rawb8af783"]}, f)
print("answers.json written")
