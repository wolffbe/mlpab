import os

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

fv = fs.get_feature_view("churntraining30fee3", 1)
tds = fv.get_training_datasets()
for td in tds:
    print("TD version:", td.version, "location:", td.location, "format:", td.data_format)

dataset_api = project.get_dataset_api()
td = [t for t in tds if t.version == 1][0]
loc = td.location
# location like hopsfs://.../Projects/<p>/<p>_Training_Datasets/churntraining30fee3_1_1
rel = loc.split(f"/Projects/{project.name}/", 1)[1]
print("relative:", rel)


def walk(path, depth=0):
    for entry in dataset_api.list(path):
        print("  " * depth + entry)
        if "." not in entry.rsplit("/", 1)[-1]:
            try:
                walk(entry, depth + 1)
            except Exception:
                pass


files = []


def collect(path):
    for entry in dataset_api.list(path):
        name = entry.rsplit("/", 1)[-1]
        if name.endswith(".parquet"):
            files.append(entry)
        elif "." not in name:
            try:
                collect(entry)
            except Exception:
                pass


collect(rel)
print("parquet files:", files)

os.makedirs("td_check", exist_ok=True)
local = []
for f in files:
    p = dataset_api.download(f, "td_check", overwrite=True)
    local.append(p)
    print("downloaded:", p)

import pandas as pd

df = pd.concat([pd.read_parquet(p) for p in local], ignore_index=True)
print("shape:", df.shape)
print("columns:", list(df.columns))
print(df.head(5).to_string())
print("null counts:\n", df.isna().sum())
print("churned counts:\n", df["churned"].value_counts())
