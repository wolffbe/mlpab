import os

os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)

import glob

import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

files = sorted(glob.glob("data/increment_*.csv"))
print("increment files:", files)

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df["row_id"] = df["row_id"].astype(str)
df["account_id"] = df["account_id"].astype(str)
df["category"] = df["category"].astype(str)
df["event_time"] = df["event_time"].astype("int64")
df["amount"] = df["amount"].astype("float64")
print("total rows:", len(df))
print(df.dtypes)

fg = fs.get_or_create_feature_group(
    name="incremental76da9e",
    version=1,
    description="Daily events increments",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
)

fg.insert(df, wait=True)
print("insert done")
print("feature group id:", fg.id)
