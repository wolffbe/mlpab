"""Platform-side ingestion for the customers4baff7 full-reload task.

Runs as a Hopsworks PYTHON job:
  1. Loads data/initial_export.csv into feature group customers4baff7 v1.
  2. Re-creates customers4baff7 v2 from scratch with the new schema
     (online-enabled) containing exactly the rows of the new export.
"""
import pandas as pd
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
dataset_api = project.get_dataset_api()

local_v1 = dataset_api.download("Resources/full_reload/initial_export.csv", overwrite=True)
local_v2 = dataset_api.download("Resources/full_reload/new_export.csv", overwrite=True)

df1 = pd.read_csv(local_v1)
df1["row_id"] = df1["row_id"].astype(str)
df1["name"] = df1["name"].astype(str)
df1["balance_eur"] = df1["balance_eur"].astype("float64")
df1["updated_at"] = df1["updated_at"].astype("int64")
print("initial export:", df1.shape, df1.dtypes.to_dict())

fg1 = fs.get_or_create_feature_group(
    name="customers4baff7",
    version=1,
    primary_key=["row_id"],
    event_time="updated_at",
    description="Customers initial export (original schema)",
)
fg1.insert(df1, wait=True)
print("v1 insert done, rows:", len(df1))

df2 = pd.read_csv(local_v2)
df2["row_id"] = df2["row_id"].astype(str)
df2["full_name"] = df2["full_name"].astype(str)
df2["balance"] = df2["balance"].astype("float64")
df2["currency"] = df2["currency"].astype(str)
df2["updated_at"] = df2["updated_at"].astype("int64")
print("new export:", df2.shape, df2.dtypes.to_dict())

# Re-create v2 from scratch: drop any existing v2 so no stale rows survive.
try:
    old = fs.get_feature_group("customers4baff7", version=2)
    if old is not None:
        old.delete()
        print("deleted pre-existing v2")
except Exception as e:
    print("no pre-existing v2:", e)

fg2 = fs.create_feature_group(
    name="customers4baff7",
    version=2,
    primary_key=["row_id"],
    event_time="updated_at",
    online_enabled=True,
    description="Customers full re-export (new schema: full_name, balance, currency)",
)
fg2.insert(df2, wait=True)
print("v2 insert done, rows:", len(df2))
print("SUCCESS")
