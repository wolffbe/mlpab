import os

os.environ["NO_PROXY"] = ""
os.environ["no_proxy"] = ""

import urllib3

urllib3.disable_warnings()

import hopsworks
import pandas as pd

proj = hopsworks.login(hostname_verification=False)
fs = proj.get_feature_store()

# --- Version 1: initial export (old schema) ---
df1 = pd.read_csv("data/initial_export.csv")
df1["row_id"] = df1["row_id"].astype(str)
df1["name"] = df1["name"].astype(str)
df1["balance_eur"] = df1["balance_eur"].astype("float64")
df1["updated_at"] = df1["updated_at"].astype("int64")
print("v1 rows:", len(df1), "cols:", list(df1.columns))

fg1 = fs.get_or_create_feature_group(
    name="customers4baff7",
    version=1,
    description="Customers table, initial export (original schema)",
    primary_key=["row_id"],
    event_time="updated_at",
    online_enabled=False,
)
fg1.insert(df1, wait=True)
print("v1 inserted")

# --- Version 2: full reload with the new, breaking schema ---
df2 = pd.read_csv("data/reload/new_export.csv")
df2["row_id"] = df2["row_id"].astype(str)
df2["full_name"] = df2["full_name"].astype(str)
df2["balance"] = df2["balance"].astype("float64")
df2["currency"] = df2["currency"].astype(str)
df2["updated_at"] = df2["updated_at"].astype("int64")
print("v2 rows:", len(df2), "cols:", list(df2.columns))

# re-create from scratch: drop any pre-existing v2 so no stale rows/columns remain
try:
    existing = fs.get_feature_group("customers4baff7", version=2)
    if existing is not None:
        existing.delete()
        print("deleted pre-existing v2")
except Exception as e:
    print("no pre-existing v2:", type(e).__name__)

fg2 = fs.create_feature_group(
    name="customers4baff7",
    version=2,
    description="Customers table, full re-export (new schema), online-enabled",
    primary_key=["row_id"],
    event_time="updated_at",
    online_enabled=True,
)
fg2.insert(df2, wait=True)
print("v2 inserted")
