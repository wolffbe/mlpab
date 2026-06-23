import pandas as pd
import hopsworks
from hsfs.feature import Feature

project = hopsworks.login()
fs = project.get_feature_store()

# ---------- Version 1: initial schema ----------
df_v1 = pd.read_csv("data/initial_export.csv")
df_v1["row_id"] = df_v1["row_id"].astype(str)
df_v1["name"] = df_v1["name"].astype(str)
df_v1["balance_eur"] = df_v1["balance_eur"].astype(float)
df_v1["updated_at"] = df_v1["updated_at"].astype("int64")
print("v1 shape:", df_v1.shape, "cols:", list(df_v1.columns))

fg1 = fs.get_or_create_feature_group(
    name="customersa8deb9",
    version=1,
    description="Customers initial export (original schema).",
    primary_key=["row_id"],
    event_time="updated_at",
    features=[
        Feature("row_id", "string", description="Unique record key"),
        Feature("name", "string", description="Customer name"),
        Feature("balance_eur", "double", description="Balance, always in EUR"),
        Feature("updated_at", "bigint", description="Event time, epoch milliseconds"),
    ],
    online_enabled=False,
    statistics_config=False,
)
fg1.insert(df_v1, wait=True)
print("v1 inserted, id:", fg1.id)

# ---------- Version 2: new breaking schema, online-enabled ----------
df_v2 = pd.read_csv("data/reload/new_export.csv")
df_v2["row_id"] = df_v2["row_id"].astype(str)
df_v2["full_name"] = df_v2["full_name"].astype(str)
df_v2["balance"] = df_v2["balance"].astype(float)
df_v2["currency"] = df_v2["currency"].astype(str)
df_v2["updated_at"] = df_v2["updated_at"].astype("int64")
print("v2 shape:", df_v2.shape, "cols:", list(df_v2.columns))

fg2 = fs.get_or_create_feature_group(
    name="customersa8deb9",
    version=2,
    description="Customers full re-export (new breaking schema), online-enabled.",
    primary_key=["row_id"],
    event_time="updated_at",
    features=[
        Feature("row_id", "string", description="Unique record key"),
        Feature("full_name", "string", description="Customer full name (replaces name)"),
        Feature("balance", "double", description="Balance; currency varies"),
        Feature("currency", "string", description="ISO currency code"),
        Feature("updated_at", "bigint", description="Event time, epoch milliseconds"),
    ],
    online_enabled=True,
    stream=True,
    statistics_config=False,
)
fg2.insert(df_v2, wait=True)
print("v2 inserted, id:", fg2.id)
print("DONE")
