import hopsworks
import pandas as pd

# Connect
project = hopsworks.login()
fs = project.get_feature_store()

# --- Version 1 ---
df_v1 = pd.read_csv("data/initial_export.csv")
print(f"V1 shape: {df_v1.shape}")
print(df_v1.dtypes)

fg_v1 = fs.get_or_create_feature_group(
    name="customers026c52",
    version=1,
    primary_key=["row_id"],
    event_time="updated_at",
    online_enabled=False,
    description="customers026c52 v1 - initial schema",
)
fg_v1.insert(df_v1, wait=True)
print("V1 inserted successfully")

# --- Version 2 ---
df_v2 = pd.read_csv("data/reload/new_export.csv")
print(f"V2 shape: {df_v2.shape}")
print(df_v2.dtypes)

fg_v2 = fs.get_or_create_feature_group(
    name="customers026c52",
    version=2,
    primary_key=["row_id"],
    event_time="updated_at",
    online_enabled=True,
    description="customers026c52 v2 - new schema with full_name, balance, currency",
)
fg_v2.insert(df_v2, wait=True)
print("V2 inserted successfully")
print("Done.")
