import hopsworks
import pandas as pd

# Connect to Hopsworks
print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

# ── Version 1 ─────────────────────────────────────────────────────────────────
print("\nLoading initial export...")
df_v1 = pd.read_csv("data/initial_export.csv")
print(f"V1 shape: {df_v1.shape}")
print(df_v1.dtypes)

# Ensure correct types
df_v1["row_id"] = df_v1["row_id"].astype(str)
df_v1["name"] = df_v1["name"].astype(str)
df_v1["balance_eur"] = df_v1["balance_eur"].astype(float)
df_v1["updated_at"] = df_v1["updated_at"].astype(int)

print("\nCreating feature group v1...")
fg_v1 = fs.get_or_create_feature_group(
    name="customers026c52",
    version=1,
    primary_key=["row_id"],
    event_time="updated_at",
    online_enabled=False,
    description="customers v1 initial schema",
)

print("Inserting data into v1...")
fg_v1.insert(df_v1)
print("V1 done.")

# ── Version 2 ─────────────────────────────────────────────────────────────────
print("\nLoading new export...")
df_v2 = pd.read_csv("data/reload/new_export.csv")
print(f"V2 shape: {df_v2.shape}")
print(df_v2.dtypes)

# Ensure correct types
df_v2["row_id"] = df_v2["row_id"].astype(str)
df_v2["full_name"] = df_v2["full_name"].astype(str)
df_v2["balance"] = df_v2["balance"].astype(float)
df_v2["currency"] = df_v2["currency"].astype(str)
df_v2["updated_at"] = df_v2["updated_at"].astype(int)

print("\nCreating feature group v2 with online enabled...")
fg_v2 = fs.get_or_create_feature_group(
    name="customers026c52",
    version=2,
    primary_key=["row_id"],
    event_time="updated_at",
    online_enabled=True,
    description="customers v2 new schema with full_name, balance, currency",
)

print("Inserting data into v2...")
fg_v2.insert(df_v2)
print("V2 done.")

print("\nAll done! Feature group customers026c52 v1 and v2 created successfully.")
