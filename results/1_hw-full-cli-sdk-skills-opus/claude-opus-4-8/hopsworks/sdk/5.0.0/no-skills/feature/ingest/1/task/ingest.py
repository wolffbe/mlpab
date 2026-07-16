import pandas as pd
import hopsworks

# --- Read both exports (marshal into a DataFrame to hand to the SDK) ---
df1 = pd.read_csv("data/transactions_export_1.csv")
df2 = pd.read_csv("data/transactions_export_2.csv")
combined = pd.concat([df1, df2], ignore_index=True)
print("Export 1 rows:", len(df1))
print("Export 2 rows:", len(df2))
print("Combined rows (with overlap):", len(combined))
print("Unique row_id:", combined["row_id"].nunique())

# Enforce correct dtypes per schema.md
combined["row_id"] = combined["row_id"].astype(str)
combined["account_id"] = combined["account_id"].astype(str)
combined["event_time"] = combined["event_time"].astype("int64")
combined["amount"] = combined["amount"].astype("float64")
combined["category"] = combined["category"].astype(str)

# --- Connect to the platform ---
proj = hopsworks.login()
fs = proj.get_feature_store()

# --- Register the feature table (online + offline) ---
fg = fs.get_or_create_feature_group(
    name="transactions3cd0a6",
    version=1,
    description="Transactions table ingested from two overlapping exports",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
)

# --- Load the full export; primary-key upsert dedupes overlapping rows ---
res = fg.insert(combined)
print("Insert returned:", res)
print("Feature group online_enabled:", fg.online_enabled)
print("Feature group id:", fg.id)
