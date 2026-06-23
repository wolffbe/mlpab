import hopsworks
import pandas as pd

# Connect
project = hopsworks.login()
fs = project.get_feature_store()

# Load all batches
dfs = [
    pd.read_csv("data/batch_1.csv"),
    pd.read_csv("data/batch_2.csv"),
    pd.read_csv("data/batch_3.csv"),
]
combined = pd.concat(dfs, ignore_index=True)

# Keep latest revision per row_id
combined = combined.sort_values("updated_at").drop_duplicates(subset=["row_id"], keep="last")
combined = combined.reset_index(drop=True)

print(f"Rows after dedup: {len(combined)}")
print(combined.dtypes)
print(combined.head())

# Ensure correct dtypes
combined["row_id"] = combined["row_id"].astype(str)
combined["status"] = combined["status"].astype(str)
combined["balance"] = combined["balance"].astype(float)
combined["updated_at"] = combined["updated_at"].astype(int)

# Get or create feature group
try:
    fg = fs.get_feature_group("accounts12723a", version=1)
    print("Found existing feature group")
except Exception:
    fg = None

if fg is None:
    fg = fs.get_or_create_feature_group(
        name="accounts12723a",
        version=1,
        primary_key=["row_id"],
        event_time="updated_at",
        online_enabled=True,
        description="Accounts table with row_id as primary key and updated_at as event time",
    )
    print("Created feature group")

print(fg)

# Insert data
fg.insert(combined, write_options={"wait_for_job": True})
print("Insert complete")
