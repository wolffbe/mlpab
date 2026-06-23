import hopsworks
import pandas as pd
import os

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Read all three batches
df1 = pd.read_csv("data/batch_1.csv")
df2 = pd.read_csv("data/batch_2.csv")
df3 = pd.read_csv("data/batch_3.csv")

# Combine all batches and keep only the latest revision per row_id
all_data = pd.concat([df1, df2, df3], ignore_index=True)
all_data = all_data.sort_values("updated_at", ascending=True)
latest = all_data.drop_duplicates(subset=["row_id"], keep="last").reset_index(drop=True)

print(f"Total rows across batches: {len(all_data)}")
print(f"Unique row_ids (latest revisions): {len(latest)}")
print(latest.head())

# Create the feature group
fg = fs.get_or_create_feature_group(
    name="accounts12723a",
    version=1,
    description="Accounts feature table with latest revisions",
    primary_key=["row_id"],
    event_time="updated_at",
    online_enabled=True,
)

print(f"Feature group created/retrieved: {fg.name} v{fg.version}")

# Insert the deduplicated latest data
fg.insert(latest, write_options={"wait_for_job": True})

print("Data inserted successfully.")
print(f"Feature group '{fg.name}' v{fg.version} is ready with {len(latest)} rows.")
