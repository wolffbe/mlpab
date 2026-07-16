import hopsworks
import pandas as pd
import os
import json

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

print("Connected to Hopsworks")

# Create/get feature group
fg = fs.get_or_create_feature_group(
    name="incremental811051",
    version=1,
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
    description="Incremental events table with daily ingestion"
)

print(f"Feature group created/retrieved: {fg.name} v{fg.version}")

# Load all increment files
data_dir = "./data"
dfs = []
for i in range(1, 7):
    filepath = os.path.join(data_dir, f"increment_{i:02d}.csv")
    df = pd.read_csv(filepath)
    dfs.append(df)
    print(f"Loaded {filepath}: {len(df)} rows")

all_data = pd.concat(dfs, ignore_index=True)
print(f"Total rows: {len(all_data)}")
print(f"Columns: {all_data.columns.tolist()}")
print(f"Dtypes:\n{all_data.dtypes}")

# Insert data into feature group
fg.insert(all_data)
print("Data inserted successfully")
