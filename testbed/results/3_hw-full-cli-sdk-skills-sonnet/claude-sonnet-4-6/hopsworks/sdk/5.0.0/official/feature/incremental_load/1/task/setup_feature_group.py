import hopsworks
import pandas as pd
import os

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Read all increment files and combine
dfs = []
for i in range(1, 7):
    df = pd.read_csv(f"data/increment_0{i}.csv")
    dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)
print(f"Total rows: {len(all_data)}")
print(all_data.dtypes)
print(all_data.head(2))

# Create feature group with online storage enabled
fg = fs.get_or_create_feature_group(
    name="incremental811051",
    version=1,
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
    description="Incremental events feature table"
)

print(f"Feature group created: {fg.name} v{fg.version}")

# Insert all data
fg.insert(all_data)
print("All increments loaded successfully")
