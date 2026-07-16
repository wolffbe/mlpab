import hopsworks
from hopsworks_common.project import Project
import pandas as pd

# Connect to Hopsworks
hopsworks.login()
project = Project()
fs = project.get_feature_store()

# Read data
df = pd.read_csv("/hopsfs/Resources/transactions.csv")

print(f"Transactions shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Create feature group
fg_name = "cctxnee3558"
fg_version = 1

try:
    fg = fs.get_feature_group(fg_name, version=fg_version)
    print(f"Feature group {fg_name} already exists")
except Exception as e:
    print(f"Creating feature group: {e}")
    fg = fs.create_feature_group(
        name=fg_name,
        version=fg_version,
        primary_key=['transaction_id'],
        event_time='datetime',
        partition_key=['cc_num'],
        online_enabled=True,
        description="Credit card fraud detection features"
    )
    print(f"Created feature group {fg_name}")

# Insert data
fg.insert(df, write_options={"wait_for_job": True})
print(f"Inserted {len(df)} rows into feature group {fg_name}")
print("Done!")
