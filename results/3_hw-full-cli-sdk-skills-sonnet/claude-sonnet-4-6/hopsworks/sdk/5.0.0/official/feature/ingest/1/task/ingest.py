import hopsworks
import pandas as pd
import hsfs

# Load and deduplicate data
df1 = pd.read_csv("data/transactions_export_1.csv")
df2 = pd.read_csv("data/transactions_export_2.csv")
df = pd.concat([df1, df2]).drop_duplicates(subset=["row_id"]).reset_index(drop=True)
print(f"Total rows after dedup: {len(df)}")
print(df.dtypes)
print(df.head(3))

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Create feature group
fg = fs.get_or_create_feature_group(
    name="transactions9c0e90",
    version=1,
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
    description="Transactions feature table",
)
print(f"Feature group: {fg.name} v{fg.version}")

# Insert data
fg.insert(df, write_options={"wait_for_job": True})
print("Insert complete.")
