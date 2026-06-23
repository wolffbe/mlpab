import hopsworks
import pandas as pd

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Load and deduplicate both export files
df1 = pd.read_csv("data/transactions_export_1.csv")
df2 = pd.read_csv("data/transactions_export_2.csv")

# Combine and deduplicate by row_id (keep first occurrence)
df = pd.concat([df1, df2]).drop_duplicates(subset=["row_id"]).reset_index(drop=True)
print(f"Total unique rows: {len(df)}")
print(df.dtypes)
print(df.head(3))

# Ensure correct dtypes
df["row_id"] = df["row_id"].astype(str)
df["account_id"] = df["account_id"].astype(str)
df["event_time"] = df["event_time"].astype("int64")
df["amount"] = df["amount"].astype(float)
df["category"] = df["category"].astype(str)

# Create feature group (feature table)
fg = fs.get_or_create_feature_group(
    name="transactions9c0e90",
    version=1,
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
    description="Transactions feature table",
)

print("Feature group created/retrieved:", fg)

# Insert all data
fg.insert(df, wait=True)
print("Insert complete.")
