"""Platform-side ingestion job: load merged transactions export into a feature group."""
import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()
dataset_api = project.get_dataset_api()

local_csv = dataset_api.download("Resources/transactions_merged.csv", overwrite=True)
df = pd.read_csv(local_csv)

df["row_id"] = df["row_id"].astype(str)
df["account_id"] = df["account_id"].astype(str)
df["category"] = df["category"].astype(str)
df["event_time"] = df["event_time"].astype("int64")
df["amount"] = df["amount"].astype("float64")

assert df["row_id"].is_unique, "row_id must be unique"
print(f"Loaded {len(df)} unique rows")

fg = fs.get_or_create_feature_group(
    name="transactions82e347",
    version=1,
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
    description="Transactions table (merged export, deduplicated by row_id)",
)
fg.insert(df, wait=True)
print("Insert complete")
