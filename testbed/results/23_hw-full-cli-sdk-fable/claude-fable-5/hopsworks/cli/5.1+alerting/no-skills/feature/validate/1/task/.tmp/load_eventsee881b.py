import json

import hopsworks
import pandas as pd

VALID_CATEGORIES = {"grocery", "travel", "salary", "rent", "other"}

project = hopsworks.login()
fs = project.get_feature_store()
dataset_api = project.get_dataset_api()

local_csv = dataset_api.download("Resources/events.csv", overwrite=True)
df = pd.read_csv(
    local_csv,
    dtype={"row_id": str, "account_id": str, "category": str},
)
df["event_time"] = df["event_time"].astype("int64")

valid_mask = (
    df["amount"].notna()
    & (df["amount"] >= 0)
    & (df["amount"] <= 10000)
    & df["category"].isin(VALID_CATEGORIES)
)
valid = df[valid_mask].copy()
rejected = df.loc[~valid_mask, "row_id"].tolist()
print(f"total={len(df)} valid={len(valid)} rejected={len(rejected)}")

valid["amount"] = valid["amount"].astype("float64")

fg = fs.get_or_create_feature_group(
    name="eventsee881b",
    version=1,
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
    description="Events satisfying the data contract (validated load)",
)
fg.insert(valid, wait=True)
print("insert done")

with open("rejected.json", "w") as f:
    json.dump({"rejected": rejected}, f)
dataset_api.upload("rejected.json", "Resources", overwrite=True)
print("rejected list uploaded to Resources/rejected.json")
