import hopsworks
import pandas as pd

project = hopsworks.login()
dataset_api = project.get_dataset_api()
local_path = dataset_api.download("Resources/features926b2c.csv/features.csv", overwrite=True)
df = pd.read_csv(local_path)
df["account_id"] = df["account_id"].astype(str)
for col in ["f1", "f2", "f3", "f4"]:
    df[col] = df[col].astype(float)

fs = project.get_feature_store()
fg = fs.get_or_create_feature_group(
    name="profiles926b2c",
    version=1,
    primary_key=["account_id"],
    online_enabled=True,
    description="Account feature profiles",
)
fg.insert(df, wait=True)
print("INGEST_OK rows=%d" % len(df))
