"""Ingest job: load predictions.csv from HopsFS into feature group
predictions646af0 v1 (primary key row_id, online-enabled)."""
import hopsworks
import pandas as pd

project = hopsworks.login()
dataset_api = project.get_dataset_api()
dataset_api.download("Resources/trainjob646af0/predictions.csv", overwrite=True)
df = pd.read_csv("predictions.csv")

fs = project.get_feature_store()
fg = fs.get_or_create_feature_group(
    name="predictions646af0",
    version=1,
    primary_key=["row_id"],
    online_enabled=True,
    description="Predictions from trainjob646af0",
)
fg.insert(df, wait=True)
print("inserted", len(df), "rows into predictions646af0 v1")
