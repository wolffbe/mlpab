import hopsworks
import pandas as pd

project = hopsworks.login()
ds = project.get_dataset_api()

local = ds.download("Resources/trainjob646af0/predictions.csv", "predictions.csv", overwrite=True)
df = pd.read_csv(local)
print(df.head(3))
print("rows:", len(df), "cols:", list(df.columns))

fs = project.get_feature_store()
fg = fs.get_or_create_feature_group(
    name="predictions646af0",
    version=1,
    primary_key=["row_id"],
    online_enabled=True,
    description="Predictions from job trainjob646af0",
)
job_result, validation = fg.insert(df, wait=True)
print("insert done")
print("materialization job state:", getattr(job_result, "state", job_result))
