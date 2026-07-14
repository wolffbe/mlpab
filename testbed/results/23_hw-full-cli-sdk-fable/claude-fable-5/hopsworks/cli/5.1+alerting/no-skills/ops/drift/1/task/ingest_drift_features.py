import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()
dataset_api = project.get_dataset_api()

local_path = dataset_api.download("Resources/features.csv", overwrite=True)
df = pd.read_csv(local_path)
df["event_time"] = pd.to_datetime(df["event_time"])

fg = fs.get_or_create_feature_group(
    name="drift_features",
    version=1,
    primary_key=["entity_id"],
    event_time="event_time",
    description="Daily feature observations for drift investigation",
    statistics_config={"enabled": True, "histograms": True, "correlations": False},
)
fg.insert(df, write_options={"wait_for_job": True})
print("Inserted rows:", len(df))
