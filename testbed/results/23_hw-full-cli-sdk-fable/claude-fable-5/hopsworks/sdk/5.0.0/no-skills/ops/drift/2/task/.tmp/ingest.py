import hw_env  # noqa: F401
import hopsworks
import pandas as pd

proj = hopsworks.login()
fs = proj.get_feature_store()

df = pd.read_csv("data/features.csv")
df["event_time"] = pd.to_datetime(df["event_time"])
print(df.dtypes)
print(len(df), "rows")

fg = fs.get_or_create_feature_group(
    name="drift_features",
    version=1,
    primary_key=["entity_id"],
    event_time="event_time",
    statistics_config={"enabled": True, "histograms": True, "correlations": False},
    description="daily feature observations for drift investigation",
)
job, _ = fg.insert(df, wait=True)
print("insert done")
