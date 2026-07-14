import pandas as pd
import hopsworks
from hsfs import engine

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("profiles926b2c", version=1)
print("FG:", fg.name, fg.version, "online:", fg.online_enabled,
      "format:", fg.time_travel_format, "stream:", fg.stream)

df = pd.read_csv("data/features.csv")
df["account_id"] = df["account_id"].astype(str)
for c in ["f1", "f2", "f3", "f4"]:
    df[c] = df[c].astype(float)

eng = engine.get_instance()
job = eng.legacy_save_dataframe(
    feature_group=fg,
    dataframe=df,
    operation="upsert",
    online_enabled=True,
    storage=None,
    offline_write_options={"wait_for_job": True},
    online_write_options={},
)
print("Ingestion job finished:", job.name if job is not None else None)
