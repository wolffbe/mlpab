import hopsworks
import pandas as pd

df = pd.read_csv("data/prediction_log.csv")
df["ts"] = pd.to_datetime(df["ts"], utc=True)
df["pred_id"] = list(range(1, len(df) + 1))
print(df.shape, df["ts"].min(), df["ts"].max())

proj = hopsworks.login()
fs = proj.get_feature_store()

fg = fs.get_or_create_feature_group(
    name="prediction_log",
    version=1,
    description="Deployed model logged predictions for monitoring",
    primary_key=["pred_id"],
    event_time="ts",
    online_enabled=False,
)
fg.insert(df, wait=True)
print("inserted")
