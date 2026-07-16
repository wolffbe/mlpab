import warnings, urllib3
warnings.filterwarnings('ignore'); urllib3.disable_warnings()
import pandas as pd
import hopsworks

df = pd.read_csv("data/prediction_log.csv")
df["ts"] = pd.to_datetime(df["ts"], utc=True)
df = df.sort_values("ts").reset_index(drop=True)
df["id"] = df.index.astype("int64")
df["event_time"] = df["ts"]
df["date"] = df["ts"].dt.strftime("%Y-%m-%d")
df["prediction"] = df["prediction"].astype("float64")
df = df[["id", "event_time", "date", "prediction"]]
print("rows", len(df))
print(df.head())
print("date range", df["date"].min(), df["date"].max())

proj = hopsworks.login()
fs = proj.get_feature_store()

from hsfs.statistics_config import StatisticsConfig
fg = fs.get_or_create_feature_group(
    name="prediction_log",
    version=1,
    description="Logged model predictions for monitoring",
    primary_key=["id"],
    event_time="event_time",
    online_enabled=False,
    statistics_config=StatisticsConfig(enabled=True, histograms=True, correlations=False),
)
fg.insert(df, write_options={"wait_for_job": True})
print("INSERT DONE; fg version", fg.version)
