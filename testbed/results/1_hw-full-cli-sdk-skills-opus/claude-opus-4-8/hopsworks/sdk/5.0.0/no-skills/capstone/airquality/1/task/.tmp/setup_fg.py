import warnings, urllib3
warnings.filterwarnings('ignore'); urllib3.disable_warnings()
import hopsworks, pandas as pd

proj = hopsworks.login()
fs = proj.get_feature_store()

df = pd.read_csv("data/airquality_history.csv")
df['date'] = pd.to_datetime(df['date'])
print("history shape", df.shape, "cols", list(df.columns))

fg = fs.get_or_create_feature_group(
    name="airq963ee7", version=1,
    description="Daily air quality + weather history with pm25 target",
    primary_key=["date"], event_time="date",
    online_enabled=True,
)
fg.insert(df, write_options={"wait_for_job": True})
print("FG inserted")
print("FEATURES", [f.name for f in fg.features])
