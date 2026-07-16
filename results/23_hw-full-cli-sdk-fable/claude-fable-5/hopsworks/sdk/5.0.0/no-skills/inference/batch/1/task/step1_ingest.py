import hopsworks
import pandas as pd

df = pd.read_csv("data/feature_history.csv")
df["account_id"] = df["account_id"].astype(str)
df["event_time"] = df["event_time"].astype("int64")
print(df.dtypes)
print(len(df), df["account_id"].nunique())

project = hopsworks.login()
fs = project.get_feature_store()

fg = fs.get_or_create_feature_group(
    name="feature_history_4fa858",
    version=1,
    primary_key=["account_id"],
    event_time="event_time",
    online_enabled=False,
    description="Feature history revisions for batch scoring task",
)
fg.insert(df, wait=True)
print("inserted", len(df), "rows into", fg.name)
