import hopsworks
import pandas as pd

project = hopsworks.login()
df = pd.read_csv("/hopsfs/Resources/valid_events.csv")
df["row_id"] = df["row_id"].astype(str)
df["account_id"] = df["account_id"].astype(str)
df["event_time"] = df["event_time"].astype("int64")
df["amount"] = df["amount"].astype("float64")
df["category"] = df["category"].astype(str)
print("rows to insert:", len(df))
print(df.dtypes)

fs = project.get_feature_store()
fg = fs.get_or_create_feature_group(
    name="eventsee881b",
    version=1,
    description="Contract-valid events export",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
)
fg.insert(df, wait=True)
print("insert complete; fg id:", fg.id)
