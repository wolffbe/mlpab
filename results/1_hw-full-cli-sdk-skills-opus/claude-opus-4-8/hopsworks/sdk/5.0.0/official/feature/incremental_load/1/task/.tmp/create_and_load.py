import glob
import pandas as pd
import hopsworks
from hsfs.feature import Feature

project = hopsworks.login()
fs = project.get_feature_store()

fg = fs.get_or_create_feature_group(
    name="incremental614551",
    version=1,
    description="Daily increment events table (row_id keyed, event-time in epoch ms).",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
    stream=True,
    features=[
        Feature("row_id", "string", description="Unique record key"),
        Feature("account_id", "string", description="Account identifier"),
        Feature("event_time", "bigint", description="Event time, epoch milliseconds"),
        Feature("amount", "double", description="Transaction amount"),
        Feature("category", "string", description="Event category"),
    ],
)

files = sorted(glob.glob("data/increment_*.csv"))
print("Loading files:", files)
total = 0
for f in files:
    df = pd.read_csv(f, dtype={"row_id": str, "account_id": str, "category": str})
    df["event_time"] = df["event_time"].astype("int64")
    df["amount"] = df["amount"].astype("float64")
    print(f"{f}: {len(df)} rows")
    fg.insert(df, wait=True)
    total += len(df)

print("FG id:", fg.id)
print("Total rows inserted:", total)
print("online_enabled:", fg.online_enabled)
