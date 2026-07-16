import pandas as pd
import hopsworks
from hsfs.feature import Feature

# 1. Read both exports and dedupe on row_id (files overlap)
df1 = pd.read_csv("data/transactions_export_1.csv")
df2 = pd.read_csv("data/transactions_export_2.csv")
df = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset=["row_id"], keep="first").reset_index(drop=True)
print(f"file1={len(df1)} file2={len(df2)} combined_unique={len(df)}")
assert df["row_id"].nunique() == len(df), "row_id not unique"

# enforce types
df = df.astype({
    "row_id": "string",
    "account_id": "string",
    "event_time": "int64",
    "amount": "float64",
    "category": "string",
})

# 2. Connect
project = hopsworks.login()
fs = project.get_feature_store()

# 3. Create online-enabled FG
fg = fs.get_or_create_feature_group(
    name="transactions3cd0a6",
    version=1,
    description="Transactions export, deduplicated by row_id; online-enabled for low-latency lookup.",
    primary_key=["row_id"],
    event_time="event_time",
    features=[
        Feature("row_id", "string", description="Unique record key"),
        Feature("account_id", "string", description="Account identifier"),
        Feature("event_time", "bigint", description="Event time, epoch milliseconds"),
        Feature("amount", "double", description="Transaction amount"),
        Feature("category", "string", description="Transaction category"),
    ],
    online_enabled=True,
    stream=True,
    statistics_config=False,
)

# 4. Insert (blocking: online ingest + offline materialization)
fg.insert(df, wait=True)
print("fg.id =", fg.id)
print("online_enabled =", fg.online_enabled)

# 5. Verify offline
off = fg.read(dataframe_type="pandas")
print("offline rows:", len(off))
print("offline unique row_id:", off["row_id"].nunique())
