import pandas as pd

import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

dtypes = {
    "row_id": "string",
    "account_id": "string",
    "event_time": "int64",
    "amount": "float64",
    "category": "string",
}

p1 = "/hopsfs/Resources/transactions_ingest/transactions_export_1.csv"
p2 = "/hopsfs/Resources/transactions_ingest/transactions_export_2.csv"
df1 = pd.read_csv(p1, dtype=dtypes)
df2 = pd.read_csv(p2, dtype=dtypes)
print("rows file1:", len(df1), "rows file2:", len(df2), flush=True)

fg = fs.get_or_create_feature_group(
    name="transactions82e347",
    version=1,
    description="Transactions export (full table, deduplicated by row_id via primary-key upsert)",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
)

print("inserting file 1 ...", flush=True)
fg.insert(df1, wait=True)
print("inserting file 2 (re-delivery, overlap upserted by primary key) ...", flush=True)
fg.insert(df2, wait=True)

check = fg.read()
print("OFFLINE_COUNT:", len(check), flush=True)
print("OFFLINE_UNIQUE_ROW_IDS:", check["row_id"].nunique(), flush=True)
print("INGEST_JOB_DONE", flush=True)
