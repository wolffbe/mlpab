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
df1 = pd.read_csv("data/transactions_export_1.csv", dtype=dtypes)
df2 = pd.read_csv("data/transactions_export_2.csv", dtype=dtypes)
print("rows file1:", len(df1), "rows file2:", len(df2))

fg = fs.get_or_create_feature_group(
    name="transactions82e347",
    version=1,
    description="Transactions export (full table, deduplicated by row_id via primary-key upsert)",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
)

print("inserting file 1 ...")
fg.insert(df1, wait=True)
print("inserting file 2 (re-delivery, overlaps upserted by primary key) ...")
fg.insert(df2, wait=True)
print("done")
