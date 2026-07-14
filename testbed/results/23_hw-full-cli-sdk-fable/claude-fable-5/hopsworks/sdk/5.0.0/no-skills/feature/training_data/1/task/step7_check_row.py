import glob

import pandas as pd

df = pd.concat(
    [pd.read_parquet(p) for p in glob.glob("td_check/*.parquet")], ignore_index=True
)
cols = [
    "account_id",
    "label_time",
    "amount",
    "balance",
    "credit_score",
    "tier",
    "sessions_7d",
    "health_score",
    "churned",
]
print(df[df.account_id == "A0001"][cols].to_string())
print("rows:", len(df), "unique accounts:", df.account_id.nunique())
