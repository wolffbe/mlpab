"""Batch-score accounts as of T and write scores4fa858 v1 (online-enabled).

Runs inside the Hopsworks cluster as a PYTHON job.
"""

import math

import hopsworks
import pandas as pd

T = 1773565200000
WEIGHTS = {"f1": -0.1306, "f2": 0.0121, "f3": -0.7418}
BIAS = -0.8397

project = hopsworks.login()
dataset_api = project.get_dataset_api()
local_path = dataset_api.download("Resources/feature_history.csv", overwrite=True)

df = pd.read_csv(local_path)
total_accounts = df["account_id"].nunique()

# Most recent revision at or before T, per account.
valid = df[df["event_time"] <= T]
valid = valid.sort_values(["account_id", "event_time"])
asof = valid.groupby("account_id", as_index=False).tail(1)

print(f"total accounts: {total_accounts}, accounts with revision <= T: {len(asof)}")
assert len(asof) == total_accounts, "some accounts have no revision at or before T"

def score(row):
    z = (
        WEIGHTS["f1"] * row["f1"]
        + WEIGHTS["f2"] * row["f2"]
        + WEIGHTS["f3"] * row["f3"]
        + BIAS
    )
    return round(1.0 / (1.0 + math.exp(-z)), 6)

scores = pd.DataFrame(
    {
        "account_id": asof["account_id"].astype(str),
        "score": asof.apply(score, axis=1).astype("float64"),
    }
).reset_index(drop=True)

print(scores.head(10))
print(f"rows to write: {len(scores)}")

fs = project.get_feature_store()
fg = fs.get_or_create_feature_group(
    name="scores4fa858",
    version=1,
    description="Batch scores as of T=1773565200000",
    primary_key=["account_id"],
    online_enabled=True,
)
fg.insert(scores, wait=True)
print("insert complete")
