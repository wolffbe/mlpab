"""Hopsworks platform job: build featureseb4964 v1 from uploaded CSVs."""

import hopsworks
import numpy as np
import pandas as pd

project = hopsworks.login()

tx_path = "/hopsfs/Resources/featureseb4964/transactions.csv"
fx_path = "/hopsfs/Resources/featureseb4964/fx_rates.csv"

tx = pd.read_csv(tx_path)
fx = pd.read_csv(fx_path)

df = tx.merge(fx, on="currency", how="left")
df["amount_usd"] = df["amount"] * df["fx_rate"]

dt = pd.to_datetime(df["event_time"], unit="ms", utc=True)
df["is_weekend"] = (dt.dt.dayofweek >= 5).astype("int64")

WINDOW_MS = 7 * 24 * 3600 * 1000
df = df.sort_values(["account_id", "event_time"], kind="mergesort").reset_index(drop=True)
amount_7d = np.zeros(len(df), dtype="float64")
for _, g in df.groupby("account_id"):
    times = g["event_time"].to_numpy(dtype="int64")
    amts = g["amount"].to_numpy(dtype="float64")
    csum = np.concatenate([[0.0], np.cumsum(amts)])
    # window [t - 7d, t], inclusive on both ends
    left = np.searchsorted(times, times - WINDOW_MS, side="left")
    right = np.searchsorted(times, times, side="right")
    amount_7d[g.index.to_numpy()] = csum[right] - csum[left]
df["amount_7d"] = amount_7d

result = df[["row_id", "account_id", "event_time", "amount_usd", "is_weekend", "amount_7d"]].copy()
result["row_id"] = result["row_id"].astype(str)
result["account_id"] = result["account_id"].astype(str)
result["event_time"] = result["event_time"].astype("int64")
result["amount_usd"] = result["amount_usd"].astype("float64")
result["amount_7d"] = result["amount_7d"].astype("float64")
result = result.sort_values("row_id").reset_index(drop=True)
print("rows:", len(result))
print(result.head())

fs = project.get_feature_store()
fg = fs.get_or_create_feature_group(
    name="featureseb4964",
    version=1,
    description="Derived transaction features: USD amount, weekend flag, 7d rolling sum",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
)
fg.insert(result, wait=True)
print("insert done")
