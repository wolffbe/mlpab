import hopsworks
import pandas as pd
import os

project = hopsworks.login()
fs = project.get_feature_store()
dataset_api = project.get_dataset_api()

# Download CSV files from HopsFS to local job temp dir
dataset_api.download("Resources/transactions.csv", local_path="/tmp/transactions.csv", overwrite=True)
dataset_api.download("Resources/fx_rates.csv", local_path="/tmp/fx_rates.csv", overwrite=True)

transactions = pd.read_csv("/tmp/transactions.csv")
fx_rates = pd.read_csv("/tmp/fx_rates.csv")

# amount_usd = amount * fx_rate for the row's currency
transactions = transactions.merge(fx_rates, on="currency", how="left")
transactions["amount_usd"] = transactions["amount"] * transactions["fx_rate"]

# is_weekend = 1 if Saturday (5) or Sunday (6) in UTC
transactions["event_time_dt"] = pd.to_datetime(transactions["event_time"], unit="ms", utc=True)
transactions["is_weekend"] = transactions["event_time_dt"].dt.dayofweek.isin([5, 6]).astype(int)

# amount_7d = rolling sum of amount for the same account over [event_time - 7days, event_time]
window_ms = 7 * 24 * 60 * 60 * 1000

def compute_7d_sum(group):
    group = group.sort_values("event_time").copy()
    times = group["event_time"].values
    amounts = group["amount"].values
    result = []
    for i, t in enumerate(times):
        mask = (times >= t - window_ms) & (times <= t)
        result.append(float(amounts[mask].sum()))
    group["amount_7d"] = result
    return group

transactions = transactions.groupby("account_id", group_keys=False).apply(compute_7d_sum)

result = transactions[["row_id", "account_id", "event_time", "amount_usd", "is_weekend", "amount_7d"]].copy()

# Create feature group with online store enabled
fg = fs.create_feature_group(
    name="featurese39b61",
    version=1,
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
    description="Derived transaction features",
)

fg.insert(result)
print("Done: inserted", len(result), "rows into featurese39b61 v1")
