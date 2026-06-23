import hopsworks
import pandas as pd
import numpy as np

# Load data
tx = pd.read_csv("data/transactions.csv")
fx = pd.read_csv("data/fx_rates.csv")

# amount_usd
tx = tx.merge(fx, on="currency", how="left")
tx["amount_usd"] = tx["amount"] * tx["fx_rate"]

# is_weekend: event_time is epoch milliseconds
tx["event_time"] = tx["event_time"].astype("int64")
tx["_dt"] = pd.to_datetime(tx["event_time"], unit="ms", utc=True)
tx["is_weekend"] = tx["_dt"].dt.dayofweek.isin([5, 6]).astype(int)

# amount_7d: rolling 7-day sum of amount per account (inclusive on both ends)
tx = tx.sort_values("event_time").reset_index(drop=True)
window_ms = 7 * 24 * 3600 * 1000  # 7 days in ms

def rolling_7d_sum(group):
    times = group["event_time"].values
    amounts = group["amount"].values
    result = np.empty(len(times), dtype=float)
    for i in range(len(times)):
        mask = (times >= times[i] - window_ms) & (times <= times[i])
        result[i] = amounts[mask].sum()
    return pd.Series(result, index=group.index)

tx["amount_7d"] = tx.groupby("account_id", group_keys=False).apply(rolling_7d_sum)

# Keep only required columns
df = tx[["row_id", "account_id", "event_time", "amount_usd", "is_weekend", "amount_7d"]].copy()
df["is_weekend"] = df["is_weekend"].astype(int)
df["amount_7d"] = df["amount_7d"].astype(float)
df["amount_usd"] = df["amount_usd"].astype(float)

print(f"DataFrame shape: {df.shape}")
print(df.head())
print(df.dtypes)

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Create feature group
fg = fs.get_or_create_feature_group(
    name="featurese39b61",
    version=1,
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
    description="Derived transaction features with FX conversion, weekend flag, and 7-day rolling sum",
)

fg.insert(df)
print("Feature group created and data inserted successfully.")
print(f"Feature group: {fg.name} v{fg.version}")
