import hopsworks
import pandas as pd

# Load data
transactions = pd.read_csv("data/transactions.csv")
fx_rates = pd.read_csv("data/fx_rates.csv")

# Merge fx_rates
df = transactions.merge(fx_rates, on="currency", how="left")

# amount_usd
df["amount_usd"] = df["amount"] * df["fx_rate"]

# is_weekend: convert epoch ms to datetime UTC
df["event_dt"] = pd.to_datetime(df["event_time"], unit="ms", utc=True)
df["is_weekend"] = df["event_dt"].dt.dayofweek.isin([5, 6]).astype(int)

# amount_7d: rolling 7-day sum of amount per account
# Sort by account and time for window calculation
df = df.sort_values(["account_id", "event_time"]).reset_index(drop=True)

def rolling_7d(group):
    group = group.sort_values("event_time")
    results = []
    times = group["event_time"].values
    amounts = group["amount"].values
    for i in range(len(times)):
        window_start = times[i] - 7 * 24 * 3600 * 1000  # 7 days in ms
        window_end = times[i]
        mask = (times >= window_start) & (times <= window_end)
        results.append(float(amounts[mask].sum()))
    group = group.copy()
    group["amount_7d"] = results
    return group

df = df.groupby("account_id", group_keys=False).apply(rolling_7d)

# Keep only required columns
df_final = df[["row_id", "account_id", "event_time", "amount_usd", "is_weekend", "amount_7d"]].copy()

print(f"Rows: {len(df_final)}")
print(df_final.dtypes)
print(df_final.head(3))

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
    description="Derived feature table with amount_usd, is_weekend, amount_7d",
)

# Insert data
fg.insert(df_final, wait=True)

print("Feature group created and data inserted successfully.")
