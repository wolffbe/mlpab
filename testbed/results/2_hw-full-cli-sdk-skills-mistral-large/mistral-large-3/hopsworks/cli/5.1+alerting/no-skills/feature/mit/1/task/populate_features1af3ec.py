import hopsworks
import pandas as pd

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Get feature groups
transactions_fg = fs.get_feature_group("transactions_raw", version=1)
fx_rates_fg = fs.get_feature_group("fx_rates_raw", version=1)
features_fg = fs.get_feature_group("features1af3ec", version=1)

# Read data
transactions_df = transactions_fg.read()
fx_rates_df = fx_rates_fg.read()

# Join and transform
merged_df = transactions_df.merge(fx_rates_df, on="currency", how="left")
merged_df["amount_usd"] = merged_df["amount"] * merged_df["fx_rate"]

# Compute is_weekend
merged_df["event_time_dt"] = pd.to_datetime(merged_df["event_time"], unit="ms", utc=True)
merged_df["is_weekend"] = merged_df["event_time_dt"].dt.dayofweek >= 5
merged_df["is_weekend"] = merged_df["is_weekend"].astype(int)

# Compute amount_7d (7-day rolling sum)
merged_df = merged_df.sort_values(["account_id", "event_time"])
merged_df["amount_7d"] = merged_df.groupby("account_id")["amount"].transform(
    lambda x: x.rolling("7D", on="event_time_dt").sum()
)

# Select and insert
result_df = merged_df[["row_id", "account_id", "event_time", "amount_usd", "is_weekend", "amount_7d"]]
features_fg.insert(result_df, write_options={"wait_for_job": True})