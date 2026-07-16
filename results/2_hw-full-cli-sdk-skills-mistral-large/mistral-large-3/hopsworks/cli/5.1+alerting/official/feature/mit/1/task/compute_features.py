#!/usr/bin/env python3

import hopsworks
import pandas as pd
from datetime import datetime, timedelta

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Get feature groups
transactions_fg = fs.get_feature_group("transactions", version=1)
fx_rates_fg = fs.get_feature_group("fx_rates", version=1)

# Read data
transactions_df = transactions_fg.read()
fx_rates_df = fx_rates_fg.read()

# Join with fx_rates to compute amount_usd
merged_df = pd.merge(
    transactions_df,
    fx_rates_df,
    on="currency",
    how="left"
)
merged_df["amount_usd"] = merged_df["amount"] * merged_df["fx_rate"]

# Compute is_weekend
merged_df["event_time_dt"] = pd.to_datetime(merged_df["event_time"], unit="ms", utc=True)
merged_df["is_weekend"] = merged_df["event_time_dt"].dt.dayofweek.isin([5, 6]).astype(int)

# Compute amount_7d (7-day rolling sum for each account)
merged_df = merged_df.sort_values(["account_id", "event_time"])
merged_df["amount_7d"] = (
    merged_df.groupby("account_id")["amount"]
    .transform(lambda x: x.rolling("7D", on="event_time_dt").sum())
    .fillna(0)
)

# Prepare the output DataFrame
output_df = merged_df[["row_id", "account_id", "event_time", "amount_usd", "is_weekend", "amount_7d"]]

# Create or get the feature group
try:
    features_fg = fs.get_feature_group("features1af3ec", version=1)
except:
    features_fg = fs.create_feature_group(
        name="features1af3ec",
        version=1,
        description="Derived features from transactions",
        primary_key=["row_id"],
        event_time="event_time",
        online_enabled=True,
        features=[
            {"name": "row_id", "type": "string"},
            {"name": "account_id", "type": "string"},
            {"name": "event_time", "type": "bigint"},
            {"name": "amount_usd", "type": "double"},
            {"name": "is_weekend", "type": "int"},
            {"name": "amount_7d", "type": "double"},
        ]
    )

# Ingest the data
features_fg.insert(output_df)