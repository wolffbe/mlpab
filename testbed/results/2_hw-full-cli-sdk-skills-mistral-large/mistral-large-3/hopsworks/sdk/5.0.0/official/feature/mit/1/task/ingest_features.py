#!/usr/bin/env python3
"""
Ingest transformed transaction data into Hopsworks Feature Group.
"""
import os
import pandas as pd
import hopsworks
from datetime import datetime, timedelta

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Load data
transactions = pd.read_csv("data/transactions.csv")
fx_rates = pd.read_csv("data/fx_rates.csv")

# Join transactions with fx_rates
transactions = transactions.merge(fx_rates, on="currency", how="left")
transactions["amount_usd"] = transactions["amount"] * transactions["fx_rate"]

# Convert event_time to datetime (epoch milliseconds)
transactions["event_time_dt"] = pd.to_datetime(transactions["event_time"], unit="ms", utc=True)

# Compute is_weekend
transactions["is_weekend"] = transactions["event_time_dt"].dt.weekday.apply(
    lambda x: 1 if x >= 5 else 0
)

# Compute amount_7d (7-day rolling sum for each account)
def rolling_7d_sum(group):
    return group.rolling("7D", on="event_time_dt")["amount"].sum()

transactions["amount_7d"] = (
    transactions
    .sort_values("event_time_dt")
    .groupby("account_id", group_keys=False)
    .apply(rolling_7d_sum)
)

# Prepare the final DataFrame for ingestion
feature_df = transactions[["row_id", "account_id", "event_time", "amount_usd", "is_weekend", "amount_7d"]]

# Create or get the Feature Group
feature_group = fs.get_or_create_feature_group(
    name="features1af3ec",
    version=1,
    description="Transaction features for account-level analysis",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,  # Enable online (low-latency) access
)

# Ingest data
feature_group.insert(feature_df, write_options={"wait_for_job": True})

print("Feature Group 'features1af3ec', version 1, created and ingested successfully.")