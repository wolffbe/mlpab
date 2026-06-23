#!/usr/bin/env python3
"""
Feature engineering for credit card fraud detection.
Ingests transactions.csv, engineers features, and writes to feature group cctxnc444ca.
"""

import hopsworks
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

# Haversine distance function (km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (
        math.sin(dLat / 2) * math.sin(dLat / 2) +
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
        math.sin(dLon / 2) * math.sin(dLon / 2)
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Read input data
transactions_df = pd.read_csv("/hopsfs/Resources/transactions.csv")
# Cast columns to match feature group schema
transactions_df["cc_num"] = transactions_df["cc_num"].astype(str)
transactions_df["is_fraud"] = pd.to_numeric(transactions_df["is_fraud"], downcast="integer")

# Convert datetime to pandas datetime
transactions_df["datetime"] = pd.to_datetime(transactions_df["datetime"])

# Sort by cc_num and datetime for rolling calculations
transactions_df = transactions_df.sort_values(["cc_num", "datetime"])

# Feature 1: Transaction velocity (count per card in the last hour)
# Sort by datetime for rolling calculations
transactions_df = transactions_df.sort_values(["cc_num", "datetime"])
# Group by card and calculate rolling count
velocity = (
    transactions_df.set_index("datetime")
    .groupby("cc_num")
    .rolling("1h")
    .transaction_id.count()
    .reset_index()
    .rename(columns={"transaction_id": "velocity_1h"})
)
transactions_df = transactions_df.merge(velocity, on=["cc_num", "datetime"], how="left")

# Feature 2: Geo distance from the card's "home" location (first transaction)
home_locations = transactions_df.groupby("cc_num").first()[["lat", "long"]]
transactions_df["home_lat"] = transactions_df["cc_num"].map(home_locations["lat"])
transactions_df["home_long"] = transactions_df["cc_num"].map(home_locations["long"])
transactions_df["geo_distance_km"] = transactions_df.apply(
    lambda row: haversine(
        row["home_lat"], row["home_long"], row["lat"], row["long"]
    ),
    axis=1,
)

# Feature 3: Rolling average amount per card in the last hour
amount_avg = (
    transactions_df.set_index("datetime")
    .groupby("cc_num")
    .rolling("1h")
    .amount.mean()
    .reset_index()
    .rename(columns={"amount": "amount_rolling_avg_1h"})
)
transactions_df = transactions_df.merge(amount_avg, on=["cc_num", "datetime"], how="left")

# Drop temporary columns
transactions_df = transactions_df.drop(columns=["home_lat", "home_long"])

# Write to feature group
fg = fs.get_feature_group("cctxnc444ca", version=1)
fg.insert(transactions_df, write_options={"wait_for_job": True})

print("Feature engineering and ingestion completed successfully.")