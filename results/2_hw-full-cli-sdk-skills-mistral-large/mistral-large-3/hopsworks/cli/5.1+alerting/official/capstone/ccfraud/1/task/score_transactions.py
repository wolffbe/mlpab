#!/usr/bin/env python3
"""
Score transactions using the trained fraud classifier and write predictions to ccpredc444ca.
"""

import hopsworks
import pandas as pd
import numpy as np
import math
import joblib
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
mr = project.get_model_registry()

# Read input data
score_df = pd.read_csv("/hopsfs/Resources/score_transactions.csv")

# Convert datetime to pandas datetime
score_df["datetime"] = pd.to_datetime(score_df["datetime"])

# Sort by cc_num and datetime for rolling calculations
score_df = score_df.sort_values(["cc_num", "datetime"])

# Feature 1: Transaction velocity (count per card in the last hour)
# Group by card and calculate rolling count
velocity = (
    score_df.set_index("datetime")
    .groupby("cc_num")
    .rolling("1h")
    .transaction_id.count()
    .reset_index()
    .rename(columns={"transaction_id": "velocity_1h"})
)
score_df = score_df.merge(velocity, on=["cc_num", "datetime"], how="left")

# Feature 2: Geo distance from the card's "home" location (first transaction in training data)
# Fetch home locations from the training feature group
train_fg = fs.get_feature_group("cctxnc444ca", version=1)
train_df = train_fg.read()
home_locations = train_df.groupby("cc_num").first()[["lat", "long"]]
score_df["home_lat"] = score_df["cc_num"].map(home_locations["lat"])
score_df["home_long"] = score_df["cc_num"].map(home_locations["long"])
score_df["geo_distance_km"] = score_df.apply(
    lambda row: haversine(
        row["home_lat"], row["home_long"], row["lat"], row["long"]
    ) if not pd.isna(row["home_lat"]) else np.nan,
    axis=1,
)

# Feature 3: Rolling average amount per card in the last hour
amount_avg = (
    score_df.set_index("datetime")
    .groupby("cc_num")
    .rolling("1h")
    .amount.mean()
    .reset_index()
    .rename(columns={"amount": "amount_rolling_avg_1h"})
)
score_df = score_df.merge(amount_avg, on=["cc_num", "datetime"], how="left")

# Drop temporary columns
score_df = score_df.drop(columns=["home_lat", "home_long"])

# Load model
model = mr.get_model("ccmodelc444ca", version=1)
model_dir = model.download()
clf = joblib.load(f"{model_dir}/fraud_model.pkl")

# Prepare features for scoring
X = score_df.drop(columns=["transaction_id", "datetime", "merchant", "category"])

# Fill missing values with 0 (for new cards or missing features)
X = X.fillna(0)

# Predict
score_df["fraud_probability"] = clf.predict_proba(X)[:, 1]

# Write predictions to feature group
pred_fg = fs.get_feature_group("ccpredc444ca", version=1)
pred_fg.insert(score_df[["transaction_id", "fraud_probability"]], write_options={"wait_for_job": True})

print("Scoring completed successfully.")