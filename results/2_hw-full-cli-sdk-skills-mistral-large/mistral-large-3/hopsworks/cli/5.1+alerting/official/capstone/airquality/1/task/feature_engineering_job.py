#!/usr/bin/env python3
"""
Job to engineer features and insert into feature group `airq2408fa`.
"""

import hopsworks
import pandas as pd
import numpy as np

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Load raw data
raw_df = pd.read_csv("/Resources/airquality_history.csv", parse_dates=["date"])

# Sort by date
raw_df = raw_df.sort_values("date").reset_index(drop=True)

# Lag features
raw_df["pm25_lag2"] = raw_df["pm25"].shift(2)
raw_df["pm25_lag3"] = raw_df["pm25"].shift(3)

# Rolling features
raw_df["pm25_rolling_3d"] = raw_df["pm25"].rolling(3, min_periods=1).mean()
raw_df["pm25_rolling_7d"] = raw_df["pm25"].rolling(7, min_periods=1).mean()

# Interaction feature
raw_df["temp_humidity_interaction"] = raw_df["temperature"] * raw_df["humidity"]

# Drop rows with NaN (due to lags)
enriched_df = raw_df.dropna()

# Get feature group
fg = fs.get_feature_group("airq2408fa", version=1)

# Insert enriched data
fg.insert(enriched_df, write_options={"wait_for_job": True})