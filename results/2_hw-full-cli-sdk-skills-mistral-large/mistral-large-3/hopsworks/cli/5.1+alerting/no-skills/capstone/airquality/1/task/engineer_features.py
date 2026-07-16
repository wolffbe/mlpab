#!/usr/bin/env python3
"""
Engineer features for air quality forecasting:
- Compute rolling 3-day and 7-day averages for PM2.5.
- Ingest into the feature group `airq2408fa`.
"""

import hopsworks
import pandas as pd
import numpy as np

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Read the input data directly from the local file
# This file is available in the job environment as it was uploaded to /Resources
import os
if os.path.exists("data/airquality_history.csv"):
    df = pd.read_csv("data/airquality_history.csv", parse_dates=["date"])
elif os.path.exists("/Resources/airquality_history.csv"):
    df = pd.read_csv("/Resources/airquality_history.csv", parse_dates=["date"])
else:
    raise FileNotFoundError("airquality_history.csv not found in expected locations")

# Compute rolling averages
df["pm25_rolling_3d"] = df["pm25"].rolling(3, min_periods=1).mean()
df["pm25_rolling_7d"] = df["pm25"].rolling(7, min_periods=1).mean()

# Fill NaN values (if any) with forward fill, then backward fill
df.fillna(method="ffill", inplace=True)
df.fillna(method="bfill", inplace=True)

# Get the feature group
fg = fs.get_feature_group("airq2408fa", version=1)

# Ingest the data
fg.insert(df, write_options={"wait_for_job": True})

print("Feature engineering and ingestion completed successfully.")