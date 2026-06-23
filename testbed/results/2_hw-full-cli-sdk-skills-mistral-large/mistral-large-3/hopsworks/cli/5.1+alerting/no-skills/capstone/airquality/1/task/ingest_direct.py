#!/usr/bin/env python3
"""
Directly ingest airquality_history.csv into the feature group `airq2408fa`.
"""

import hopsworks
import pandas as pd
import numpy as np

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Read the input data
df = pd.read_csv("data/airquality_history.csv", parse_dates=["date"])

# Compute rolling averages
df["pm25_rolling_3d"] = df["pm25"].rolling(3, min_periods=1).mean()
df["pm25_rolling_7d"] = df["pm25"].rolling(7, min_periods=1).mean()

# Fill NaN values (if any) with forward fill, then backward fill
df.fillna(method="ffill", inplace=True)
df.fillna(method="bfill", inplace=True)

# Get the feature group
fg = fs.get_feature_group("airq2408fa", version=1)

# Ingest the data using Spark to bypass delta issues
fg.insert(df, write_options={"wait_for_job": True, "use_spark": True})

print("Data ingested successfully into airq2408fa.")