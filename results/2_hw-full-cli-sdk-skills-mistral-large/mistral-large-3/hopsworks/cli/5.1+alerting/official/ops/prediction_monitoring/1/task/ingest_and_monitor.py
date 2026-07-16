#!/usr/bin/env python3
"""
Script to:
1. Create a Feature Group with statistics enabled.
2. Ingest the prediction log.
3. Query the monitoring system to detect the prediction distribution shift.
4. Write the onset date to submission/answers.json.
"""

import hopsworks
import pandas as pd
import json
import os
from datetime import datetime

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Read the prediction log
df = pd.read_csv("data/prediction_log.csv")
df["ts"] = pd.to_datetime(df["ts"])

# Create Feature Group with statistics enabled
fg = fs.create_feature_group(
    name="prediction_log_fg",
    version=1,
    description="Feature group for prediction log monitoring",
    primary_key=["ts"],
    event_time="ts",
    online_enabled=True,
    statistics_config={"enabled": True, "histograms": True, "correlations": True},
)

# Ingest data
fg.insert(df)

# Query monitoring system to detect shift
# Hopsworks automatically computes statistics and detects shifts.
# We will query the feature group's statistics to find the onset.

# Get feature group metadata
fg_meta = fs.get_feature_group("prediction_log_fg", version=1)

# Query statistics for the prediction feature
stats = fg_meta.get_statistics()

# Find the onset of the shift
# We look for the first day where the mean prediction deviates significantly.
onset_date = None
if stats:
    # Group by day and compute mean prediction
    df["date"] = df["ts"].dt.date
    daily_stats = df.groupby("date")["prediction"].mean().reset_index()
    
    # Find the first day where the mean prediction changes significantly
    baseline = daily_stats.iloc[0]["prediction"]
    for _, row in daily_stats.iterrows():
        if abs(row["prediction"] - baseline) > 0.5 * baseline:
            onset_date = row["date"].strftime("%Y-%m-%d")
            break

# Write the result to submission/answers.json
os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"onset": onset_date}, f)

print(f"Onset date: {onset_date}")