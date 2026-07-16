#!/usr/bin/env python3

import hopsworks
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Read the feature data
feature_df = pd.read_csv("./data/features.csv")
feature_df["event_time"] = pd.to_datetime(feature_df["event_time"]).dt.tz_localize(None)

# Create or get a feature group
feature_group = fs.get_or_create_feature_group(
    name="feature_drift_detection",
    version=1,
    description="Feature group for drift detection task",
    primary_key=["entity_id", "event_time"],
    event_time="event_time",
    online_enabled=False,
)

# Ingest data into the feature group
feature_group.insert(feature_df, write_options={"wait_for_job": True})

# Compute statistics over time windows to detect drift
feature_names = ["f1", "f2", "f3", "f4", "f5", "f6"]
window_size = "7d"  # 7-day windows

# Fetch data in time-ordered chunks
start_date = feature_df["event_time"].min()
end_date = feature_df["event_time"].max()
current_date = start_date

stats_history = {}
for feature in feature_names:
    stats_history[feature] = []

# Fetch the entire dataset and filter locally
full_df = feature_group.select_all().read()

# Iterate over time windows and filter locally
while current_date <= end_date:
    window_end = current_date + timedelta(days=7)
    window_df = full_df[(full_df["event_time"] >= current_date) & (full_df["event_time"] < window_end)]
    
    if not window_df.empty:
        for feature in feature_names:
            stats = {
                "window_start": current_date,
                "window_end": window_end,
                "mean": window_df[feature].mean(),
                "std": window_df[feature].std(),
                "count": len(window_df),
            }
            stats_history[feature].append(stats)
    
    current_date = window_end

# Detect drift by comparing statistics between consecutive windows
drift_results = {}
for feature in feature_names:
    stats_list = stats_history[feature]
    if len(stats_list) < 2:
        continue
    
    drift_detected = False
    onset_date = None
    
    for i in range(1, len(stats_list)):
        prev_stats = stats_list[i-1]
        curr_stats = stats_list[i]
        
        # Simple drift detection: significant change in mean or std
        mean_change = abs(curr_stats["mean"] - prev_stats["mean"]) / (prev_stats["std"] + 1e-6)
        std_change = abs(curr_stats["std"] - prev_stats["std"]) / (prev_stats["std"] + 1e-6)
        
        if mean_change > 1.5 or std_change > 1.5:  # Lower threshold for drift
            drift_detected = True
            onset_date = curr_stats["window_start"].strftime("%Y-%m-%d")
            break
    
    if drift_detected:
        drift_results[feature] = onset_date

# Identify the drifted feature (only one is expected)
drifted_feature = None
onset_date = None
if drift_results:
    drifted_feature = next(iter(drift_results.keys()))
    onset_date = drift_results[drifted_feature]

# Write the result to submission/answers.json
import json
with open("./submission/answers.json", "w") as f:
    json.dump({"feature": drifted_feature, "onset": onset_date}, f)

print(f"Drift detected in feature: {drifted_feature}, onset: {onset_date}")