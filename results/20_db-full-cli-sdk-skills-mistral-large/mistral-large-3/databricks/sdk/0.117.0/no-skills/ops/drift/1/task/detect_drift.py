#!/usr/bin/env python3
"""
Detect feature drift by reading data/features.csv locally and computing statistics.
"""
import os
import json
import pandas as pd
from datetime import datetime

# Read the data
print("Reading data...")
df = pd.read_csv("data/features.csv")
df['event_time'] = pd.to_datetime(df['event_time'])
df['day'] = df['event_time'].dt.date

# Compute daily statistics
daily_stats = df.groupby('day').agg({
    'f1': 'mean',
    'f2': 'mean',
    'f3': 'mean',
    'f4': 'mean',
    'f5': 'mean',
    'f6': 'mean'
}).reset_index()

# Compute baseline (first day)
baseline = daily_stats.iloc[0]

# Compute relative change for each feature
daily_stats['f1_change'] = abs(daily_stats['f1'] - baseline['f1']) / abs(baseline['f1'])
daily_stats['f2_change'] = abs(daily_stats['f2'] - baseline['f2']) / abs(baseline['f2'])
daily_stats['f3_change'] = abs(daily_stats['f3'] - baseline['f3']) / abs(baseline['f3'])
daily_stats['f4_change'] = abs(daily_stats['f4'] - baseline['f4']) / abs(baseline['f4'])
daily_stats['f5_change'] = abs(daily_stats['f5'] - baseline['f5']) / abs(baseline['f5'])
daily_stats['f6_change'] = abs(daily_stats['f6'] - baseline['f6']) / abs(baseline['f6'])

# Find the feature with the largest change
drifted_feature = None
onset_date = None
max_change = 0.0

for _, row in daily_stats.iterrows():
    for feature in ["f1", "f2", "f3", "f4", "f5", "f6"]:
        change = row[f"{feature}_change"]
        if change > max_change:
            max_change = change
            drifted_feature = feature
            onset_date = str(row['day'])

# Write the result to submission/answers.json
if drifted_feature and onset_date:
    answer = {
        "feature": drifted_feature,
        "onset": onset_date
    }
    os.makedirs("submission", exist_ok=True)
    with open("submission/answers.json", "w") as f:
        json.dump(answer, f)
    print(f"Drift detected: {answer}")
else:
    print("No drift detected or insufficient data.")