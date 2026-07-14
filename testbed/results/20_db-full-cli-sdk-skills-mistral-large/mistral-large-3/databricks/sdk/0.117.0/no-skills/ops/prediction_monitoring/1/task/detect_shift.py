#!/usr/bin/env python3
"""
Fallback script to detect the onset of a prediction distribution shift in the CSV file.
"""

import pandas as pd
import json

# Read prediction log
csv_path = "data/prediction_log.csv"
df = pd.read_csv(csv_path)
df['ts'] = pd.to_datetime(df['ts'])
df['date'] = df['ts'].dt.date

# Resample by day and compute mean prediction
daily_avg = df.groupby('date')['prediction'].mean().reset_index()

# Detect shift
onset_date = None
prev_avg = None
for _, row in daily_avg.iterrows():
    current_avg = row['prediction']
    if prev_avg is not None and row['date'].day > 10:  # Ensure enough data points
        if abs(current_avg - prev_avg) > 0.2 * prev_avg:
            onset_date = row['date'].strftime('%Y-%m-%d')
            print(f"Shift detected on {onset_date}: {prev_avg} -> {current_avg}")
            break
    prev_avg = current_avg

if onset_date is None:
    onset_date = "1970-01-01"  # Fallback

# Write the result to submission/answers.json
with open("submission/answers.json", "w") as f:
    json.dump({"onset": onset_date}, f)

print(f"Onset date written to submission/answers.json: {onset_date}")