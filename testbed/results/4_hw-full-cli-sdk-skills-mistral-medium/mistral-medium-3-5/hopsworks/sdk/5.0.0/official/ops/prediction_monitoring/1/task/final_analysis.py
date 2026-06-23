#!/usr/bin/env python3
import hopsworks
import csv
import json
import os

# Login to Hopsworks
hopsworks.login()

# Get the current project
project = hopsworks.get_current_project()
print(f"Project: {project.name}")

# Get feature store
fs = project.get_feature_store()
print(f"Feature store: {fs.name}")

# Read the prediction log using csv (standard library)
prediction_log_path = "data/prediction_log.csv"
predictions = []
with open(prediction_log_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        predictions.append({
            'ts': row['ts'],
            'prediction': float(row['prediction'])
        })

print(f"Loaded {len(predictions)} predictions")

# Create or get feature group
fg_name = "prediction_log_fg"
try:
    fg = fs.get_feature_group(name=fg_name, version=1)
    print(f"Feature group already exists: {fg.name}")
except:
    fg = fs.create_feature_group(
        name=fg_name,
        version=1,
        description="Feature group for prediction logs",
        online_enabled=False
    )
    print(f"Created feature group: {fg.name}")

# Try to use pandas if available through hopsworks
try:
    import pandas as pd
    # Create dataframe
    df = pd.DataFrame(predictions)
    df['date'] = df['ts'].str.split('T').str[0]
    
    # Insert data into feature group
    try:
        fg.insert(df)
        print(f"Inserted {len(df)} rows into feature group")
    except Exception as e:
        print(f"Error inserting data: {e}")
except ImportError:
    print("Pandas not available, skipping feature group insertion")

# Now analyze the data to find the shift
# Group by date and compute mean
from collections import defaultdict
daily_sums = defaultdict(float)
daily_counts = defaultdict(int)
dates = []

for p in predictions:
    date_part = p['ts'].split('T')[0]
    if date_part not in dates:
        dates.append(date_part)
    daily_sums[date_part] += p['prediction']
    daily_counts[date_part] += 1

# Compute daily means
daily_means = {}
for date in dates:
    daily_means[date] = daily_sums[date] / daily_counts[date]

# Sort dates
sorted_dates = sorted(daily_means.keys())

# Find the largest jump
max_jump = 0
onset_date = None
for i in range(1, len(sorted_dates)):
    date1 = sorted_dates[i-1]
    date2 = sorted_dates[i]
    jump = abs(daily_means[date2] - daily_means[date1])
    if jump > max_jump:
        max_jump = jump
        onset_date = date2

print(f"\nLargest jump: {max_jump:.4f} at {onset_date}")
print("\nDaily means around shift:")
idx = sorted_dates.index(onset_date)
for i in range(max(0, idx-2), min(len(sorted_dates), idx+3)):
    print(f"  {sorted_dates[i]}: {daily_means[sorted_dates[i]]:.4f}")

# Write the answer
os.makedirs('submission', exist_ok=True)
with open('submission/answers.json', 'w') as f:
    json.dump({"onset": onset_date}, f)

print(f"\nAnswer written: {onset_date}")
