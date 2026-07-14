#!/usr/bin/env python3
import os
import json
import csv
from datetime import datetime

# Read the CSV file
print("Reading CSV file...")
with open('data/features.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Read {len(rows)} rows")

# Parse dates and extract features
dates = []
features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
feature_values = {feat: [] for feat in features}

for row in rows:
    date_str = row['event_time']
    date = datetime.fromisoformat(date_str.replace('Z', '+00:00')).date()
    dates.append(date)
    
    for feat in features:
        feature_values[feat].append(float(row[feat]))

print(f"Dates range: {min(dates)} to {max(dates)}")

# Group by date and compute daily statistics
from collections import defaultdict

daily_data = defaultdict(lambda: {feat: [] for feat in features})
for i, date in enumerate(dates):
    for feat in features:
        daily_data[date][feat].append(feature_values[feat][i])

# Compute daily means
daily_means = {}
for date in sorted(daily_data.keys()):
    daily_means[date] = {}
    for feat in features:
        values = daily_data[date][feat]
        daily_means[date][feat] = sum(values) / len(values)

# Convert to sorted list
sorted_dates = sorted(daily_means.keys())
date_strs = [date.isoformat() for date in sorted_dates]

print(f"\nFirst few days:")
for date in sorted_dates[:5]:
    means = daily_means[date]
    print(f"  {date.isoformat()}: f1={means['f1']:.3f}, f2={means['f2']:.3f}, f3={means['f3']:.3f}, f4={means['f4']:.3f}, f5={means['f5']:.3f}, f6={means['f6']:.3f}")

print(f"\nLast few days:")
for date in sorted_dates[-5:]:
    means = daily_means[date]
    print(f"  {date.isoformat()}: f1={means['f1']:.3f}, f2={means['f2']:.3f}, f3={means['f3']:.3f}, f4={means['f4']:.3f}, f5={means['f5']:.3f}, f6={means['f6']:.3f}")

# Now let's detect drift by looking for significant changes in mean
print("\nDetecting drift...")

# Extract feature means in order
feature_means = {}
for feat in features:
    feature_means[feat] = [daily_means[date][feat] for date in sorted_dates]

# Find the drift point for each feature using a simple change detection
def detect_drift(means, dates):
    """Detect drift by finding the point with maximum change in rolling mean"""
    if len(means) < 2:
        return None, 0
    
    # Calculate rolling mean with window of 7 days
    window = 7
    rolling_means = []
    for i in range(len(means)):
        start = max(0, i - window + 1)
        rolling_mean = sum(means[start:i+1]) / (i - start + 1)
        rolling_means.append(rolling_mean)
    
    # Find the point with maximum absolute change
    max_change = 0
    max_change_idx = 0
    for i in range(1, len(rolling_means)):
        change = abs(rolling_means[i] - rolling_means[i-1])
        if change > max_change:
            max_change = change
            max_change_idx = i
    
    return dates[max_change_idx], max_change

# Detect drift for each feature
feature_drifts = {}
for feat in features:
    onset, change = detect_drift(feature_means[feat], sorted_dates)
    feature_drifts[feat] = {
        'onset': onset.isoformat() if onset else None,
        'change': change
    }
    print(f"  {feat}: onset={onset.isoformat() if onset else None}, change={change:.4f}")

# Find the feature with the largest change
max_feature = max(feature_drifts.keys(), key=lambda x: feature_drifts[x]['change'])
max_onset = feature_drifts[max_feature]['onset']
max_change = feature_drifts[max_feature]['change']

print(f"\nFeature with largest drift: {max_feature}")
print(f"Onset date: {max_onset}")
print(f"Change magnitude: {max_change:.4f}")

# Verify by looking at the actual distribution shift
print(f"\nVerifying drift for {max_feature} around {max_onset}...")

# Find the index of the onset date
onset_date = datetime.fromisoformat(max_onset).date()
onset_idx = sorted_dates.index(onset_date) if onset_date in sorted_dates else len(sorted_dates) // 2

print(f"Before {max_onset}: mean={feature_means[max_feature][onset_idx-1]:.4f}")
print(f"After {max_onset}: mean={feature_means[max_feature][onset_idx]:.4f}")

# Check all features for significant changes
print("\nChecking all features for significant changes:")
for feat in features:
    before = feature_means[feat][onset_idx - 1] if onset_idx > 0 else feature_means[feat][0]
    after = feature_means[feat][onset_idx]
    change = abs(after - before)
    rel_change = change / abs(before) if before != 0 else float('inf')
    print(f"  {feat}: before={before:.4f}, after={after:.4f}, change={change:.4f}, rel_change={rel_change:.2%}")

# Create submission directory
os.makedirs('submission', exist_ok=True)

# Write the answer
answer = {
    "feature": max_feature,
    "onset": max_onset
}

with open('submission/answers.json', 'w') as f:
    json.dump(answer, f)

print(f"\nAnswer written to submission/answers.json: {answer}")
