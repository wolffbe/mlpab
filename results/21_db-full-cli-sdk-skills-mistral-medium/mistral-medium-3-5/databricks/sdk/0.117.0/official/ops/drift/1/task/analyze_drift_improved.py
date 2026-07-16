#!/usr/bin/env python3
import os
import json
import csv
from datetime import datetime
from collections import defaultdict

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

# Print daily means for f5 to see the transition
print(f"\nDaily means for f5:")
for date in sorted_dates:
    print(f"  {date.isoformat()}: {daily_means[date]['f5']:.3f}")

# Now let's detect the exact transition point for each feature
# We'll look for the point where the mean changes the most from one day to the next
print("\nDetecting exact transition points...")

feature_transitions = {}
for feat in features:
    means = [daily_means[date][feat] for date in sorted_dates]
    
    # Find the point with maximum absolute change from one day to the next
    max_change = 0
    max_change_idx = 0
    for i in range(1, len(means)):
        change = abs(means[i] - means[i-1])
        if change > max_change:
            max_change = change
            max_change_idx = i
    
    transition_date = sorted_dates[max_change_idx]
    feature_transitions[feat] = {
        'date': transition_date,
        'change': max_change,
        'before': means[max_change_idx-1],
        'after': means[max_change_idx]
    }
    print(f"  {feat}: transition at {transition_date.isoformat()}, change={max_change:.4f}, before={means[max_change_idx-1]:.4f}, after={means[max_change_idx]:.4f}")

# Find the feature with the largest change
max_feature = max(feature_transitions.keys(), key=lambda x: feature_transitions[x]['change'])
max_date = feature_transitions[max_feature]['date']
max_change = feature_transitions[max_feature]['change']

print(f"\nFeature with largest transition: {max_feature}")
print(f"Transition date: {max_date.isoformat()}")
print(f"Change magnitude: {max_change:.4f}")

# Verify by looking at all features
print(f"\nAll features transitions:")
for feat in features:
    trans = feature_transitions[feat]
    print(f"  {feat}: {trans['date'].isoformat()}, change={trans['change']:.4f}, before={trans['before']:.4f}, after={trans['after']:.4f}")

# Create submission directory
os.makedirs('submission', exist_ok=True)

# Write the answer
answer = {
    "feature": max_feature,
    "onset": max_date.isoformat()
}

with open('submission/answers.json', 'w') as f:
    json.dump(answer, f)

print(f"\nAnswer written to submission/answers.json: {answer}")
