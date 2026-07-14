#!/usr/bin/env python3
import csv
import json
from collections import defaultdict

# Read the CSV
features_data = []
with open('data/features.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        features_data.append(row)

print(f"Loaded {len(features_data)} rows")

# Group by date
daily_data = defaultdict(list)
for row in features_data:
    date_str = row['event_time'].split('T')[0]
    daily_data[date_str].append(row)

# Calculate daily statistics
features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
daily_stats = []
for date in sorted(daily_data.keys()):
    rows = daily_data[date]
    stats = {'date': date}
    for feature in features:
        values = [float(row[feature]) for row in rows]
        avg = sum(values) / len(values)
        variance = sum((x - avg) ** 2 for x in values) / len(values)
        std = variance ** 0.5
        stats[f'avg_{feature}'] = avg
        stats[f'std_{feature}'] = std
    stats['count'] = len(rows)
    daily_stats.append(stats)

print("\nDaily statistics (first 10 and last 10 days):")
for i, stats in enumerate(daily_stats):
    if i < 10 or i >= len(daily_stats) - 10:
        date = stats['date']
        print(f"{date}: f1={stats['avg_f1']:.2f}, f2={stats['avg_f2']:.2f}, f3={stats['avg_f3']:.2f}, f4={stats['avg_f4']:.2f}, f5={stats['avg_f5']:.2f}, f6={stats['avg_f6']:.2f}")

# Analyze for drift - look for significant changes in mean
print("\nAnalyzing for drift...")

feature_drift = {}
for feature in features:
    changes = []
    for i in range(1, len(daily_stats)):
        prev_avg = daily_stats[i-1][f'avg_{feature}']
        curr_avg = daily_stats[i][f'avg_{feature}']
        change = abs(curr_avg - prev_avg)
        relative_change = change / abs(prev_avg) if prev_avg != 0 else 0
        changes.append((daily_stats[i]['date'], change, relative_change))
    
    # Find the maximum change
    if changes:
        max_change = max(changes, key=lambda x: x[2])
        feature_drift[feature] = max_change
        print(f"{feature}: max relative change = {max_change[2]:.4f} on {max_change[0]} (absolute: {max_change[1]:.4f})")

# Find the feature with the most significant drift
drifted_feature = max(feature_drift.items(), key=lambda x: x[1][2])
onset_date = drifted_feature[1][0]

print(f"\nDrift detected: feature={drifted_feature[0]}, onset={onset_date}")

# Write the answer
answer = {
    "feature": drifted_feature[0],
    "onset": onset_date
}

with open('submission/answers.json', 'w') as f:
    json.dump(answer, f)

print(f"Answer written to submission/answers.json: {answer}")
