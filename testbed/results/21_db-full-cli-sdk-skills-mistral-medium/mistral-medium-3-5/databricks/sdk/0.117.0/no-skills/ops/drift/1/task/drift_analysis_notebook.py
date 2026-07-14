# Databricks notebook source
# MAGIC %md
# MAGIC ## Drift Analysis Notebook
# MAGIC This notebook analyzes feature drift in the uploaded dataset.

# COMMAND ----------

# MAGIC %python
import csv
import json
from collections import defaultdict
from datetime import datetime

# Read the uploaded CSV file
# The file should be at /Workspace/Users/<username>/<prefix>/features.csv
# But we'll try to read it from the current working directory first

import os
print("Current working directory:", os.getcwd())
print("Files:", os.listdir())

# Try to find the features.csv file
features_file = None
for root, dirs, files in os.walk('.'):
    if 'features.csv' in files:
        features_file = os.path.join(root, 'features.csv')
        break

if not features_file:
    # Try in /dbfs
    import dbutils
    try:
        dbutils.fs.ls('/FileStore/tables/')
        features_file = '/dbfs/FileStore/tables/mlpabc6cfe1_features.csv'
    except:
        pass

print(f"Using features file: {features_file}")

# Read the CSV
features_data = []
with open(features_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        features_data.append(row)

print(f"Loaded {len(features_data)} rows")

# COMMAND ----------

# MAGIC %python
# Group by date and calculate statistics
from collections import defaultdict

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

print("Daily statistics calculated")

# COMMAND ----------

# MAGIC %python
# Analyze for drift
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

# Write the answer to a file
answer = {
    "feature": drifted_feature[0],
    "onset": onset_date
}

# Write to workspace
import json
with open('/Workspace/Users/' + username + '/mlpabc6cfe1/answers.json', 'w') as f:
    json.dump(answer, f)

print(f"Answer: {answer}")

dbutils.notebook.exit(answer)
