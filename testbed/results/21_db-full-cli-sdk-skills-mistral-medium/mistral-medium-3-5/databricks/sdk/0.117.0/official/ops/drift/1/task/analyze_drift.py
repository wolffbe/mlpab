#!/usr/bin/env python3
import os
import json
import tempfile
from datetime import datetime

# Set up Databricks SDK
import databricks.sdk
from databricks.sdk.service.catalog import *

# Connect to workspace
w = databricks.sdk.WorkspaceClient()

# Get environment variables
SCHEMA = os.environ['MLPAB_DATABRICKS_SCHEMA']
PREFIX = os.environ['MLPAB_DATABRICKS_PREFIX']

# Create a catalog name from schema (workspace.mlpab7bda90 -> mlpab7bda90)
catalog_name = SCHEMA.split('.')[-1]

print(f"Schema: {SCHEMA}")
print(f"Catalog: {catalog_name}")
print(f"Prefix: {PREFIX}")

# First, let's check if the catalog exists
try:
    catalog = w.catalogs.get(catalog_name)
    print(f"Catalog {catalog_name} exists")
except Exception as e:
    print(f"Catalog {catalog_name} doesn't exist, creating...")
    # Create catalog if it doesn't exist
    w.catalogs.create(name=catalog_name, comment="MLPAB drift detection catalog")
    print(f"Created catalog {catalog_name}")

# Check if schema exists
try:
    schema = w.schemas.get(full_name=SCHEMA)
    print(f"Schema {SCHEMA} exists")
except Exception as e:
    print(f"Schema {SCHEMA} doesn't exist, creating...")
    w.schemas.create(name=SCHEMA, catalog_name=catalog_name, comment="MLPAB drift detection schema")
    print(f"Created schema {SCHEMA}")

# Upload the CSV data to DBFS
print("\nUploading data to DBFS...")
dbfs_path = f"/FileStore/tables/{PREFIX}_features.csv"
with open('data/features.csv', 'r') as f:
    content = f.read()

# Write to DBFS
with w.dbfs.upload(dbfs_path, overwrite=True) as f:
    f.write(content.encode('utf-8'))
print(f"Uploaded data to {dbfs_path}")

# Create a table from the CSV
print("\nCreating table from CSV...")
table_name = f"{PREFIX}_features"
full_table_name = f"{SCHEMA}.{table_name}"

# Create table using SQL
w.statement_execution.execute_statement(
    warehouse_id="0d82456b41c93327",
    catalog=catalog_name,
    schema=SCHEMA.split('.')[-1],
    statement=f"""
    CREATE TABLE IF NOT EXISTS {full_table_name} (
        entity_id STRING,
        event_time TIMESTAMP,
        f1 DOUBLE,
        f2 DOUBLE,
        f3 DOUBLE,
        f4 DOUBLE,
        f5 DOUBLE,
        f6 DOUBLE
    )
    USING CSV
    OPTIONS (
        path "{dbfs_path}",
        header "true",
        inferSchema "true"
    )
    """
)
print(f"Created table {full_table_name}")

# Now analyze the data for drift
# We'll compute statistics for each feature over time
print("\nAnalyzing feature distributions over time...")

# Query to get daily statistics for each feature
query = f"""
WITH daily_stats AS (
    SELECT 
        DATE(event_time) as date,
        AVG(f1) as avg_f1, STDDEV(f1) as std_f1,
        AVG(f2) as avg_f2, STDDEV(f2) as std_f2,
        AVG(f3) as avg_f3, STDDEV(f3) as std_f3,
        AVG(f4) as avg_f4, STDDEV(f4) as std_f4,
        AVG(f5) as avg_f5, STDDEV(f5) as std_f5,
        AVG(f6) as avg_f6, STDDEV(f6) as std_f6,
        COUNT(*) as count
    FROM {full_table_name}
    GROUP BY DATE(event_time)
    ORDER BY date
)
SELECT * FROM daily_stats
"""

result = w.statement_execution.execute_statement(
    warehouse_id="0d82456b41c93327",
    catalog=catalog_name,
    schema=SCHEMA.split('.')[-1],
    statement=query
)

# Get the results
results = []
for row in result.result.data_array:
    results.append(row)

print(f"\nRetrieved {len(results)} days of statistics")
print("\nFirst few days:")
for i, row in enumerate(results[:5]):
    print(f"  {row[0]}: f1={row[1]:.3f}, f2={row[3]:.3f}, f3={row[5]:.3f}, f4={row[7]:.3f}, f5={row[9]:.3f}, f6={row[11]:.3f}")

print("\nLast few days:")
for i, row in enumerate(results[-5:]):
    print(f"  {row[0]}: f1={row[1]:.3f}, f2={row[3]:.3f}, f3={row[5]:.3f}, f4={row[7]:.3f}, f5={row[9]:.3f}, f6={row[11]:.3f}")

# Now let's detect drift by looking for significant changes in mean
# We'll use a simple approach: find the feature with the largest relative change
print("\nDetecting drift...")

# Extract dates and feature means
dates = [row[0] for row in results]
features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
feature_means = {}
for i, feat in enumerate(features):
    feature_means[feat] = [results[j][1 + i*2] for j in range(len(results))]

# Find the drift point for each feature
# We'll look for the point where the mean changes the most
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
    onset, change = detect_drift(feature_means[feat], dates)
    feature_drifts[feat] = {
        'onset': onset,
        'change': change
    }
    print(f"  {feat}: onset={onset}, change={change:.4f}")

# Find the feature with the largest change
max_feature = max(feature_drifts.keys(), key=lambda x: feature_drifts[x]['change'])
max_onset = feature_drifts[max_feature]['onset']
max_change = feature_drifts[max_feature]['change']

print(f"\nFeature with largest drift: {max_feature}")
print(f"Onset date: {max_onset}")
print(f"Change magnitude: {max_change:.4f}")

# Let's also verify by looking at the actual distribution shift
# Query to get percentiles before and after the drift
print(f"\nVerifying drift for {max_feature} around {max_onset}...")

# Find the index of the onset date
onset_idx = dates.index(max_onset) if max_onset in dates else len(dates) // 2

# Get statistics before and after
before_date = dates[onset_idx - 10] if onset_idx >= 10 else dates[0]
after_date = dates[onset_idx + 10] if onset_idx + 10 < len(dates) else dates[-1]

print(f"Before {max_onset}: mean={feature_means[max_feature][onset_idx-1]:.4f}")
print(f"After {max_onset}: mean={feature_means[max_feature][onset_idx]:.4f}")

# Let's also check the other features to make sure only one drifted
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
