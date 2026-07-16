#!/usr/bin/env python3
import os
import json

# Set up Databricks SDK
import databricks.sdk

# Connect to workspace
w = databricks.sdk.WorkspaceClient()

# Get environment variables
SCHEMA = os.environ['MLPAB_DATABRICKS_SCHEMA']
PREFIX = os.environ['MLPAB_DATABRICKS_PREFIX']

print(f"Schema: {SCHEMA}")
print(f"Prefix: {PREFIX}")

# Get current user
current_user = w.current_user.me().user_name
print(f"Current user: {current_user}")

# Upload the CSV data to workspace filesystem
print("\nUploading data to workspace...")
workspace_path = f"/Users/{current_user}/{PREFIX}/features.csv"

# Read the CSV file
with open('data/features.csv', 'rb') as f:
    content = f.read()

# Upload to workspace
w.workspace.upload(path=workspace_path, content=content, overwrite=True)
print(f"Uploaded data to {workspace_path}")

# Use the workspace warehouse
warehouse_id = "4dfab06c923fe3cc"

# Create a temporary table from the CSV using the workspace catalog
print("\nCreating table from CSV...")

# Use the workspace catalog
catalog_name = "workspace"
schema_name = SCHEMA.split('.')[-1]  # mlpab7bda90

# Create table using SQL with workspace catalog
# We'll use the workspace path - but this might not work directly
# Let's try using the workspace path in the SQL
w.statement_execution.execute_statement(
    warehouse_id=warehouse_id,
    catalog=catalog_name,
    schema=schema_name,
    statement=f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.{PREFIX}_features (
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
        path "{workspace_path}",
        header "true",
        inferSchema "true"
    )
    """
)
print(f"Created table {SCHEMA}.{PREFIX}_features")

# Now analyze the data for drift
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
    FROM {SCHEMA}.{PREFIX}_features
    GROUP BY DATE(event_time)
    ORDER BY date
)
SELECT * FROM daily_stats
"""

result = w.statement_execution.execute_statement(
    warehouse_id=warehouse_id,
    catalog=catalog_name,
    schema=schema_name,
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
print("\nDetecting drift...")

# Extract dates and feature means
dates = [row[0] for row in results]
features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
feature_means = {}
for i, feat in enumerate(features):
    feature_means[feat] = [results[j][1 + i*2] for j in range(len(results))]

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

# Verify by looking at the actual distribution shift
print(f"\nVerifying drift for {max_feature} around {max_onset}...")

# Find the index of the onset date
onset_idx = dates.index(max_onset) if max_onset in dates else len(dates) // 2

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
