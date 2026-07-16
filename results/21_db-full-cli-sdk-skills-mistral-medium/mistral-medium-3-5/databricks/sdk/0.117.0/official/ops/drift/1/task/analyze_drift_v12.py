#!/usr/bin/env python3
import os
import json
import base64

# Set up Databricks SDK
import databricks.sdk
from databricks.sdk.service.workspace import ImportFormat

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

# Upload the CSV data to workspace filesystem using import_ with RAW format
print("\nUploading data to workspace...")
workspace_path = f"/Users/{current_user}/{PREFIX}/features.csv"

# Read the CSV file
with open('data/features.csv', 'rb') as f:
    content = f.read()

# Encode as base64
content_b64 = base64.b64encode(content).decode('utf-8')

# Import to workspace with RAW format
w.workspace.import_(path=workspace_path, content=content_b64, format=ImportFormat.RAW, overwrite=True)
print(f"Imported data to {workspace_path}")

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

# Now let's detect the exact transition point for each feature
print("\nDetecting exact transition points...")

# Extract dates and feature means
dates = [row[0] for row in results]
features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
feature_means = {}
for i, feat in enumerate(features):
    feature_means[feat] = [results[j][1 + i*2] for j in range(len(results))]

# Find the point with maximum absolute change from one day to the next
feature_transitions = {}
for feat in features:
    means = feature_means[feat]
    
    max_change = 0
    max_change_idx = 0
    for i in range(1, len(means)):
        change = abs(means[i] - means[i-1])
        if change > max_change:
            max_change = change
            max_change_idx = i
    
    transition_date = dates[max_change_idx]
    feature_transitions[feat] = {
        'date': transition_date,
        'change': max_change,
        'before': means[max_change_idx-1],
        'after': means[max_change_idx]
    }
    print(f"  {feat}: transition at {transition_date}, change={max_change:.4f}, before={means[max_change_idx-1]:.4f}, after={means[max_change_idx]:.4f}")

# Find the feature with the largest change
max_feature = max(feature_transitions.keys(), key=lambda x: feature_transitions[x]['change'])
max_date = feature_transitions[max_feature]['date']
max_change = feature_transitions[max_feature]['change']

print(f"\nFeature with largest transition: {max_feature}")
print(f"Transition date: {max_date}")
print(f"Change magnitude: {max_change:.4f}")

# Create submission directory
os.makedirs('submission', exist_ok=True)

# Write the answer
answer = {
    "feature": max_feature,
    "onset": max_date
}

with open('submission/answers.json', 'w') as f:
    json.dump(answer, f)

print(f"\nAnswer written to submission/answers.json: {answer}")
