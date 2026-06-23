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

# Read the prediction log
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
fg_name = "prediction_monitoring_fg"
try:
    fg = fs.get_feature_group(name=fg_name, version=1)
    print(f"Feature group already exists: {fg.name}")
except:
    fg = fs.create_feature_group(
        name=fg_name,
        version=1,
        description="Feature group for monitoring model predictions",
        online_enabled=False
    )
    print(f"Created feature group: {fg.name}")

# Prepare data for insertion
# We need to extract date from timestamp for partitioning
data_to_insert = []
for p in predictions:
    # Parse timestamp to get date
    ts = p['ts']
    # Extract date part (YYYY-MM-DD)
    date_part = ts.split('T')[0]
    data_to_insert.append({
        'ts': ts,
        'prediction': p['prediction'],
        'date': date_part
    })

# Insert data into feature group
try:
    fg.insert(data_to_insert, write_options={"wait_for_job": False})
    print(f"Inserted {len(data_to_insert)} rows into feature group")
except Exception as e:
    print(f"Error inserting data: {e}")
    # Try without write_options
    try:
        fg.insert(data_to_insert)
        print(f"Inserted {len(data_to_insert)} rows into feature group (async)")
    except Exception as e2:
        print(f"Second error: {e2}")

# Now let's try to use the platform's SQL capabilities to analyze the data
# First, let's check if we can query the feature group
try:
    # Get the feature view or query directly
    print("\nTrying to query feature group...")
    
    # Wait a bit for the data to be available
    import time
    time.sleep(5)
    
    # Try to get statistics using the feature store SQL
    # First, let's see what feature groups exist
    fgs = fs.get_feature_groups()
    print(f"Feature groups: {[fg.name for fg in fgs]}")
    
    # Try to create a feature view
    fv_name = "prediction_monitoring_fv"
    try:
        fv = fs.get_feature_view(name=fv_name, version=1)
        print(f"Feature view exists: {fv.name}")
    except:
        try:
            fv = fs.create_feature_view(
                name=fv_name,
                version=1,
                query=fg.select_all()
            )
            print(f"Created feature view: {fv.name}")
        except Exception as e:
            print(f"Error creating feature view: {e}")
    
    # Try to get training dataset to compute statistics
    print("\nTrying to create training dataset...")
    try:
        td = fs.create_training_dataset(
            name="prediction_analysis_td",
            version=1,
            data_format="csv",
            description="Training dataset for prediction analysis"
        )
        print(f"Created training dataset: {td.name}")
    except Exception as e:
        print(f"Error creating training dataset: {e}")
        
except Exception as e:
    print(f"Error in analysis: {e}")

# Since we need to identify the shift, let's do a simple analysis
# Group by date and compute mean
from collections import defaultdict
daily_sums = defaultdict(float)
daily_counts = defaultdict(int)
daily_dates = []

for p in predictions:
    date_part = p['ts'].split('T')[0]
    if date_part not in daily_dates:
        daily_dates.append(date_part)
    daily_sums[date_part] += p['prediction']
    daily_counts[date_part] += 1

# Compute daily means
daily_means = []
for date in sorted(daily_dates):
    mean_val = daily_sums[date] / daily_counts[date]
    daily_means.append((date, mean_val))

print("\nDaily means:")
for date, mean_val in daily_means[:10]:
    print(f"  {date}: {mean_val:.4f}")

# Find the largest jump in means
max_jump = 0
onset_date = None
for i in range(1, len(daily_means)):
    jump = abs(daily_means[i][1] - daily_means[i-1][1])
    if jump > max_jump:
        max_jump = jump
        onset_date = daily_means[i][0]

print(f"\nLargest jump: {max_jump:.4f} at {onset_date}")

# Also check the means around the jump
print("\nAround the shift:")
for i in range(max(0, daily_means.index((onset_date, daily_sums[onset_date]/daily_counts[onset_date])) - 2), 
               min(len(daily_means), daily_means.index((onset_date, daily_sums[onset_date]/daily_counts[onset_date])) + 3)):
    date, mean_val = daily_means[i]
    print(f"  {date}: {mean_val:.4f}")

# Write the answer
os.makedirs('submission', exist_ok=True)
with open('submission/answers.json', 'w') as f:
    json.dump({"onset": onset_date}, f)

print(f"\nAnswer written to submission/answers.json: {onset_date}")
