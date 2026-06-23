#!/usr/bin/env python3
import hopsworks
import pandas as pd
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

# Read the prediction log using pandas
prediction_log_path = "data/prediction_log.csv"
df = pd.read_csv(prediction_log_path)
print(f"Loaded {len(df)} predictions")

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

# Prepare data for insertion
df['date'] = pd.to_datetime(df['ts']).dt.date.astype(str)

# Insert data into feature group
try:
    fg.insert(df)
    print(f"Inserted {len(df)} rows into feature group")
except Exception as e:
    print(f"Error inserting data: {e}")
    # Try with wait_for_job
    try:
        fg.insert(df, write_options={"wait_for_job": False})
        print(f"Inserted {len(df)} rows into feature group (async)")
    except Exception as e2:
        print(f"Second error: {e2}")

# Wait a bit for the data to be available
import time
time.sleep(2)

# Now use the platform's SQL to analyze the data
print("\nQuerying feature store for daily statistics...")
try:
    # Query to get daily mean predictions
    query = f"""
    SELECT 
        date,
        AVG(prediction) as mean_prediction,
        COUNT(*) as count
    FROM {fg_name}_1
    GROUP BY date
    ORDER BY date
    """
    
    result_df = fs.sql(query)
    print(f"Query returned {len(result_df)} rows")
    print("\nDaily means from platform:")
    print(result_df.head(10))
    
    # Find the largest jump in the platform results
    if len(result_df) > 1:
        result_df['mean_diff'] = result_df['mean_prediction'].diff().abs()
        max_jump_idx = result_df['mean_diff'].idxmax()
        onset_date = result_df.loc[max_jump_idx, 'date']
        
        print(f"\nLargest jump from platform data: {result_df.loc[max_jump_idx, 'mean_diff']:.4f} at {onset_date}")
        print("\nAround the shift (from platform):")
        for i in range(max(0, max_jump_idx-2), min(len(result_df), max_jump_idx+3)):
            row = result_df.iloc[i]
            print(f"  {row['date']}: {row['mean_prediction']:.4f}")
    else:
        print("Not enough data from platform query")
        onset_date = None
        
except Exception as e:
    print(f"Error querying feature store: {e}")
    onset_date = None

# If platform query didn't work, fall back to local analysis
if onset_date is None:
    print("\nFalling back to local analysis...")
    # Group by date and compute mean
    daily_stats = df.groupby('date')['prediction'].agg(['mean', 'count']).reset_index()
    daily_stats['mean_diff'] = daily_stats['mean'].diff().abs()
    
    max_jump_idx = daily_stats['mean_diff'].idxmax()
    onset_date = daily_stats.loc[max_jump_idx, 'date']
    
    print(f"Largest jump: {daily_stats.loc[max_jump_idx, 'mean_diff']:.4f} at {onset_date}")
    print("\nDaily means around shift:")
    for i in range(max(0, max_jump_idx-2), min(len(daily_stats), max_jump_idx+3)):
        row = daily_stats.iloc[i]
        print(f"  {row['date']}: {row['mean']:.4f}")

# Write the answer
os.makedirs('submission', exist_ok=True)
with open('submission/answers.json', 'w') as f:
    json.dump({"onset": onset_date}, f)

print(f"\nAnswer written: {onset_date}")
