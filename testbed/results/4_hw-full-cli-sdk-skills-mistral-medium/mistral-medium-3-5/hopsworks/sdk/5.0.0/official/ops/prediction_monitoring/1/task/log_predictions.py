#!/usr/bin/env python3
import hopsworks
import pandas as pd
import os

# Login to Hopsworks
hopsworks.login()

# Get the project
project = hopsworks.project()

# Read the prediction log
prediction_log_path = "data/prediction_log.csv"
df = pd.read_csv(prediction_log_path)

# Convert timestamp to datetime
df['ts'] = pd.to_datetime(df['ts'])

# Create a feature group for predictions
# First, check if it exists
try:
    fg = project.get_feature_group(name="prediction_monitoring_fg", version=1)
    print(f"Feature group already exists: {fg.name}")
except:
    # Create new feature group
    fg = project.create_feature_group(
        name="prediction_monitoring_fg",
        version=1,
        description="Feature group for monitoring model predictions"
    )
    print(f"Created feature group: {fg.name}")

# Log the predictions to the feature group
# We need to add the predictions as features
# Let's extract date from timestamp
df['date'] = df['ts'].dt.date
df['hour'] = df['ts'].dt.hour

# Prepare data for feature store
feature_data = df[['ts', 'prediction', 'date', 'hour']].copy()

# Insert into feature group
try:
    fg.insert(feature_data, write_options={"wait_for_job": True})
    print("Successfully inserted predictions into feature group")
except Exception as e:
    print(f"Error inserting into feature group: {e}")
    # Try without wait_for_job
    try:
        fg.insert(feature_data)
        print("Inserted predictions into feature group (async)")
    except Exception as e2:
        print(f"Second error: {e2}")

# Now let's also create a monitoring setup
# Check what monitoring capabilities are available
print("\nAvailable project methods:")
print([m for m in dir(project) if not m.startswith('_')])

# Try to access model registry
try:
    mr = project.get_model_registry()
    print("\nModel registry available")
    print([m for m in dir(mr) if not m.startswith('_')])
except Exception as e:
    print(f"Model registry error: {e}")

# Try to create a monitoring job or use statistics
try:
    # Get feature view
    fv = project.get_feature_view(name="prediction_monitoring_fv", version=1)
    print(f"Feature view exists: {fv.name}")
except:
    try:
        fv = project.create_feature_view(
            name="prediction_monitoring_fv",
            version=1,
            query=fg.select_all()
        )
        print(f"Created feature view: {fv.name}")
    except Exception as e:
        print(f"Error creating feature view: {e}")

# Let's also try to get statistics on the prediction column
print("\nComputing statistics on predictions...")
print(f"Mean prediction: {df['prediction'].mean()}")
print(f"Std prediction: {df['prediction'].std()}")
print(f"Min prediction: {df['prediction'].min()}")
print(f"Max prediction: {df['prediction'].max()}")

# Find the shift point by analyzing daily statistics
df['date'] = pd.to_datetime(df['ts']).dt.date
daily_stats = df.groupby('date')['prediction'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()

print("\nDaily statistics:")
print(daily_stats.head(10))

# Find the onset date where mean jumps significantly
# Calculate rolling mean
daily_stats['rolling_mean'] = daily_stats['mean'].rolling(window=3, center=False).mean()
daily_stats['mean_diff'] = daily_stats['mean'].diff().abs()

# Find the largest jump
max_jump_idx = daily_stats['mean_diff'].idxmax()
onset_date = daily_stats.loc[max_jump_idx, 'date']

print(f"\nLargest mean jump at: {onset_date}")
print(f"Jump size: {daily_stats.loc[max_jump_idx, 'mean_diff']}")

# Also check around that date
print("\nAround the shift:")
for i in range(max(0, max_jump_idx-2), min(len(daily_stats), max_jump_idx+3)):
    row = daily_stats.iloc[i]
    print(f"  {row['date']}: mean={row['mean']:.4f}, std={row['std']:.4f}, count={row['count']}")

# Write the answer
import json
os.makedirs('submission', exist_ok=True)
with open('submission/answers.json', 'w') as f:
    json.dump({"onset": str(onset_date)}, f)

print(f"\nAnswer written to submission/answers.json: {onset_date}")
