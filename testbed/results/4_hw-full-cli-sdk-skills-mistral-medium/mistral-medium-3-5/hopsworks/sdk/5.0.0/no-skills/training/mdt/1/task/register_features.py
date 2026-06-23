#!/usr/bin/env python3

import hopsworks
import pandas as pd
import numpy as np

# Read the data
train_df = pd.read_csv('data/features_train.csv')
serve_df = pd.read_csv('data/features_serve.csv')

# Compute mean and population std from training data only
# Population std uses ddof=0 (no Bessel's correction)
features = ['f1', 'f2', 'f3', 'f4']
means = train_df[features].mean()
stds = train_df[features].std(ddof=0)

print("Training means:", means.to_dict())
print("Training stds (population):", stds.to_dict())

# Standardize both splits using training statistics
def standardize(df, split_name):
    df_std = df.copy()
    for feature in features:
        df_std[feature] = (df[feature] - means[feature]) / stds[feature]
    df_std['split'] = split_name
    return df_std

train_std = standardize(train_df, 'train')
serve_std = standardize(serve_df, 'serve')

# Round to 6 decimals
for feature in features:
    train_std[feature] = train_std[feature].round(6)
    serve_std[feature] = serve_std[feature].round(6)

# Combine both splits
combined = pd.concat([train_std, serve_std], ignore_index=True)

# Reorder columns: row_id, split, f1, f2, f3, f4
combined = combined[['row_id', 'split', 'f1', 'f2', 'f3', 'f4']]

print(f"\nCombined shape: {combined.shape}")
print(f"Combined columns: {combined.columns.tolist()}")
print(f"\nFirst few rows:")
print(combined.head())
print(f"\nLast few rows:")
print(combined.tail())

# Connect to Hopsworks and register the feature table
print("\nConnecting to Hopsworks...")
hopsworks.login()

# Get the feature store
project = hopsworks.get_current_project()
fs = project.get_feature_store()

# Create or get the feature group
fg_name = "scalede10147"
fg_version = 1

# Check if feature group exists, if not create it
fg = fs.get_feature_group(fg_name, version=fg_version)
if fg is None:
    # Create the feature group
    fg = fs.create_feature_group(
        name=fg_name,
        version=fg_version,
        description="Standardized features using training statistics",
        primary_key=["row_id"],
        online_enabled=True  # Enable for low-latency lookup
    )
    print(f"Created feature group {fg_name} v{fg_version}")
else:
    print(f"Feature group {fg_name} v{fg_version} already exists")

# Insert the data into the feature group
print(f"\nInserting {len(combined)} rows into feature group...")
fg.insert(combined, write_options={"wait_for_job": True})

print(f"\nFeature table {fg_name} v{fg_version} registered successfully!")
print("Features are available for low-latency lookup (online/real-time access).")
