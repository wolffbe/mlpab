#!/usr/bin/env python3

import hopsworks
import pandas as pd
import numpy as np
from math import sqrt

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Read the data files
requests_df = pd.read_csv('data/requests.csv')
profiles_df = pd.read_csv('data/profiles.csv')

# Merge requests with profiles on account_id
merged_df = pd.merge(requests_df, profiles_df, on='account_id', how='left')

# Calculate distance_deg and score
def calculate_distance(row):
    lat_diff = row['request_lat'] - row['home_lat']
    lon_diff = row['request_lon'] - row['home_lon']
    distance = sqrt(lat_diff**2 + lon_diff**2)
    return round(distance, 6)

merged_df['distance_deg'] = merged_df.apply(calculate_distance, axis=1)
merged_df['score'] = round(merged_df['base_score'] - 0.1 * merged_df['distance_deg'], 6)

# Select only the required columns
result_df = merged_df[['request_id', 'account_id', 'distance_deg', 'score']]

# Create the feature group
fg = fs.create_feature_group(
    name='scored9c0a6e',
    version=1,
    description='Feature table with request scoring based on distance from home location',
    primary_key=['request_id'],
    online_enabled=True,  # Enable online access for low-latency lookup
    statistics_config=False  # Disable statistics to speed up the process
)

# Insert the data into the feature group
fg.insert(result_df, wait=True)

print("Feature group 'scored9c0a6e', version 1 created successfully with online access enabled.")