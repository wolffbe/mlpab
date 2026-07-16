#!/usr/bin/env python3
import hopsworks
import pandas as pd

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Read input
feature_history = pd.read_csv("/Projects/mlpab1e2ad4/Resources/feature_history.csv")

# Get feature group
fg = fs.get_feature_group("feature_history", version=1)

# Insert data
fg.insert(feature_history)