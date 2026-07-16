#!/usr/bin/env python3
"""
Ingest feature profiles into Hopsworks and retrieve feature vectors for lookup keys.
"""

import hopsworks
import pandas as pd
import json
import os

# Load data
features_df = pd.read_csv("data/features.csv")
with open("data/lookup_keys.txt", "r") as f:
    lookup_keys = [line.strip() for line in f.readlines()]

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Retrieve existing feature group
feature_group = fs.get_feature_group(
    name="profiles1daac7",
    version=1
)

# Retrieve existing feature view
feature_view = fs.get_feature_view(
    name="profiles1daac7_fv",
    version=1
)

# Fetch feature vectors for lookup keys
vectors = {}
for account_id in lookup_keys:
    try:
        feature_vector = feature_view.get_feature_vector(
            entry={"account_id": account_id},
            return_type="pandas"
        )
        vectors[account_id] = [
            feature_vector["f1"].values[0],
            feature_vector["f2"].values[0],
            feature_vector["f3"].values[0],
            feature_vector["f4"].values[0]
        ]
    except Exception as e:
        print(f"Failed to fetch features for {account_id}: {e}")

# Write results
with open("submission/answers.json", "w") as f:
    json.dump({"vectors": vectors}, f, indent=2)

print("Done.")