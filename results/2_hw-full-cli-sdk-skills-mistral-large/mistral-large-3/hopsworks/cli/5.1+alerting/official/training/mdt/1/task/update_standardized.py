#!/usr/bin/env python3
"""
Standardize features in-place for scaled21081b using precomputed statistics.
"""

import hopsworks
import pandas as pd

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Load raw data
fg = fs.get_feature_group("scaled21081b", version=1)
df = fg.read()

# Precomputed statistics
stats = {
    "f1": {"mean": 2.4247, "std": 3.00145},
    "f2": {"mean": -3.26328, "std": 1.34485},
    "f3": {"mean": -0.0300147, "std": 1.41874},
    "f4": {"mean": 1.0404, "std": 2.64138}
}

# Standardize features
for col in ["f1", "f2", "f3", "f4"]:
    df[col] = ((df[col] - stats[col]["mean"]) / stats[col]["std"]).round(6)

# Update feature group in-place
fg.insert(df, overwrite=True, write_options={"wait_for_job": True})

print("Standardization complete. Updated rows:", len(df))