#!/usr/bin/env python3
"""
Standardize features (f1, f2, f3, f4) using training split statistics and insert into scaled21081b.
"""

import hopsworks
import pandas as pd
import numpy as np

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Load raw data
train_fg = fs.get_feature_group("temp_train_raw", version=1)
train_df = train_fg.read()

serve_fg = fs.get_feature_group("temp_serve_raw", version=1)
serve_df = serve_fg.read()

# Compute statistics (mean, std) for training split
stats = train_df.agg({
    "f1": ["mean", "std"],
    "f2": ["mean", "std"],
    "f3": ["mean", "std"],
    "f4": ["mean", "std"]
}).T

stats.columns = ["mean", "std"]
stats["std"] = stats["std"].fillna(1.0)  # Avoid division by zero

# Standardize both splits
def standardize(df, split_name):
    df = df.copy()
    for col in ["f1", "f2", "f3", "f4"]:
        df[col] = ((df[col] - stats.loc[col, "mean"]) / stats.loc[col, "std"]).round(6)
    df["split"] = split_name
    return df

standardized_train = standardize(train_df, "train")
standardized_serve = standardize(serve_df, "serve")

# Combine and insert into scaled21081b
standardized_df = pd.concat([standardized_train, standardized_serve], ignore_index=True)
scaled_fg = fs.get_feature_group("scaled21081b", version=1)
scaled_fg.insert(standardized_df, write_options={"wait_for_job": True})

print("Standardization complete. Inserted rows:", len(standardized_df))