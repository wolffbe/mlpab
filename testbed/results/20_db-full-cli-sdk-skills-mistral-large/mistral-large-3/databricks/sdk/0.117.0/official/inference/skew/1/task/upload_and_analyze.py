#!/usr/bin/env python3
"""
Uploads training and serving data to Unity Catalog, computes summary statistics,
and identifies the feature with training/serving skew.
"""

import os
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog

# Initialize WorkspaceClient
w = WorkspaceClient()

# Environment variables
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # workspace.<run-id>
catalog_name, schema = schema_name.split(".")
volume_name = f"{os.getenv('MLPAB_DATABRICKS_PREFIX')}_data_volume"

# Create volume
try:
    w.volumes.create(
        catalog_name=catalog_name,
        schema_name=schema,
        name=volume_name,
        volume_type=catalog.VolumeType.MANAGED
    )
    print(f"Created volume: {catalog_name}.{schema}.{volume_name}")
except Exception as e:
    print(f"Volume may already exist: {e}")

# Upload files to volume using dbutils
volume_path = f"/Volumes/{catalog_name}/{schema}/{volume_name}"
training_dbfs_path = f"/dbfs{volume_path}/training_sample.csv"
serving_dbfs_path = f"/dbfs{volume_path}/serving_log.csv"

# Read local files and write to DBFS
os.makedirs(f"/dbfs{volume_path}", exist_ok=True)
with open("data/training_sample.csv", "rb") as f:
    training_data = f.read()
with open("data/serving_log.csv", "rb") as f:
    serving_data = f.read()

with open(training_dbfs_path, "wb") as f:
    f.write(training_data)
with open(serving_dbfs_path, "wb") as f:
    f.write(serving_data)

print("Uploaded files to volume.")

# Read data into DataFrames
training_df = pd.read_csv(training_dbfs_path)
serving_df = pd.read_csv(serving_dbfs_path)

# Compute summary statistics for each feature
features = ["f1", "f2", "f3", "f4", "f5"]
stats = []

for feature in features:
    training_mean = training_df[feature].mean()
    training_std = training_df[feature].std()
    serving_mean = serving_df[feature].mean()
    serving_std = serving_df[feature].std()
    
    # Compute divergence (absolute difference in mean and std)
    mean_diff = abs(training_mean - serving_mean)
    std_diff = abs(training_std - serving_std)
    
    stats.append({
        "feature": feature,
        "training_mean": training_mean,
        "training_std": training_std,
        "serving_mean": serving_mean,
        "serving_std": serving_std,
        "mean_diff": mean_diff,
        "std_diff": std_diff
    })

# Identify the feature with the highest divergence
stats_df = pd.DataFrame(stats)
stats_df["combined_diff"] = stats_df["mean_diff"] + stats_df["std_diff"]
divergent_feature = stats_df.loc[stats_df["combined_diff"].idxmax()]["feature"]

# Write the result to submission/answers.json
result = {
    "feature": divergent_feature,
    "cause": "Significant divergence in mean and standard deviation between training and serving distributions."
}

os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    import json
    json.dump(result, f, indent=2)

print(f"Result written to submission/answers.json: {result}")