#!/usr/bin/env python3
"""
Identifies the feature with training/serving skew by comparing summary statistics
between training_sample.csv and serving_log.csv.
"""

import os
import pandas as pd
import json

# Read data into DataFrames
training_df = pd.read_csv("data/training_sample.csv")
serving_df = pd.read_csv("data/serving_log.csv")

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
    json.dump(result, f, indent=2)

print(f"Result written to submission/answers.json: {result}")