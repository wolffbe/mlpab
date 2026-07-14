#!/usr/bin/env python3
"""
Investigate feature drift in the provided dataset using local pandas operations.
"""
import os
import json
import pandas as pd

# Environment variables
CATALOG_SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")
PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")

# Step 1: Compute daily statistics locally using pandas
def compute_daily_stats():
    # Read the local CSV file
    df = pd.read_csv("data/features.csv")
    df['event_time'] = pd.to_datetime(df['event_time']).dt.date
    
    # Compute daily statistics
    stats_df = df.groupby('event_time').agg({
        'f1': ['mean', 'std'],
        'f2': ['mean', 'std'],
        'f3': ['mean', 'std'],
        'f4': ['mean', 'std'],
        'f5': ['mean', 'std'],
        'f6': ['mean', 'std']
    }).reset_index()
    
    # Flatten the MultiIndex columns
    stats_df.columns = ['event_time', 'avg_f1', 'std_f1', 'avg_f2', 'std_f2', 'avg_f3', 'std_f3', 'avg_f4', 'std_f4', 'avg_f5', 'std_f5', 'avg_f6', 'std_f6']
    
    return stats_df

# Step 2: Detect drift by identifying sudden changes in statistics
def detect_drift(stats_df):
    features = ["f1", "f2", "f3", "f4", "f5", "f6"]
    drift_info = {}
    
    for feature in features:
        avg_col = f"avg_{feature}"
        std_col = f"std_{feature}"
        
        # Calculate rolling mean and std to smooth out noise
        stats_df[f"rolling_avg_{feature}"] = stats_df[avg_col].rolling(window=3).mean()
        stats_df[f"rolling_std_{feature}"] = stats_df[std_col].rolling(window=3).mean()
        
        # Calculate the difference between consecutive days
        stats_df[f"diff_{feature}"] = stats_df[f"rolling_avg_{feature}"].diff().abs()
        
        # Identify the day with the maximum difference
        max_diff_idx = stats_df[f"diff_{feature}"].idxmax()
        max_diff = stats_df[f"diff_{feature}"].max()
        
        # Store the drift info if the difference is significant
        if max_diff > 0.5 * stats_df[f"rolling_std_{feature}"].mean():
            drift_info[feature] = {
                "onset": stats_df.loc[max_diff_idx, "event_time"].strftime("%Y-%m-%d"),
                "diff": max_diff
            }
    
    # Identify the feature with the largest drift
    if drift_info:
        drifted_feature = max(drift_info.items(), key=lambda x: x[1]["diff"])
        return {
            "feature": drifted_feature[0],
            "onset": drifted_feature[1]["onset"]
        }
    else:
        return None

# Step 3: Write the result to submission/answers.json
def write_result(result):
    with open("submission/answers.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    stats_df = compute_daily_stats()
    result = detect_drift(stats_df)
    
    if result:
        print(f"Detected drift in feature: {result['feature']} on {result['onset']}")
        write_result(result)
    else:
        print("No drift detected.")