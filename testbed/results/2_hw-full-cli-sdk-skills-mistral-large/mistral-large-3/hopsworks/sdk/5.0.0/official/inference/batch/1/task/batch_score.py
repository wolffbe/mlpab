#!/usr/bin/env python3

import hopsworks
import pandas as pd
import numpy as np
from datetime import datetime

# Constants from the task
T = 1773496800000  # as-of timestamp in epoch milliseconds
FEATURE_TABLE_NAME = "scores770373"
FEATURE_TABLE_VERSION = 1

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Load and process data
def load_and_process_data():
    # Load feature history
    feature_history = pd.read_csv("data/feature_history.csv")
    
    # Convert event_time to numeric (in case it's read as string)
    feature_history['event_time'] = pd.to_numeric(feature_history['event_time'])
    
    # Filter to get the most recent revision at or before T for each account
    # First filter all records at or before T
    valid_records = feature_history[feature_history['event_time'] <= T]
    
    # Then get the most recent record for each account
    latest_records = valid_records.sort_values('event_time', ascending=False).groupby('account_id').first().reset_index()
    
    return latest_records

# Compute scores
def compute_scores(feature_data):
    # Load model weights
    import json
    with open("data/model.json", "r") as f:
        model = json.load(f)
    
    # Extract weights and bias
    w_f1 = model["weights"]["f1"]
    w_f2 = model["weights"]["f2"]
    w_f3 = model["weights"]["f3"]
    bias = model["bias"]
    
    # Compute linear combination
    linear_combination = (
        w_f1 * feature_data["f1"] + 
        w_f2 * feature_data["f2"] + 
        w_f3 * feature_data["f3"] + 
        bias
    )
    
    # Apply sigmoid and round to 6 decimal places
    scores = sigmoid(linear_combination).round(6)
    
    return scores

def main():
    # Connect to Hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    # Load and process feature data
    feature_data = load_and_process_data()
    
    # Compute scores
    feature_data["score"] = compute_scores(feature_data)
    
    # Prepare the final dataframe with only account_id and score
    scores_df = feature_data[["account_id", "score"]].copy()
    
    # Create feature group
    scores_fg = fs.create_feature_group(
        name=FEATURE_TABLE_NAME,
        version=FEATURE_TABLE_VERSION,
        description="Batch scores for accounts as of T=1773496800000",
        primary_key=["account_id"],
        online_enabled=True,  # Make available for online/real-time access
        statistics_config=False  # Disable statistics for this simple table
    )
    
    # Insert data into the feature group
    scores_fg.insert(scores_df, wait=True)
    
    print(f"Successfully created feature table '{FEATURE_TABLE_NAME}', version {FEATURE_TABLE_VERSION}")
    print(f"Table is available for online/real-time access: {scores_fg.online_enabled}")

if __name__ == "__main__":
    main()