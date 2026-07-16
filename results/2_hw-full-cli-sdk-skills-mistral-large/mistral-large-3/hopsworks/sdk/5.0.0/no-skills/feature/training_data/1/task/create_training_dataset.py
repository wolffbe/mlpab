#!/usr/bin/env python3
"""
Create a versioned training dataset named `churntraining1e2e16`, version 1, on Hopsworks.

For each (account_id, label_time) in labels.csv, fetch the most recent feature values
from the source tables where event_time <= label_time.
"""

import hopsworks
import pandas as pd
import os

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Read input files
labels_df = pd.read_csv("./data/labels.csv")
transactions_df = pd.read_csv("./data/transactions.csv")
transactions_late_df = pd.read_csv("./data/transactions_late.csv")
profiles_df = pd.read_csv("./data/profiles.csv")
activity_df = pd.read_csv("./data/activity.csv")
account_health_df = pd.read_csv("./data/account_health.csv")

# Combine transactions and transactions_late
transactions_combined = pd.concat([transactions_df, transactions_late_df], ignore_index=True)

# Initialize the output DataFrame
output_columns = [
    "account_id", "label_time", "amount", "balance", 
    "credit_score", "tier", "sessions_7d", "health_score", "churned"
]
output_rows = []

# For each (account_id, label_time) in labels.csv, fetch the most recent feature values
for _, row in labels_df.iterrows():
    account_id = row["account_id"]
    label_time = row["label_time"]
    churned = row["churned"]
    
    # Fetch most recent transaction
    tx_filtered = transactions_combined[
        (transactions_combined["account_id"] == account_id) &
        (transactions_combined["event_time"] <= label_time)
    ]
    tx_most_recent = tx_filtered.sort_values("event_time", ascending=False).head(1)
    
    # Fetch most recent profile
    profile_filtered = profiles_df[
        (profiles_df["account_id"] == account_id) &
        (profiles_df["event_time"] <= label_time)
    ]
    profile_most_recent = profile_filtered.sort_values("event_time", ascending=False).head(1)
    
    # Fetch most recent activity
    activity_filtered = activity_df[
        (activity_df["account_id"] == account_id) &
        (activity_df["event_time"] <= label_time)
    ]
    activity_most_recent = activity_filtered.sort_values("event_time", ascending=False).head(1)
    
    # Fetch most recent account health
    health_filtered = account_health_df[
        (account_health_df["account_id"] == account_id) &
        (account_health_df["event_time"] <= label_time)
    ]
    health_most_recent = health_filtered.sort_values("event_time", ascending=False).head(1)
    
    # Prepare the output row
    output_row = {
        "account_id": account_id,
        "label_time": label_time,
        "amount": tx_most_recent["amount"].values[0] if not tx_most_recent.empty else None,
        "balance": tx_most_recent["balance"].values[0] if not tx_most_recent.empty else None,
        "credit_score": profile_most_recent["credit_score"].values[0] if not profile_most_recent.empty else None,
        "tier": profile_most_recent["tier"].values[0] if not profile_most_recent.empty else None,
        "sessions_7d": activity_most_recent["sessions_7d"].values[0] if not activity_most_recent.empty else None,
        "health_score": health_most_recent["health_score"].values[0] if not health_most_recent.empty else None,
        "churned": churned
    }
    output_rows.append(output_row)

# Create the output DataFrame
output_df = pd.DataFrame(output_rows, columns=output_columns)

# Drop rows with missing values (if any)
output_df = output_df.dropna()

# Create a feature group and upload the dataset
fg = fs.create_feature_group(
    name="churntraining1e2e16",
    version=1,
    description="Training dataset for churn prediction. One row per (account_id, label_time) in labels.csv, with the most recent feature values at or before label_time.",
    primary_key=["account_id", "label_time"],
    event_time="label_time"
)

# Upload the dataset
fg.insert(output_df)

print("Training dataset 'churntraining1e2e16', version 1, has been created on Hopsworks.")