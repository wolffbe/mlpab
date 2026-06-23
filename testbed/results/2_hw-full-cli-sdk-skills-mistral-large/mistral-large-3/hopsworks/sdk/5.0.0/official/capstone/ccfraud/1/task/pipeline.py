#!/usr/bin/env python3
"""
Full FTI pipeline for credit-card fraud detection on Hopsworks.
1. Engineer fraud features into feature group `cctxnc444ca`.
2. Assemble training dataset `cctdc444ca`.
3. Train and register classifier `ccmodelc444ca` with metrics (remote job).
4. Score `score_transactions.csv` into feature table `ccpredc444ca` (remote job).
"""

import hopsworks
import pandas as pd
import numpy as np

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Load data
transactions_df = pd.read_csv("data/transactions.csv", parse_dates=["datetime"])
score_transactions_df = pd.read_csv("data/score_transactions.csv", parse_dates=["datetime"])

# --- Feature Engineering ---
def engineer_features(df):
    # Sort by card and time
    df = df.sort_values(by=["cc_num", "datetime"])
    
    # Transaction velocity: count of transactions in the last 1 hour (simplified)
    df["transaction_velocity_1h"] = (df.groupby("cc_num").cumcount() + 1).astype(float)
    
    # Transaction velocity: count of transactions in the last 24 hours (simplified)
    df["transaction_velocity_24h"] = (df.groupby("cc_num").cumcount() + 1).astype(float)
    
    # Amount statistics: mean and std of transaction amounts (simplified)
    df["amount_mean_24h"] = df.groupby("cc_num")["amount"].transform("mean")
    df["amount_std_24h"] = df.groupby("cc_num")["amount"].transform("std")
    
    # Geo distance: distance from the card's first transaction location
    df["first_lat"] = df.groupby("cc_num")["lat"].transform("first")
    df["first_long"] = df.groupby("cc_num")["long"].transform("first")
    df["geo_distance"] = np.sqrt(
        (df["lat"] - df["first_lat"]) ** 2 + (df["long"] - df["first_long"]) ** 2
    )
    
    # Time since last transaction
    df["time_since_last_tx"] = df.groupby("cc_num")["datetime"].diff().dt.total_seconds() / 3600
    
    # Fill NaN values
    df = df.fillna({
        "transaction_velocity_1h": 1,
        "transaction_velocity_24h": 1,
        "amount_mean_24h": df["amount"],
        "amount_std_24h": 0,
        "geo_distance": 0,
        "time_since_last_tx": 24,
    })
    
    # Geo distance: distance from the card's first transaction location
    df["first_lat"] = df.groupby("cc_num")["lat"].transform("first")
    df["first_long"] = df.groupby("cc_num")["long"].transform("first")
    df["geo_distance"] = np.sqrt(
        (df["lat"] - df["first_lat"]) ** 2 + (df["long"] - df["first_long"]) ** 2
    )
    
    # Time since last transaction
    df["time_since_last_tx"] = df.groupby("cc_num")["datetime"].diff().dt.total_seconds() / 3600
    
    # Fill NaN values (first transactions for a card)
    df = df.fillna({
        "transaction_velocity_1h": 1,
        "transaction_velocity_24h": 1,
        "amount_mean_24h": df["amount"],
        "amount_std_24h": 0,
        "geo_distance": 0,
        "time_since_last_tx": 24,
    })
    
    return df

# Engineer features for training data
transactions_df = engineer_features(transactions_df)

# --- Feature Group Creation ---
feature_group = fs.get_or_create_feature_group(
    name="cctxnc444ca",
    version=1,
    description="Fraud detection features for credit-card transactions",
    primary_key=["transaction_id"],
    event_time="datetime",
    online_enabled=True,
)

# Insert data into feature group
feature_group.insert(transactions_df, write_options={"wait_for_job": True})

# --- Training Dataset Creation ---
training_dataset = fs.create_training_dataset(
    name="cctdc444ca",
    version=2,
    description="Training dataset for fraud detection model",
    data_format="csv",
    statistics_config={"enabled": True, "histograms": True, "correlations": True},
)

# Populate training dataset
query = feature_group.select_all()
training_dataset.save(query, write_options={"wait_for_job": True})

# --- Remote Training Job ---
with open("train_script.py", "r") as f:
    train_script = f.read()

# Create and run training job
train_job = project.create_job(
    name="train_fraud_model",
    script=train_script,
    job_type="PYTHON",
    run_as_agent=True,
)
train_job.run(wait=True)

# --- Remote Scoring Job ---
# Engineer features for scoring data
score_transactions_df = engineer_features(score_transactions_df)

# Write scoring data to a temporary feature group
score_fg = fs.get_or_create_feature_group(
    name="temp_score_transactions",
    version=1,
    description="Temporary feature group for scoring transactions",
    primary_key=["transaction_id"],
    event_time="datetime",
    online_enabled=False,
)
score_fg.insert(score_transactions_df, write_options={"wait_for_job": True})

with open("score_script.py", "r") as f:
    score_script = f.read()

# Create and run scoring job
score_job = project.create_job(
    name="score_fraud_transactions",
    script=score_script,
    job_type="PYTHON",
    run_as_agent=True,
)
score_job.run(wait=True)

print("Pipeline completed successfully.")