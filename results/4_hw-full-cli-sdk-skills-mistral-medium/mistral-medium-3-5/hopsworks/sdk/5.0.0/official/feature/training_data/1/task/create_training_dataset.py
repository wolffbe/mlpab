#!/usr/bin/env python3
import hopsworks
import pandas as pd
import os

# Login to Hopsworks
project = hopsworks.login()

# Read all the CSV files from the data directory
print("Reading CSV files...")
labels_df = pd.read_csv("data/labels.csv")
transactions_df = pd.read_csv("data/transactions.csv")
transactions_late_df = pd.read_csv("data/transactions_late.csv")
profiles_df = pd.read_csv("data/profiles.csv")
activity_df = pd.read_csv("data/activity.csv")
account_health_df = pd.read_csv("data/account_health.csv")

# Combine transactions
all_transactions_df = pd.concat([transactions_df, transactions_late_df], ignore_index=True)

# For each (account_id, label_time) in labels, find the most recent feature values at or before label_time
print("Processing data...")

result_rows = []
for _, label_row in labels_df.iterrows():
    account_id = label_row['account_id']
    label_time = label_row['label_time']
    churned = label_row['churned']
    
    # Get most recent transaction for this account at or before label_time
    tx_mask = (all_transactions_df['account_id'] == account_id) & (all_transactions_df['event_time'] <= label_time)
    if tx_mask.any():
        tx_recent = all_transactions_df[tx_mask].loc[all_transactions_df[tx_mask]['event_time'].idxmax()]
        amount = tx_recent['amount']
        balance = tx_recent['balance']
    else:
        amount = None
        balance = None
    
    # Get most recent profile for this account at or before label_time
    profile_mask = (profiles_df['account_id'] == account_id) & (profiles_df['event_time'] <= label_time)
    if profile_mask.any():
        profile_recent = profiles_df[profile_mask].loc[profiles_df[profile_mask]['event_time'].idxmax()]
        credit_score = profile_recent['credit_score']
        tier = profile_recent['tier']
    else:
        credit_score = None
        tier = None
    
    # Get most recent activity for this account at or before label_time
    activity_mask = (activity_df['account_id'] == account_id) & (activity_df['event_time'] <= label_time)
    if activity_mask.any():
        activity_recent = activity_df[activity_mask].loc[activity_df[activity_mask]['event_time'].idxmax()]
        sessions_7d = activity_recent['sessions_7d']
    else:
        sessions_7d = None
    
    # Get most recent account_health for this account at or before label_time
    health_mask = (account_health_df['account_id'] == account_id) & (account_health_df['event_time'] <= label_time)
    if health_mask.any():
        health_recent = account_health_df[health_mask].loc[account_health_df[health_mask]['event_time'].idxmax()]
        health_score = health_recent['health_score']
    else:
        health_score = None
    
    result_rows.append({
        'account_id': account_id,
        'label_time': label_time,
        'amount': amount,
        'balance': balance,
        'credit_score': credit_score,
        'tier': tier,
        'sessions_7d': sessions_7d,
        'health_score': health_score,
        'churned': churned
    })

# Create DataFrame
result_df = pd.DataFrame(result_rows)

# Check for any None values
print("\nNull values:")
print(result_df.isnull().sum())

# Save to CSV temporarily
result_df.to_csv("churntraining3a03fd.csv", index=False)
print("\nSaved to churntraining3a03fd.csv")
print("\nFirst few rows:")
print(result_df.head())

# Now create the training dataset on Hopsworks
print("\nCreating training dataset on Hopsworks...")
fs = project.get_feature_store()

training_dataset = fs.create_training_dataset(
    name="churntraining3a03fd",
    version=1,
    description="Churn training dataset with features joined at label_time",
    data_format="csv"
)

# Save the data to the training dataset
print("Saving data to training dataset...")
training_dataset.save(result_df)

print("\nTraining dataset created successfully!")
print(f"Name: {training_dataset.name}")
print(f"Version: {training_dataset.version}")
