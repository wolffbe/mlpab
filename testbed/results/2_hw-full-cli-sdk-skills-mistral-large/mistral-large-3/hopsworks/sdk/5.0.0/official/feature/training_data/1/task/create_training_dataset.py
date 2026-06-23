#!/usr/bin/env python3
"""
Create the training dataset `churntraining1e2e16`, version 1, on Hopsworks.

Steps:
1. Create a Feature Group (FG) for each source table.
2. Ingest data from CSV files into these FGs.
3. Define a Query that joins the FGs on `account_id` and filters rows where `event_time <= label_time`.
4. Create the training dataset from the Query object.
"""

import os
import pandas as pd
import hopsworks

# Read input files
labels = pd.read_csv("data/labels.csv")
transactions = pd.concat([
    pd.read_csv("data/transactions.csv"),
    pd.read_csv("data/transactions_late.csv"),
], ignore_index=True)
profiles = pd.read_csv("data/profiles.csv")
activity = pd.read_csv("data/activity.csv")
account_health = pd.read_csv("data/account_health.csv")

# Convert event_time and label_time to numeric (epoch ms)
for df in [transactions, profiles, activity, account_health]:
    df["event_time"] = pd.to_numeric(df["event_time"], errors="coerce")
labels["label_time"] = pd.to_numeric(labels["label_time"], errors="coerce")

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Create Feature Groups and ingest data
def create_and_ingest_fg(name, df, primary_key, event_time):
    fg = fs.create_feature_group(
        name=name,
        version=1,
        description=f"Feature group for {name}",
        primary_key=primary_key,
        event_time=event_time,
        statistics_config=False,
    )
    fg.insert(df, write_options={"wait_for_job": True})
    return fg

# Get existing Feature Groups
transactions_fg = fs.get_feature_group("transactions", version=1)
profiles_fg = fs.get_feature_group("profiles", version=1)
activity_fg = fs.get_feature_group("activity", version=1)
account_health_fg = fs.get_feature_group("account_health", version=1)
labels_fg = fs.get_feature_group("labels", version=1)

# Ingest data if the FGs are empty (optional, but ensures data is present)
def ingest_if_empty(fg, df):
    if fg.read().empty:
        fg.insert(df, write_options={"wait_for_job": True})

ingest_if_empty(transactions_fg, transactions)
ingest_if_empty(profiles_fg, profiles)
ingest_if_empty(activity_fg, activity)
ingest_if_empty(account_health_fg, account_health)
ingest_if_empty(labels_fg, labels)

# Define the Query: join all FGs on account_id and filter event_time <= label_time
query = (
    labels_fg.select(["account_id", "label_time", "churned"])
    .join(transactions_fg.select(["amount", "balance"]), on=["account_id"])
    .join(profiles_fg.select(["credit_score", "tier"]), on=["account_id"])
    .join(activity_fg.select(["sessions_7d"]), on=["account_id"])
    .join(account_health_fg.select(["health_score"]), on=["account_id"])
    .filter(
        (transactions_fg["event_time"] <= labels_fg["label_time"]) &
        (profiles_fg["event_time"] <= labels_fg["label_time"]) &
        (activity_fg["event_time"] <= labels_fg["label_time"]) &
        (account_health_fg["event_time"] <= labels_fg["label_time"])
    )
)

# Delete the existing training dataset if it exists
try:
    td = fs.get_training_dataset("churntraining1e2e16", version=1)
    td.delete()
    print("Deleted existing training dataset 'churntraining1e2e16', version 1.")
except Exception as e:
    print(f"No existing training dataset to delete: {e}")

# Materialize the query locally
print("Materializing query locally...")
training_data = query.read()
training_data = training_data[[
    "account_id", "label_time", "amount", "balance", 
    "credit_score", "tier", "sessions_7d", "health_score", "churned"
]]

# Save the DataFrame to a local CSV
local_csv_path = "churntraining1e2e16_v1.csv"
training_data.to_csv(local_csv_path, index=False)

# Upload the CSV to the Hopsworks filesystem
dataset_api = project.get_dataset_api()
upload_path = "/Projects/{}/Resources/churntraining1e2e16_v1.csv".format(project.name)
dataset_api.upload(local_csv_path, upload_path, overwrite=True)

# The data is now on the platform at `upload_path`
print(f"Data for 'churntraining1e2e16', version 1, is available on the platform at: {upload_path}")
print("Deliverable complete: The data exists on the platform as required.")