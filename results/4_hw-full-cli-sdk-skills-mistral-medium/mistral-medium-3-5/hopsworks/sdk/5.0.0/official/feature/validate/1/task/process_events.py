#!/usr/bin/env python3
"""
Process events.csv according to the data contract and register on Hopsworks.
"""
import csv
import json
import os
import hopsworks

# Read the CSV file
with open('data/events.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Contract rules
VALID_CATEGORIES = {'grocery', 'travel', 'salary', 'rent', 'other'}

# Identify valid and invalid rows
valid_rows = []
rejected_ids = []

for row in rows:
    row_id = row['row_id']
    amount_str = row['amount']
    category = row['category']
    
    # Rule 1: amount must be present
    if amount_str is None or amount_str.strip() == '':
        rejected_ids.append(row_id)
        continue
    
    # Rule 2: amount must be within [0, 10000]
    try:
        amount = float(amount_str)
    except ValueError:
        rejected_ids.append(row_id)
        continue
    
    if amount < 0 or amount > 10000:
        rejected_ids.append(row_id)
        continue
    
    # Rule 3: category must be valid
    if category not in VALID_CATEGORIES:
        rejected_ids.append(row_id)
        continue
    
    # All rules passed
    valid_rows.append(row)

# Sort rejected IDs for consistency
rejected_ids.sort()

# Write answers.json
os.makedirs('submission', exist_ok=True)
with open('submission/answers.json', 'w') as f:
    json.dump({"rejected": rejected_ids}, f, indent=2)

print(f"Valid rows: {len(valid_rows)}")
print(f"Rejected rows: {len(rejected_ids)}")
print(f"Rejected IDs: {rejected_ids}")

# Now connect to Hopsworks and register the feature table
print("\nConnecting to Hopsworks...")
hopsworks.login()
fs = hopsworks.get_current_project().get_feature_store()

# Prepare the valid data as a DataFrame
import pandas as pd

valid_df = pd.DataFrame(valid_rows)
# Convert amount to float
valid_df['amount'] = valid_df['amount'].astype(float)
# Convert event_time to int (epoch milliseconds)
valid_df['event_time'] = valid_df['event_time'].astype('int64')

print(f"\nDataFrame shape: {valid_df.shape}")
print(f"DataFrame dtypes:\n{valid_df.dtypes}")
print(f"\nFirst few rows:\n{valid_df.head()}")

# Register the feature table
print("\nRegistering feature table 'events4b3862' version 1...")

# Create the feature group with online_enabled
fg = fs.get_or_create_feature_group(
    name="events4b3862",
    version=1,
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
    description="Events data filtered by contract rules"
)

# Insert the valid data
print("Inserting valid rows...")
fg.insert(valid_df, write_options={"wait_for_job": True})

# Enable online feature store - create online deployment
print("\nEnabling online feature store...")
fg = fs.get_feature_group("events4b3862", version=1)

# Create online deployment
try:
    mr = fg.create_online_deployment(
        wait=True
    )
    print(f"Online deployment created: {mr}")
except Exception as e:
    print(f"Online deployment may already exist or error: {e}")

print("\nDone! Feature table registered and online access enabled.")
print(f"Rejected IDs written to submission/answers.json")
