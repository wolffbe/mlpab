#!/usr/bin/env python
import hopsworks
import hsfs
import pandas as pd

# Connect to Hopsworks
hopsworks.login()

# Get the feature store
fs_api = hsfs.core.feature_store_api.FeatureStoreApi()
fs_list = fs_api.get_all()
fs = fs_list[0]
print(f"Feature store: {fs.name} (id: {fs.id})")

# Read both CSV files
print("Reading CSV files...")
df1 = pd.read_csv("data/transactions_export_1.csv")
df2 = pd.read_csv("data/transactions_export_2.csv")

print(f"File 1: {len(df1)} rows")
print(f"File 2: {len(df2)} rows")

# Concatenate and deduplicate by row_id
print("Combining and deduplicating...")
df_combined = pd.concat([df1, df2])
df_deduped = df_combined.drop_duplicates(subset=["row_id"], keep="first")
print(f"Combined unique rows: {len(df_deduped)}")

# Define feature table name and version
feature_group_name = "transactionse6da16"
feature_group_version = 1

# Check if feature group already exists
try:
    fg = fs.get_feature_group(feature_group_name, feature_group_version)
    print(f"Feature group {feature_group_name} v{feature_group_version} already exists")
except:
    # Create the feature group
    print(f"Creating feature group {feature_group_name} v{feature_group_version}...")
    
    # Define features
    from hsfs.feature import Feature
    features = [
        Feature("row_id", "String", description="Unique record key"),
        Feature("account_id", "String"),
        Feature("event_time", "Long", description="Event time in epoch milliseconds"),
        Feature("amount", "Double"),
        Feature("category", "String"),
    ]
    
    fg = fs.create_feature_group(
        name=feature_group_name,
        version=feature_group_version,
        description="Transactions feature table",
        primary_key=["row_id"],
        event_time="event_time",
        online_enabled=True,  # Enable online/real-time access
        features=features,
    )
    print(f"Created feature group: {fg}")

# Get the feature group (in case it already existed)
fg = fs.get_feature_group(feature_group_name, feature_group_version)

# Insert data
print(f"Inserting data into feature group...")
fg.insert(df_deduped)
print(f"Data inserted successfully")

# Verify online is enabled
print(f"Online enabled: {fg.online_enabled}")

print("Done!")
