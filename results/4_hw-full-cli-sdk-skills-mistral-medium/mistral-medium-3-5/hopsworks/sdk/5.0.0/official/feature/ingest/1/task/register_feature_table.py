#!/usr/bin/env python3
"""
Register feature table 'transactionse6da16' v1 from the two export CSV files.
Deduplicates by row_id (keeping first occurrence) and loads all unique rows.
"""
import pandas as pd
import hopsworks

# Connect to Hopsworks
hopsworks.login()
project = hopsworks.get_current_project()
fs = project.get_feature_store()

# Read both CSV files
print("Reading CSV files...")
df1 = pd.read_csv("data/transactions_export_1.csv")
df2 = pd.read_csv("data/transactions_export_2.csv")

print(f"File 1: {len(df1)} rows")
print(f"File 2: {len(df2)} rows")

# Concatenate and deduplicate by row_id, keeping first occurrence
print("Deduplicating by row_id...")
df_combined = pd.concat([df1, df2], ignore_index=True)
df_deduped = df_combined.drop_duplicates(subset=["row_id"], keep="first")

print(f"Combined (before dedup): {len(df_combined)} rows")
print(f"Deduplicated: {len(df_deduped)} rows")

# Verify we have all unique row_ids
print(f"Unique row_ids: {df_deduped['row_id'].nunique()}")

# Register the feature table
print("\nRegistering feature table 'transactionse6da16' v1...")

fg_name = "transactionse6da16"
version = 1

# Create the feature group with online enabled
fg = fs.get_or_create_feature_group(
    name=fg_name,
    version=version,
    description="Transactions feature table from export files",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
)

print(f"Feature group {fg_name} v{version} created/accessed")
print(f"  Online enabled: {fg.online_enabled}")

# Insert the data
print(f"\nInserting {len(df_deduped)} rows into feature table...")
job, report = fg.insert(
    df_deduped,
    write_options={"wait": True},
    wait=True
)

print("\nDone! Feature table registered and data loaded.")

# Verify the data was loaded
fg = fs.get_feature_group(fg_name, version=version)
print(f"\nFeature group info:")
print(f"  Name: {fg.name}")
print(f"  Version: {fg.version}")
print(f"  Online enabled: {fg.online_enabled}")
print(f"  Primary key: {fg.primary_key}")
print(f"  Event time: {fg.event_time}")

# Check the data
try:
    df_read = fg.read()
    print(f"  Rows in feature table: {len(df_read)}")
    print(f"  Columns: {list(df_read.columns)}")
except Exception as e:
    print(f"  Error reading back: {e}")

print("\nFeature table 'transactionse6da16' v1 is ready with online access enabled.")
