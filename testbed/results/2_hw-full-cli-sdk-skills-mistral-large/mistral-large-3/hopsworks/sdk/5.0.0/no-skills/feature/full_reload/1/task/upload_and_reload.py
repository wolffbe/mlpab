#!/usr/bin/env python3
"""
Register and reload feature table customers1ae8a4 on Hopsworks.
Version 1: initial_export.csv (original schema).
Version 2: new_export.csv (new schema), online-enabled.
"""

import hopsworks
import pandas as pd
import os

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Paths
data_dir = "./data"
initial_export = os.path.join(data_dir, "initial_export.csv")
new_export = os.path.join(data_dir, "reload/new_export.csv")

# --- Version 1: Register and upload initial export ---
print("Creating version 1 of customers1ae8a4...")
customers_v1_df = pd.read_csv(initial_export)

# Register feature table
customers_v1_fg = fs.create_feature_group(
    name="customers1ae8a4",
    version=1,
    description="Initial export of customers table (original schema)",
    primary_key=["row_id"],
    event_time="updated_at",
    online_enabled=False,
)

# Upload data
customers_v1_fg.insert(customers_v1_df, write_options={"wait_for_job": True})
print("Version 1 created and data uploaded.")

# --- Version 2: Drop version 1, register new schema, upload new export ---
print("Creating version 2 of customers1ae8a4...")
customers_v2_df = pd.read_csv(new_export)

# Register feature table (new schema)
customers_v2_fg = fs.create_feature_group(
    name="customers1ae8a4",
    version=2,
    description="Full reload of customers table (new schema)",
    primary_key=["row_id"],
    event_time="updated_at",
    online_enabled=True,  # Enable online access
)

# Upload data
customers_v2_fg.insert(customers_v2_df, write_options={"wait_for_job": True})
print("Version 2 created, data uploaded, and online access enabled.")

print("Task complete.")