#!/usr/bin/env python3
"""
Enable online access for the feature table.
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog

# Initialize Databricks WorkspaceClient
w = WorkspaceClient()

# Get schema and prefix from environment variables
full_schema_name = os.environ['MLPAB_DATABRICKS_SCHEMA']
table_name = "recse3a36e"
full_table_name = f"{full_schema_name}.{table_name}"

# Enable online access for low-latency lookup
try:
    w.online_tables.create(
        catalog.CreateOnlineTableRequest(
            source_table_full_name=full_table_name,
            primary_key_columns=["rec_id"]
        )
    )
    print(f"Online table {full_table_name} created successfully.")
except Exception as e:
    if "already exists" not in str(e):
        raise

print(f"Feature table {full_table_name} is ready for online access.")