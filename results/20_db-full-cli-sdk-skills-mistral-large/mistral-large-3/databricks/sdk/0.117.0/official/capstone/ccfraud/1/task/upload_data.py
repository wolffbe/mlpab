#!/usr/bin/env python3
"""
Uploads transactions.csv and score_transactions.csv to a Unity Catalog volume.
"""
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog

# Initialize WorkspaceClient
w = WorkspaceClient()

# Config
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # workspace.mlpab21b96f
volume_name = f"{os.getenv('MLPAB_DATABRICKS_PREFIX')}_fraud_data"
volume_path = f"{schema_name}.{volume_name}"
local_transactions_path = "data/transactions.csv"
local_score_path = "data/score_transactions.csv"

# Create volume
print(f"Creating volume: {volume_path}")
w.volumes.create(
    catalog_name="workspace",
    schema_name=schema_name.split(".")[1],
    name=volume_name,
    volume_type=catalog.VolumeType.MANAGED,
)

# Upload files
print(f"Uploading {local_transactions_path} to {volume_path}")
w.files.upload(
    f"/Volumes/{volume_path}/transactions.csv",
    open(local_transactions_path, "rb"),
    overwrite=True,
)

print(f"Uploading {local_score_path} to {volume_path}")
w.files.upload(
    f"/Volumes/{volume_path}/score_transactions.csv",
    open(local_score_path, "rb"),
    overwrite=True,
)

print("Data upload complete.")