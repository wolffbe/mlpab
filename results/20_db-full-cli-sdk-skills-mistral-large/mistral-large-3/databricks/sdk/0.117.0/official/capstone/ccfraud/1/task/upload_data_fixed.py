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
catalog_name = schema_name.split(".")[0]  # workspace
schema_name_only = schema_name.split(".")[1]  # mlpab21b96f
volume_name = f"{os.getenv('MLPAB_DATABRICKS_PREFIX')}_fraud_data"
volume_path = f"/Volumes/{catalog_name}/{schema_name_only}/{volume_name}"
local_transactions_path = "data/transactions.csv"
local_score_path = "data/score_transactions.csv"

# Ensure schema exists
print(f"Ensuring schema exists: {schema_name}")
try:
    w.schemas.get(f"{catalog_name}.{schema_name_only}")
except Exception as e:
    print(f"Schema does not exist, creating: {e}")
    w.schemas.create(
        catalog_name=catalog_name,
        name=schema_name_only,
    )

# Create volume
print(f"Creating volume: {volume_name}")
try:
    w.volumes.create(
        catalog_name=catalog_name,
        schema_name=schema_name_only,
        name=volume_name,
        volume_type=catalog.VolumeType.MANAGED,
    )
except Exception as e:
    print(f"Volume may already exist: {e}")

# Upload files
print(f"Uploading {local_transactions_path} to {volume_path}")
w.files.upload(
    f"{volume_path}/transactions.csv",
    open(local_transactions_path, "rb"),
    overwrite=True,
)

print(f"Uploading {local_score_path} to {volume_path}")
w.files.upload(
    f"{volume_path}/score_transactions.csv",
    open(local_score_path, "rb"),
    overwrite=True,
)

print("Data upload complete.")