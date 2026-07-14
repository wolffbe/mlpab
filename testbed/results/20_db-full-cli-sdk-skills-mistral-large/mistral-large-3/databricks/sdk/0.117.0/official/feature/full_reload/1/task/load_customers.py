#!/usr/bin/env python3
"""
Register and load feature tables for customersc23945, versions 1 and 2.
Enable online access for version 2.
"""

import os
import databricks.sdk
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog
from databricks.sdk.service import ml

# Initialize the Databricks WorkspaceClient
w = WorkspaceClient()

# Environment variables for isolation
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")
prefix = os.getenv("MLPAB_DATABRICKS_PREFIX")

# Ensure the schema exists
try:
    w.schemas.create(name=schema_name.split(".")[-1], catalog_name=schema_name.split(".")[0])
except Exception as e:
    print(f"Schema may already exist or error: {e}")

# Define the feature table name
feature_table_name = f"{schema_name}.customersc23945"

# --- Version 1: Initial Export ---
# Register and load the initial export
initial_export_path = "./data/initial_export.csv"

# Create a volume to stage the data
volume_name = f"{schema_name}.{prefix}_volume"
try:
    w.volumes.create(
        name=f"{prefix}_volume",
        catalog_name=schema_name.split(".")[0],
        schema_name=schema_name.split(".")[1],
        volume_type=catalog.VolumeType.MANAGED
    )
except Exception as e:
    print(f"Volume may already exist or error: {e}")

# Load data into the feature table (version 1)
w.api_client.do(
    "POST", 
    "/api/2.0/feature-store/feature-tables/create",
    body={
        "name": feature_table_name,
        "primary_keys": ["row_id"],
        "timestamp_keys": ["updated_at"],
        "schema": [
            {"name": "row_id", "type": "string"},
            {"name": "name", "type": "string"},
            {"name": "balance_eur", "type": "double"},
            {"name": "updated_at", "type": "long"}
        ],
        "description": "Initial export of customers table (version 1)"
    }
)

w.api_client.do(
    "POST", 
    "/api/2.0/feature-store/feature-tables/write",
    body={
        "name": feature_table_name,
        "path": os.path.abspath("./data/initial_export.csv"),
        "source_type": "CSV",
        "mode": "OVERWRITE"
    }
)

# --- Version 2: New Export ---
# Register and load the new export
new_export_path = "./data/reload/new_export.csv"

# Recreate the feature table (version 2)
w.api_client.do(
    "POST", 
    "/api/2.0/feature-store/feature-tables/create",
    body={
        "name": feature_table_name,
        "primary_keys": ["row_id"],
        "timestamp_keys": ["updated_at"],
        "schema": [
            {"name": "row_id", "type": "string"},
            {"name": "full_name", "type": "string"},
            {"name": "balance", "type": "double"},
            {"name": "currency", "type": "string"},
            {"name": "updated_at", "type": "long"}
        ],
        "description": "Re-export of customers table (version 2)"
    }
)

w.api_client.do(
    "POST", 
    "/api/2.0/feature-store/feature-tables/write",
    body={
        "name": feature_table_name,
        "path": os.path.abspath("./data/reload/new_export.csv"),
        "source_type": "CSV",
        "mode": "OVERWRITE"
    }
)

# --- Enable Online Access for Version 2 ---
# Create an online table for version 2
online_table_name = f"{prefix}_online_customersc23945"
try:
    w.api_client.do(
        "POST", 
        "/api/2.0/feature-store/online-stores/create",
        body={
            "name": online_table_name,
            "feature_table_name": feature_table_name,
            "primary_key_columns": ["row_id"]
        }
    )
    print(f"Online table '{online_table_name}' created successfully.")
except Exception as e:
    print(f"Online table may already exist or error: {e}")

print("Feature tables and online access configured successfully.")