#!/usr/bin/env python3
"""
Create a feature table named `profiles395e7c`, version 1, in the specified schema.
Load data from data/features.csv, enable online access, and retrieve feature vectors
for the keys in data/lookup_keys.txt via the online store.
"""

import os
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog
from databricks.sdk.service import sql
from databricks.sdk.service import compute
from databricks.sdk.service import ml
from databricks.feature_engineering import FeatureEngineeringClient

# Environment variables
SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # workspace.<run-id>
PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")  # mlpab05c624

# Feature table details
FEATURE_TABLE_NAME = "profiles395e7c"
FEATURE_TABLE_VERSION = 1
RECORD_KEY = "account_id"

# Paths
data_path = "data/features.csv"
lookup_keys_path = "data/lookup_keys.txt"
output_path = "submission/answers.json"

# Initialize clients
w = WorkspaceClient()
fe = FeatureEngineeringClient()


def read_lookup_keys():
    """Read account_ids from lookup_keys.txt."""
    with open(lookup_keys_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def create_feature_table():
    """Create a feature table in Unity Catalog and enable online access."""
    full_table_name = f"{SCHEMA}.{FEATURE_TABLE_NAME}"
    
    # Check if the table already exists
    try:
        w.tables.get(full_table_name)
        print(f"Table {full_table_name} already exists. Skipping creation.")
        return full_table_name
    except Exception:
        pass
    
    # Create a temporary volume to stage the CSV
    volume_name = f"{PREFIX}_temp_volume"
    full_volume_name = f"{SCHEMA}.{volume_name}"
    
    try:
        w.volumes.create(
            catalog_name=SCHEMA.split(".")[0],
            schema_name=SCHEMA.split(".")[1],
            name=volume_name,
            volume_type=catalog.VolumeType.MANAGED
        )
    except Exception as e:
        print(f"Volume may already exist: {e}")
    
    # Upload the CSV to the volume
    volume_path = f"/Volumes/{SCHEMA.split('.')[0]}/{SCHEMA.split('.')[1]}/{volume_name}"
    dbfs_path = f"{volume_path}/features.csv"
    
    # Read the CSV content and write to the volume
    with open(data_path, "r") as f:
        csv_content = f.read()
    
    # Write to the volume using the WorkspaceClient
    volume_file_path = f"{volume_path}/features.csv"
    w.files.upload(volume_file_path, csv_content, overwrite=True)
    
    # Create the feature table using SQL
    create_table_sql = f"""
    CREATE TABLE {full_table_name} (
        account_id STRING,
        f1 DOUBLE,
        f2 DOUBLE,
        f3 DOUBLE,
        f4 DOUBLE
    )
    USING DELTA
    LOCATION '{volume_path}/feature_table'
    AS SELECT * FROM csv.`{volume_file_path}`
    """
    
    w.statement_execution.execute_statement(
        warehouse_id=w.warehouses.list()[0].id,
        catalog=SCHEMA.split(".")[0],
        schema=SCHEMA.split(".")[1],
        statement=create_table_sql
    ).result()
    
    # Enable online table for low-latency access
    fe.create_table(
        name=full_table_name,
        primary_keys=[RECORD_KEY],
        df=None,  # Not needed since the table already exists
        schema=None,  # Not needed since the table already exists
        description="Feature table for account profiles"
    )
    
    # Enable online store
    fe.publish_table(
        name=full_table_name,
        online_store_spec=ml.OnlineStoreSpec(
            pubsub_spec=ml.PubSubSpec(
                channel="online_feature_store"
            )
        )
    )
    
    return full_table_name


def retrieve_online_features(account_ids):
    """Retrieve feature vectors for the given account_ids via the online store."""
    full_table_name = f"{SCHEMA}.{FEATURE_TABLE_NAME}"
    
    # Retrieve feature vectors
    vectors = {}
    for account_id in account_ids:
        try:
            row = fe.get_online_feature(
                name=full_table_name,
                lookup_key={RECORD_KEY: account_id}
            )
            if row:
                # Extract feature values in order f1, f2, f3, f4
                feature_values = [
                    row[f"f1"],
                    row[f"f2"],
                    row[f"f3"],
                    row[f"f4"]
                ]
                vectors[account_id] = feature_values
        except Exception as e:
            print(f"Error retrieving features for {account_id}: {e}")
    
    return vectors


def main():
    # Create the feature table and enable online access
    create_feature_table()
    
    # Read lookup keys
    account_ids = read_lookup_keys()
    
    # Retrieve feature vectors via online store
    vectors = retrieve_online_features(account_ids)
    
    # Write the output
    os.makedirs("submission", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"vectors": vectors}, f, indent=2)
    
    print(f"Feature vectors written to {output_path}")


if __name__ == "__main__":
    main()