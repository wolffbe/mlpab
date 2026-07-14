#!/usr/bin/env python3
"""
Enable online access for the feature table transactions4adadd.
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import ml

# Environment variables
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # format: workspace.<run-id>

# Table configuration
FEATURE_TABLE_NAME = "transactions4adadd"
FEATURE_TABLE_FULL_NAME = f"{SCHEMA}.{FEATURE_TABLE_NAME}"
ONLINE_TABLE_NAME = f"{PREFIX}_{FEATURE_TABLE_NAME}_online"


def main():
    # Initialize client
    client = WorkspaceClient()
    
    print(f"Enabling online access for table {FEATURE_TABLE_FULL_NAME}")
    
    # Step 1: Create an online store if it doesn't exist
    print("Checking for online store...")
    
    try:
        # Try to get the online store
        online_store = client.feature_store.get_online_store("workspace")
        print(f"Online store already exists: {online_store}")
    except Exception as e:
        print(f"Online store not found, creating: {e}")
        # Create the online store
        online_store = client.feature_store.create_online_store(
            name="workspace",
            region=os.environ.get("AWS_REGION", "us-west-2")
        )
        print(f"Online store created: {online_store}")
    
    # Step 2: Publish the table to the online store
    print("Publishing table to online store...")
    
    publish_response = client.feature_store.publish_table(
        source_table_name=FEATURE_TABLE_FULL_NAME,
        publish_spec=ml.PublishSpec(
            online_store="workspace",
            online_table_name=ONLINE_TABLE_NAME,
            publish_mode=ml.PublishSpecPublishMode.SNAPSHOT
        )
    )
    
    print(f"Online access enabled: {publish_response}")
    
    print("Online access setup complete!")
    print(f"Feature table: {FEATURE_TABLE_FULL_NAME}")
    print(f"Online table: {ONLINE_TABLE_NAME}")

if __name__ == "__main__":
    main()