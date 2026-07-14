#!/usr/bin/env python3
"""
Create a feature table named transactions4adadd version 1 on Databricks platform.
Register with record key row_id and event-time column event_time (epoch milliseconds).
Enable online/real-time access for low-latency lookup.
"""

import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog
from databricks.sdk.service import ml
from databricks.sdk.service import sql

# Environment variables
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # format: workspace.<run-id>

# Table configuration
FEATURE_TABLE_NAME = "transactions4adadd"
FEATURE_TABLE_FULL_NAME = f"{SCHEMA}.{FEATURE_TABLE_NAME}"
ONLINE_TABLE_NAME = f"{PREFIX}_{FEATURE_TABLE_NAME}_online"
VERSION = 1
RECORD_KEY = "row_id"
EVENT_TIME_COL = "event_time"

# Source data files
DATA_FILES = [
    "data/transactions_export_1.csv",
    "data/transactions_export_2.csv"
]

def main():
    # Initialize client
    client = WorkspaceClient()
    
    # Get a SQL warehouse
    warehouses = list(client.warehouses.list())
    if not warehouses:
        raise Exception("No SQL warehouses available")
    
    warehouse_id = warehouses[0].id
    print(f"Using SQL warehouse: {warehouse_id}")
    
    # Start the warehouse if it's stopped
    if warehouses[0].state == "STOPPED":
        print("Starting SQL warehouse...")
        client.warehouses.start_and_wait(warehouse_id=warehouse_id)
    
    print(f"Creating feature table {FEATURE_TABLE_FULL_NAME} version {VERSION}")
    
    # Step 1: Create a volume for the data files
    print("Creating volume for data files...")
    volume_name = f"{PREFIX}_data_volume"
    volume_full_name = f"{SCHEMA}.{volume_name}"
    
    try:
        client.volumes.create(
            catalog_name=SCHEMA.split('.')[0],
            schema_name=SCHEMA.split('.')[1],
            name=volume_name,
            volume_type=catalog.VolumeType.MANAGED
        )
    except Exception as e:
        print(f"Volume may already exist: {e}")
    
    # Step 2: Upload data files to the volume
    print("Uploading data files to volume...")
    for data_file in DATA_FILES:
        file_name = os.path.basename(data_file)
        
        with open(data_file, 'rb') as f:
            client.files.upload(
                file_path=f"/Volumes/{SCHEMA.split('.')[0]}/{SCHEMA.split('.')[1]}/{volume_name}/{file_name}",
                contents=f,
                overwrite=True
            )
    
    # Step 3: Create a temporary table with the combined data
    print("Creating temporary table with combined data...")
    
    # Create SQL to create a temporary table and load data from both files in the volume
    temp_table_name = f"temp_{FEATURE_TABLE_NAME}_load"
    
    # The correct format for volume paths in SQL is: /Volumes/<catalog>/<schema>/<volume>/<file>
    volume_data_path = f"/Volumes/{SCHEMA.split('.')[0]}/{SCHEMA.split('.')[1]}/{volume_name}"
    
    create_temp_table_sql = f"""
    CREATE OR REPLACE TEMPORARY VIEW {temp_table_name}
    USING CSV
    OPTIONS (
        path = "{volume_data_path}",
        header = "true",
        inferSchema = "true",
        mergeSchema = "true",
        recursiveFileLookup = "true"
    )
    """
    
    # Execute the SQL to create the temporary table
    temp_response = client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        catalog=SCHEMA.split('.')[0],
        schema=SCHEMA.split('.')[1],
        statement=create_temp_table_sql
    )
    
    # Wait for completion
    while temp_response.status.state not in [sql.StatementState.SUCCEEDED, sql.StatementState.FAILED, sql.StatementState.CANCELED]:
        temp_response = client.statement_execution.get_statement(temp_response.statement_id)
        time.sleep(1)
    
    if temp_response.status.state != sql.StatementState.SUCCEEDED:
        print(f"Temporary table creation failed: {temp_response.status}")
        return
    
    # Step 2: Create the target table with deduplication (since files overlap)
    print("Creating target table with deduplicated data...")
    create_target_table_sql = f"""
    CREATE OR REPLACE TABLE {FEATURE_TABLE_FULL_NAME} AS
    SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY {RECORD_KEY} ORDER BY {EVENT_TIME_COL} DESC) as rn
        FROM {temp_table_name}
    )
    WHERE rn = 1
    """
    
    # Execute the SQL to create the target table
    target_response = client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        catalog=SCHEMA.split('.')[0],
        schema=SCHEMA.split('.')[1],
        statement=create_target_table_sql
    )
    
    # Wait for completion
    while target_response.status.state not in [sql.StatementState.SUCCEEDED, sql.StatementState.FAILED, sql.StatementState.CANCELED]:
        target_response = client.statement_execution.get_statement(target_response.statement_id)
        time.sleep(1)
    
    if target_response.status.state != sql.StatementState.SUCCEEDED:
        print(f"Target table creation failed: {target_response.status}")
        return
    
    # Step 3: Register the table as a feature table
    print("Registering feature table...")
    
    # Publish the table as a feature table
    publish_response = client.feature_store.publish_table(
        source_table_name=FEATURE_TABLE_FULL_NAME,
        publish_spec=ml.PublishSpec(
            online_store=SCHEMA.split('.')[0],  # catalog
            online_table_name=ONLINE_TABLE_NAME,
            publish_mode=ml.PublishSpecPublishMode.SNAPSHOT
        )
    )
    
    print(f"Feature table published: {publish_response}")
    
    # Step 4: Create online table for low-latency access
    print("Creating online table for low-latency access...")
    
    online_table = client.online_tables.create(
        table=catalog.OnlineTable(
            name=f"{SCHEMA}.{ONLINE_TABLE_NAME}",
            spec=catalog.OnlineTableSpec(
                source_table_full_name=FEATURE_TABLE_FULL_NAME,
                primary_key_columns=[RECORD_KEY],
                timeseries_key=EVENT_TIME_COL
            )
        )
    )
    
    print(f"Online table created: {online_table}")
    
    print("Feature table registration complete!")
    print(f"Feature table: {FEATURE_TABLE_FULL_NAME}")
    print(f"Online table: {SCHEMA}.{ONLINE_TABLE_NAME}")

if __name__ == "__main__":
    main()