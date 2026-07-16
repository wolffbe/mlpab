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
from databricks.sdk.service import sql
from databricks.sdk.service import ml

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
    
    # Step 3: Create the target table with deduplicated data
    print("Creating target table with deduplicated data...")
    
    # Create SQL to create the table directly from the volume files
    volume_data_path = f"/Volumes/{SCHEMA.split('.')[0]}/{SCHEMA.split('.')[1]}/{volume_name}"
    
    # First create the table
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {FEATURE_TABLE_FULL_NAME} (
        row_id STRING,
        account_id STRING,
        event_time BIGINT,
        amount DOUBLE,
        category STRING
    )
    USING DELTA
    """
    
    # Execute the create table statement
    response = client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        catalog=SCHEMA.split('.')[0],
        schema=SCHEMA.split('.')[1],
        statement=create_table_sql
    )
    
    # Wait for completion
    while response.status.state not in [sql.StatementState.SUCCEEDED, sql.StatementState.FAILED, sql.StatementState.CANCELED]:
        response = client.statement_execution.get_statement(response.statement_id)
        time.sleep(1)
    
    if response.status.state != sql.StatementState.SUCCEEDED:
        print(f"Table creation failed: {response.status}")
        return
    
    # Then insert deduplicated data
    insert_sql = f"""
    INSERT INTO {FEATURE_TABLE_FULL_NAME}
    SELECT row_id, account_id, event_time, amount, category FROM (
        SELECT
            _c0 as row_id,
            _c1 as account_id,
            CAST(_c2 as BIGINT) as event_time,
            CAST(_c3 as DOUBLE) as amount,
            _c4 as category,
            ROW_NUMBER() OVER (PARTITION BY _c0 ORDER BY CAST(_c2 as BIGINT) DESC) as rn
        FROM csv.`{volume_data_path}`
        WHERE _c0 != 'row_id'
    )
    WHERE rn = 1
    """
    
    # Execute the create table statement
    response = client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        catalog=SCHEMA.split('.')[0],
        schema=SCHEMA.split('.')[1],
        statement=create_table_sql
    )
    
    # Wait for completion
    while response.status.state not in [sql.StatementState.SUCCEEDED, sql.StatementState.FAILED, sql.StatementState.CANCELED]:
        response = client.statement_execution.get_statement(response.statement_id)
        time.sleep(1)
    
    if response.status.state != sql.StatementState.SUCCEEDED:
        print(f"Table creation failed: {response.status}")
        return
    
    # Execute the insert statement
    response = client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        catalog=SCHEMA.split('.')[0],
        schema=SCHEMA.split('.')[1],
        statement=insert_sql
    )
    
    # Wait for completion
    while response.status.state not in [sql.StatementState.SUCCEEDED, sql.StatementState.FAILED, sql.StatementState.CANCELED]:
        response = client.statement_execution.get_statement(response.statement_id)
        time.sleep(1)
    
    if response.status.state != sql.StatementState.SUCCEEDED:
        print(f"Data insertion failed: {response.status}")
        return
    
    print(f"Table created successfully: {FEATURE_TABLE_FULL_NAME}")
    
    # Step 4: Enable online access using feature store publish_table
    print("Enabling online access for low-latency lookup...")
    
    # The catalog name is "workspace"
    catalog_name = "workspace"
    
    publish_response = client.feature_store.publish_table(
        source_table_name=FEATURE_TABLE_FULL_NAME,
        publish_spec=ml.PublishSpec(
            online_store=catalog_name,
            online_table_name=ONLINE_TABLE_NAME,
            publish_mode=ml.PublishSpecPublishMode.SNAPSHOT
        )
    )
    
    print(f"Online access enabled: {publish_response}")
    
    print("Feature table registration complete!")
    print(f"Feature table: {FEATURE_TABLE_FULL_NAME}")
    print(f"Online table: {SCHEMA}.{ONLINE_TABLE_NAME}")

if __name__ == "__main__":
    main()