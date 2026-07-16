#!/usr/bin/env python3
import os
import json
import time
import databricks.sdk
from databricks.sdk.service.sql import Disposition, Format
from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode

# Initialize the workspace client
w = databricks.sdk.WorkspaceClient()

# Environment variables
SCHEMA = os.environ['MLPAB_DATABRICKS_SCHEMA']
PREFIX = os.environ['MLPAB_DATABRICKS_PREFIX']

# Table configuration
TABLE_NAME = "profilesd1eca7"
FULL_TABLE_NAME = f"{SCHEMA}.{TABLE_NAME}"
ONLINE_TABLE_NAME = f"{SCHEMA}.{TABLE_NAME}_v1"  # Version 1

# Warehouse to use
WAREHOUSE_ID = "8a93fc195da2ceb1"  # mlpab-grader

def main():
    print("Starting workflow...")
    
    # Step 1: Upload CSV to DBFS
    print("\n1. Uploading CSV to DBFS...")
    csv_path = "data/features.csv"
    dbfs_path = f"/FileStore/{PREFIX}/features.csv"
    
    with open(csv_path, 'r') as f:
        csv_content = f.read()
    
    w.dbfs.put(dbfs_path, csv_content, overwrite=True)
    print(f"   Uploaded to {dbfs_path}")
    
    # Step 2: Create Delta table from CSV
    print("\n2. Creating Delta table...")
    create_table_sql = f"""
    CREATE OR REPLACE TABLE {FULL_TABLE_NAME} 
    USING DELTA
    AS SELECT * FROM csv.`dbfs:{dbfs_path}`
    """
    
    response = w.statement_execution.execute_statement(
        statement=create_table_sql,
        warehouse_id=WAREHOUSE_ID,
        wait_timeout="30s",
        on_wait_timeout="CONTINUE"
    )
    
    statement_id = response.statement_id
    print(f"   Statement ID: {statement_id}")
    
    # Poll for completion
    for i in range(30):
        status_resp = w.statement_execution.get_statement(statement_id)
        state = status_resp.status.state
        if state in ["SUCCEEDED", "FAILED", "CANCELED"]:
            break
        time.sleep(2)
    
    if state == "FAILED":
        print(f"   Error: {status_resp.status.error}")
        raise Exception(f"Failed to create table")
    
    print(f"   Table {FULL_TABLE_NAME} created")
    
    # Step 3: Publish as online feature table
    print("\n3. Publishing as online feature table...")
    
    # First, we need to create an online store
    online_store_name = f"{SCHEMA}_store"
    
    # Check if online store exists
    try:
        online_store = w.feature_store.get_online_store(online_store_name)
        print(f"   Using existing online store: {online_store_name}")
    except Exception:
        # Create online store
        from databricks.sdk.service.ml import OnlineStore
        online_store = OnlineStore(
            name=online_store_name,
            capacity="SMALL"
        )
        online_store = w.feature_store.create_online_store(online_store)
        print(f"   Created online store: {online_store_name}")
    
    # Publish the table
    publish_spec = PublishSpec(
        online_store=online_store_name,
        online_table_name=ONLINE_TABLE_NAME,
        publish_mode=PublishSpecPublishMode.SNAPSHOT
    )
    
    publish_response = w.feature_store.publish_table(
        source_table_name=FULL_TABLE_NAME,
        publish_spec=publish_spec
    )
    
    print(f"   Published table to online store")
    
    # Wait for the online table to be ready
    print(f"   Waiting for online table to be ready...")
    for i in range(60):
        try:
            # Check if online table exists
            online_table = w.online_tables.get(ONLINE_TABLE_NAME)
            if online_table and online_table.table_serving_url:
                print(f"   Online table is ready at: {online_table.table_serving_url}")
                break
        except Exception as e:
            pass
        time.sleep(5)
    
    # Step 4: Read lookup keys
    print("\n4. Reading lookup keys...")
    with open("data/lookup_keys.txt", 'r') as f:
        lookup_keys = [line.strip() for line in f.readlines()]
    
    print(f"   Found {len(lookup_keys)} keys")
    
    # Step 5: Query through online table
    print("\n5. Querying online table...")
    vectors = {}
    
    # Query the online table using SQL
    keys_str = ", ".join([f"'{key}'" for key in lookup_keys])
    query = f"""
    SELECT account_id, f1, f2, f3, f4 
    FROM {ONLINE_TABLE_NAME}
    WHERE account_id IN ({keys_str})
    """
    
    response = w.statement_execution.execute_statement(
        statement=query,
        warehouse_id=WAREHOUSE_ID,
        wait_timeout="30s",
        disposition=Disposition.INLINE,
        format=Format.JSON_ARRAY,
        on_wait_timeout="CONTINUE"
    )
    
    statement_id = response.statement_id
    print(f"   Query statement ID: {statement_id}")
    
    # Wait for results
    for i in range(30):
        result_resp = w.statement_execution.get_statement(statement_id)
        state = result_resp.status.state
        if state in ["SUCCEEDED", "FAILED", "CANCELED"]:
            break
        time.sleep(2)
    
    if state == "FAILED":
        print(f"   Error: {result_resp.status.error}")
        raise Exception(f"Query failed")
    
    # Process results
    if result_resp.result and result_resp.result.data_array:
        result_data = json.loads(result_resp.result.data_array)
        print(f"   Retrieved {len(result_data)} rows")
        
        for row in result_data:
            account_id = row[0]
            f1, f2, f3, f4 = float(row[1]), float(row[2]), float(row[3]), float(row[4])
            vectors[account_id] = [f1, f2, f3, f4]
    
    print(f"   Processed {len(vectors)} vectors")
    
    # Step 6: Write results
    print("\n6. Writing results...")
    output = {"vectors": vectors}
    
    with open("submission/answers.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"   Written to submission/answers.json")
    print("\nDone!")

if __name__ == "__main__":
    main()
