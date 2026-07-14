#!/usr/bin/env python3
import os
import json
import time
import requests
import databricks.sdk
from databricks.sdk.service.sql import Disposition, Format
from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec

# Initialize the workspace client
w = databricks.sdk.WorkspaceClient()

# Environment variables
SCHEMA = os.environ['MLPAB_DATABRICKS_SCHEMA']
PREFIX = os.environ['MLPAB_DATABRICKS_PREFIX']

# Table configuration
TABLE_NAME = "profilesd1eca7"
FULL_TABLE_NAME = f"{SCHEMA}.{TABLE_NAME}"

# Warehouse to use
WAREHOUSE_ID = "8a93fc195da2ceb1"  # mlpab-grader

def main():
    print("Starting workflow...")
    
    # Step 1: Upload CSV to workspace using files API
    print("\n1. Uploading CSV to workspace...")
    csv_path = "data/features.csv"
    workspace_dir = f"/Users/{w.current_user.me().emails[0].value}/{PREFIX}"
    workspace_file = f"{workspace_dir}/features.csv"
    
    # Create directory first
    try:
        w.files.create_directory(workspace_dir)
        print(f"   Created directory: {workspace_dir}")
    except Exception as e:
        print(f"   Directory may already exist: {e}")
    
    # Upload file
    with open(csv_path, 'rb') as f:
        w.files.upload(workspace_file, f, overwrite=True)
    print(f"   Uploaded to {workspace_file}")
    
    # Step 2: Create Delta table from CSV
    print("\n2. Creating Delta table...")
    # Use the workspace file path with file: protocol
    create_table_sql = f"""
    CREATE OR REPLACE TABLE {FULL_TABLE_NAME} 
    USING DELTA
    AS SELECT * FROM csv.`{workspace_file}`
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
    
    # Step 3: Create online table
    print("\n3. Creating online table...")
    online_table = OnlineTable(
        name=FULL_TABLE_NAME,
        spec=OnlineTableSpec(
            source_table_full_name=FULL_TABLE_NAME,
            primary_key_columns=["account_id"],
            perform_full_copy=True
        )
    )
    
    create_op = w.online_tables.create(online_table)
    print(f"   Created online table: {FULL_TABLE_NAME}")
    
    # Wait for it to be active
    print(f"   Waiting for online table to become active...")
    online_table_info = None
    for i in range(60):
        try:
            online_table_info = w.online_tables.get(FULL_TABLE_NAME)
            if online_table_info and online_table_info.table_serving_url:
                print(f"   Online table is ready at: {online_table_info.table_serving_url}")
                break
        except Exception as e:
            pass
        time.sleep(5)
    
    if not online_table_info or not online_table_info.table_serving_url:
        raise Exception("Online table did not become active")
    
    # Step 4: Read lookup keys
    print("\n4. Reading lookup keys...")
    with open("data/lookup_keys.txt", 'r') as f:
        lookup_keys = [line.strip() for line in f.readlines()]
    
    print(f"   Found {len(lookup_keys)} keys: {lookup_keys}")
    
    # Step 5: Query through online table's REST API
    print("\n5. Querying online table through REST API...")
    vectors = {}
    
    # Get the host and token
    host = w.config.host
    token = w.config.token
    
    # The online table serving URL should be used
    # Format: /api/2.1/online-tables/{full_name}/get
    full_name_encoded = FULL_TABLE_NAME.replace('.', '%2E')
    
    # Query each key individually through the online table
    for key in lookup_keys:
        url = f"{host}/api/2.1/online-tables/{full_name_encoded}/get"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        params = {
            "keys": key,
            "requested_columns": "f1,f2,f3,f4"
        }
        
        print(f"   Querying key: {key}")
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {data}")
            # Process the response
            if 'rows' in data and len(data['rows']) > 0:
                row = data['rows'][0]
                # The row should contain the feature values
                # The format might be: [account_id, f1, f2, f3, f4]
                if len(row) >= 5:
                    account_id = row[0]
                    f1, f2, f3, f4 = float(row[1]), float(row[2]), float(row[3]), float(row[4])
                    vectors[account_id] = [f1, f2, f3, f4]
                else:
                    print(f"   Warning: Unexpected row format: {row}")
        else:
            print(f"   Error for key {key}: {response.status_code} - {response.text}")
    
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
