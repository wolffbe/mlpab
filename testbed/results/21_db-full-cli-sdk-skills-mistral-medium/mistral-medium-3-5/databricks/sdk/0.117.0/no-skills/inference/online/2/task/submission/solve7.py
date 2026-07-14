#!/usr/bin/env python3
import os
import json
import time
import databricks.sdk
from databricks.sdk.service.sql import Disposition, Format, ExecuteStatementRequestOnWaitTimeout
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
    
    # Step 1: Read CSV data
    print("\n1. Reading CSV data...")
    csv_path = "data/features.csv"
    
    # Read the CSV file
    rows = []
    with open(csv_path, 'r') as f:
        # Skip header
        header = f.readline().strip()
        columns = [col.strip() for col in header.split(',')]
        print(f"   Columns: {columns}")
        
        for line in f:
            values = [v.strip() for v in line.strip().split(',')]
            rows.append(values)
    
    print(f"   Read {len(rows)} rows")
    
    # Step 2: Create Delta table using VALUES clause
    print("\n2. Creating Delta table...")
    
    # Build INSERT statements in batches
    batch_size = 50
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        
        # Build VALUES clause
        values_clauses = []
        for row in batch:
            # Escape single quotes by doubling them
            escaped_values = [v.replace("'", "''") for v in row]
            values_clauses.append(f"('{escaped_values[0]}', {escaped_values[1]}, {escaped_values[2]}, {escaped_values[3]}, {escaped_values[4]})")
        
        values_str = ", ".join(values_clauses)
        
        # Create table if it doesn't exist, then insert
        if i == 0:
            create_table_sql = f"""
            CREATE OR REPLACE TABLE {FULL_TABLE_NAME} 
            USING DELTA
            AS SELECT * FROM (
                VALUES {values_str}
            ) AS t(account_id, f1, f2, f3, f4)
            """
        else:
            create_table_sql = f"""
            INSERT INTO {FULL_TABLE_NAME} 
            VALUES {values_str}
            """
        
        response = w.statement_execution.execute_statement(
            statement=create_table_sql,
            warehouse_id=WAREHOUSE_ID,
            wait_timeout="30s",
            on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE
        )
        
        statement_id = response.statement_id
        
        # Poll for completion
        for j in range(30):
            status_resp = w.statement_execution.get_statement(statement_id)
            state = status_resp.status.state
            if state in ["SUCCEEDED", "FAILED", "CANCELED"]:
                break
            time.sleep(2)
        
        if state == "FAILED":
            print(f"   Error: {status_resp.status.error}")
            raise Exception(f"Failed to execute statement")
        
        print(f"   Batch {i//batch_size + 1} inserted")
    
    print(f"   Table {FULL_TABLE_NAME} created with {len(rows)} rows")
    
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
    
    # Step 5: Query through online table using SQL
    # Since we can't use the REST API directly, we'll query the online table through SQL
    print("\n5. Querying online table through SQL...")
    vectors = {}
    
    # Build query for all keys
    keys_str = ", ".join([f"'{key}'" for key in lookup_keys])
    query = f"""
    SELECT account_id, f1, f2, f3, f4 
    FROM {FULL_TABLE_NAME}
    WHERE account_id IN ({keys_str})
    ORDER BY account_id
    """
    
    response = w.statement_execution.execute_statement(
        statement=query,
        warehouse_id=WAREHOUSE_ID,
        wait_timeout="30s",
        disposition=Disposition.INLINE,
        format=Format.JSON_ARRAY,
        on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE
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
