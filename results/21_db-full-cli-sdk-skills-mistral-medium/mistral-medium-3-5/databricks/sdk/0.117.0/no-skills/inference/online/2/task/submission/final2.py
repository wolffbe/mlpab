#!/usr/bin/env python3
import os
import json
import time
import databricks.sdk
from databricks.sdk.service.sql import ExecuteStatementRequestOnWaitTimeout

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

def execute_sql(statement, warehouse_id=WAREHOUSE_ID, wait_timeout="50s"):
    """Helper to execute SQL and wait for completion"""
    response = w.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=warehouse_id,
        wait_timeout=wait_timeout,
        on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE
    )
    
    statement_id = response.statement_id
    
    # Poll for completion
    for i in range(50):
        status_resp = w.statement_execution.get_statement(statement_id)
        state = status_resp.status.state
        if state in ["SUCCEEDED", "FAILED", "CANCELED"]:
            if state == "FAILED":
                raise Exception(f"SQL failed: {status_resp.status.error}")
            return status_resp
        time.sleep(2)
    
    raise Exception("SQL execution timed out")

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
    
    # Step 2: Delete existing table if it exists
    print("\n2. Deleting existing table...")
    try:
        w.tables.delete(FULL_TABLE_NAME)
        print(f"   Deleted table {FULL_TABLE_NAME}")
    except Exception as e:
        print(f"   Table may not exist: {e}")
    
    # Step 3: Create Delta table
    print("\n3. Creating Delta table...")
    
    # First create an empty table
    create_table_sql = f"""
    CREATE OR REPLACE TABLE {FULL_TABLE_NAME} 
    (account_id STRING, f1 DOUBLE, f2 DOUBLE, f3 DOUBLE, f4 DOUBLE)
    USING DELTA
    """
    
    result = execute_sql(create_table_sql)
    print(f"   Table {FULL_TABLE_NAME} created")
    
    # Then insert data in batches
    batch_size = 20
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        
        # Build VALUES clause
        values_clauses = []
        for row in batch:
            # Escape single quotes by doubling them
            escaped_values = [v.replace("'", "''") for v in row]
            values_clauses.append(f"('{escaped_values[0]}', {escaped_values[1]}, {escaped_values[2]}, {escaped_values[3]}, {escaped_values[4]})")
        
        values_str = ", ".join(values_clauses)
        
        insert_sql = f"""
        INSERT INTO {FULL_TABLE_NAME} 
        VALUES {values_str}
        """
        
        result = execute_sql(insert_sql)
        print(f"   Batch {i//batch_size + 1} inserted ({len(batch)} rows)")
    
    print(f"   Table {FULL_TABLE_NAME} populated with {len(rows)} rows")
    
    # Step 4: Read lookup keys
    print("\n4. Reading lookup keys...")
    with open("data/lookup_keys.txt", 'r') as f:
        lookup_keys = [line.strip() for line in f.readlines()]
    
    print(f"   Found {len(lookup_keys)} keys: {lookup_keys}")
    
    # Step 5: Query the table for the lookup keys
    print("\n5. Querying table...")
    vectors = {}
    
    # Build query for all keys
    keys_str = ", ".join([f"'{key}'" for key in lookup_keys])
    query = f"""
    SELECT account_id, f1, f2, f3, f4 
    FROM {FULL_TABLE_NAME}
    WHERE account_id IN ({keys_str})
    ORDER BY account_id
    """
    
    result = execute_sql(query, wait_timeout="50s")
    
    # Process results
    if result.result and result.result.data_array:
        import json
        result_data = json.loads(result.result.data_array)
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
