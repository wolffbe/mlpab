#!/usr/bin/env python3
"""
Test with a tiny CSV file.
"""
import time
from databricks.sdk import WorkspaceClient

wc = WorkspaceClient()
WAREHOUSE_ID = 'a832b544eb7dc3fe'
WORKSPACE_PATH = '/Workspace/Users/benedict@hopsworks.ai/mlpabf21a49'

def execute_sql(statement, warehouse_id=WAREHOUSE_ID):
    """Execute a SQL statement and return the result."""
    print(f"Executing SQL: {statement[:100]}...")
    result = wc.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=warehouse_id,
        wait_timeout="10s"
    )
    
    # Wait for completion with timeout
    start_time = time.time()
    while time.time() - start_time < 30:  # 30 second timeout
        status = wc.statement_execution.get_statement(result.statement_id)
        if status.status.state in ['SUCCEEDED', 'FAILED', 'CANCELED']:
            if status.status.state == 'FAILED':
                error_msg = getattr(status.status, 'error', 'Unknown error')
                print(f"SQL execution failed: {error_msg}")
                raise Exception(f"SQL execution failed: {error_msg}")
            return status
        time.sleep(2)
    
    raise Exception("SQL execution timed out")

def main():
    print("Testing with tiny CSV...")
    
    # First, create a tiny CSV file in workspace
    tiny_csv_content = """transaction_id,cc_num,datetime,amount,merchant,category,lat,long,is_fraud
T000000001,420200000267,2025-09-01T00:08:00Z,33.79,m_476,travel,47.0249,-82.3922,0
T000000002,420200000267,2025-09-01T00:18:00Z,45.23,m_476,travel,47.0249,-82.3922,0
T000000003,493100000068,2025-09-01T00:09:00Z,64.37,m_438,fuel,34.716,-91.2607,0"""
    
    from databricks.sdk.service.workspace import ImportFormat
    wc.workspace.upload(f'{WORKSPACE_PATH}/tiny_test.csv', tiny_csv_content.encode('utf-8'), format=ImportFormat.AUTO, overwrite=True)
    print("Tiny CSV uploaded")
    
    # Create table from tiny CSV
    execute_sql(f"""
    CREATE OR REPLACE TABLE workspace.mlpabf21a49.test_tiny_table
    USING CSV
    OPTIONS (
        path '{WORKSPACE_PATH}/tiny_test.csv',
        header 'true',
        inferSchema 'true'
    )
    """)
    print("Tiny test table created!")
    
    # Verify table
    result = execute_sql(f"SELECT COUNT(*) as count FROM workspace.mlpabf21a49.test_tiny_table")
    statement_result = wc.statement_execution.get_statement_result_chunk_n(
        result.statement_id, 1
    )
    print(f"Tiny table row count: {statement_result.result.data_array}")

if __name__ == "__main__":
    main()
