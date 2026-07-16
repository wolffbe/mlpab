#!/usr/bin/env python3
"""
Test simple CSV table creation.
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
    while time.time() - start_time < 60:  # 60 second timeout
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
    print("Testing simple CSV table creation...")
    
    # Create a simple table from CSV
    execute_sql(f"""
    CREATE OR REPLACE TABLE workspace.mlpabf21a49.test_csv_table
    USING CSV
    OPTIONS (
        path '{WORKSPACE_PATH}/transactions.csv',
        header 'true',
        inferSchema 'true'
    )
    """)
    print("Test CSV table created!")
    
    # Verify table
    result = execute_sql(f"SELECT COUNT(*) as count FROM workspace.mlpabf21a49.test_csv_table")
    statement_result = wc.statement_execution.get_statement_result_chunk_n(
        result.statement_id, 1
    )
    print(f"Test table row count: {statement_result.result.data_array}")

if __name__ == "__main__":
    main()
