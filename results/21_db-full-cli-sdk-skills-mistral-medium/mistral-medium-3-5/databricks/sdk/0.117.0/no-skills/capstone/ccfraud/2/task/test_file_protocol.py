#!/usr/bin/env python3
"""
Test file protocol for CSV reading.
"""
import time
from databricks.sdk import WorkspaceClient

wc = WorkspaceClient()
WAREHOUSE_ID = 'a832b544eb7dc3fe'

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
    print("Testing file protocol...")
    
    # Try different file protocols
    protocols = [
        "file:/Workspace/Users/benedict@hopsworks.ai/mlpabf21a49/transactions.csv",
        "/Workspace/Users/benedict@hopsworks.ai/mlpabf21a49/transactions.csv",
        "file:///Workspace/Users/benedict@hopsworks.ai/mlpabf21a49/transactions.csv",
    ]
    
    for i, path in enumerate(protocols):
        try:
            execute_sql(f"""
            CREATE OR REPLACE TABLE workspace.mlpabf21a49.test_file_{i}
            USING CSV
            OPTIONS (
                path '{path}',
                header 'true',
                inferSchema 'true'
            )
            """)
            print(f"Protocol {i+1} succeeded!")
            
            # Verify
            result = execute_sql(f"SELECT COUNT(*) as count FROM workspace.mlpabf21a49.test_file_{i}")
            statement_result = wc.statement_execution.get_statement_result_chunk_n(
                result.statement_id, 1
            )
            print(f"Protocol {i+1} row count: {statement_result.result.data_array}")
            break
        except Exception as e:
            print(f"Protocol {i+1} failed: {e}")

if __name__ == "__main__":
    main()
