#!/usr/bin/env python3
"""
Test CSV without inferSchema.
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
    print("Testing CSV without inferSchema...")
    
    # Try without inferSchema
    try:
        execute_sql(f"""
        CREATE OR REPLACE TABLE workspace.mlpabf21a49.test_no_infer
        USING CSV
        OPTIONS (
            path '{WORKSPACE_PATH}/transactions.csv',
            header 'true'
        )
        """)
        print("CSV without inferSchema succeeded!")
        
        # Verify
        result = execute_sql(f"SELECT COUNT(*) as count FROM workspace.mlpabf21a49.test_no_infer")
        statement_result = wc.statement_execution.get_statement_result_chunk_n(
            result.statement_id, 1
        )
        print(f"Row count: {statement_result.result.data_array}")
    except Exception as e:
        print(f"CSV without inferSchema failed: {e}")

if __name__ == "__main__":
    main()
