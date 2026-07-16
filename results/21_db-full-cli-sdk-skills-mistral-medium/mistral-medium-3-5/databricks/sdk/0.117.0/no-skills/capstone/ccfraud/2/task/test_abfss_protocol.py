#!/usr/bin/env python3
"""
Test ABFSS protocol.
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
    print("Testing ABFSS protocol...")
    
    # Try ABFSS protocol (Azure Blob File System)
    try:
        execute_sql("""
        CREATE OR REPLACE TABLE workspace.mlpabf21a49.test_abfss
        USING CSV
        OPTIONS (
            path 'abfss://workspace@dbstorage-prod-ymoo7.dfs.core.windows.net/Users/benedict@hopsworks.ai/mlpabf21a49/transactions.csv',
            header 'true',
            inferSchema 'true'
        )
        """)
        print("ABFSS protocol succeeded!")
        
        # Verify
        result = execute_sql(f"SELECT COUNT(*) as count FROM workspace.mlpabf21a49.test_abfss")
        statement_result = wc.statement_execution.get_statement_result_chunk_n(
            result.statement_id, 1
        )
        print(f"Row count: {statement_result.result.data_array}")
    except Exception as e:
        print(f"ABFSS protocol failed: {e}")

if __name__ == "__main__":
    main()
