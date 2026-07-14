#!/usr/bin/env python3
"""
Create raw tables from CSV files.
"""
import os
import time
from databricks.sdk import WorkspaceClient

# Configuration
SCHEMA = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpabf21a49')
PREFIX = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpabf21a49')

wc = WorkspaceClient()
WAREHOUSE_ID = 'a832b544eb7dc3fe'
WORKSPACE_PATH = f'/Workspace/Users/benedict@hopsworks.ai/{PREFIX}'

def execute_sql(statement, warehouse_id=WAREHOUSE_ID):
    """Execute a SQL statement and return the result."""
    print(f"Executing SQL: {statement[:100]}...")
    result = wc.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=warehouse_id,
        wait_timeout="10s"
    )
    
    # Wait for completion
    while True:
        status = wc.statement_execution.get_statement(result.statement_id)
        if status.status.state in ['SUCCEEDED', 'FAILED', 'CANCELED']:
            if status.status.state == 'FAILED':
                error_msg = getattr(status.status, 'error', 'Unknown error')
                print(f"SQL execution failed: {error_msg}")
                raise Exception(f"SQL execution failed: {error_msg}")
            return status
        time.sleep(1)

def main():
    print("Creating raw tables...")
    
    # Create transactions table
    execute_sql(f"""
    CREATE OR REPLACE TABLE {SCHEMA}.raw_transactions
    USING CSV
    OPTIONS (
        path '{WORKSPACE_PATH}/transactions.csv',
        header 'true',
        inferSchema 'true'
    )
    """)
    print("Raw transactions table created!")
    
    # Create score transactions table
    execute_sql(f"""
    CREATE OR REPLACE TABLE {SCHEMA}.raw_score_transactions
    USING CSV
    OPTIONS (
        path '{WORKSPACE_PATH}/score_transactions.csv',
        header 'true',
        inferSchema 'true'
    )
    """)
    print("Raw score transactions table created!")
    
    # Verify tables
    result = execute_sql(f"SELECT COUNT(*) as count FROM {SCHEMA}.raw_transactions")
    statement_result = wc.statement_execution.get_statement_result_chunk_n(
        result.statement_id, 1
    )
    print(f"Transactions table row count: {statement_result.result.data_array}")
    
    result = execute_sql(f"SELECT COUNT(*) as count FROM {SCHEMA}.raw_score_transactions")
    statement_result = wc.statement_execution.get_statement_result_chunk_n(
        result.statement_id, 1
    )
    print(f"Score transactions table row count: {statement_result.result.data_array}")

if __name__ == "__main__":
    main()
