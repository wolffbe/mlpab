#!/usr/bin/env python3
from databricks.sdk import WorkspaceClient
import time

wc = WorkspaceClient()

# Try to create a table from workspace CSV using different approaches
approaches = [
    "SELECT * FROM CSV.`/Users/benedict@hopsworks.ai/mlpabf21a49/transactions.csv`",
    "SELECT * FROM CSV.`file:/Users/benedict@hopsworks.ai/mlpabf21a49/transactions.csv`",
    "SELECT * FROM CSV.`/Workspace/Users/benedict@hopsworks.ai/mlpabf21a49/transactions.csv`",
]

for i, approach in enumerate(approaches):
    try:
        result = wc.statement_execution.execute_statement(
            statement=approach,
            warehouse_id='a832b544eb7dc3fe',
            wait_timeout='10s'
        )
        print(f'Approach {i+1} started: {result.statement_id}')
        
        # Wait and check status
        time.sleep(5)
        status = wc.statement_execution.get_statement(result.statement_id)
        print(f'Approach {i+1} status: {status.status.state}')
        
        if status.status.state == 'FAILED':
            print(f'Approach {i+1} error: {status.status.error}')
        elif status.status.state == 'SUCCEEDED':
            print(f'Approach {i+1} succeeded!')
            break
    except Exception as e:
        print(f'Approach {i+1} failed: {e}')
