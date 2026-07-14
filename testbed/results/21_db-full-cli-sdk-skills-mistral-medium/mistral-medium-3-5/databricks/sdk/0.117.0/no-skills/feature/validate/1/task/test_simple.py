#!/usr/bin/env python3

import os
from databricks.sdk import WorkspaceClient

def main():
    warehouse_id = '4dfab06c923fe3cc'
    ws = WorkspaceClient()
    
    # Drop table first
    try:
        ws.tables.delete('workspace.mlpab0e0197.eventsa45e2a')
        print('Table dropped')
    except:
        pass
    
    # Create table
    create_sql = 'CREATE TABLE IF NOT EXISTS workspace.mlpab0e0197.eventsa45e2a (row_id STRING, account_id STRING, event_time BIGINT, amount DOUBLE, category STRING) USING DELTA'
    result = ws.statement_execution.execute_statement(
        statement=create_sql,
        warehouse_id=warehouse_id,
        wait_timeout='30s'
    )
    print(f'Table created: {result.status.state}')
    
    # Try insert
    insert_sql = 'INSERT INTO workspace.mlpab0e0197.eventsa45e2a SELECT "R00999" as row_id, "A9999" as account_id, 1234567890000 as event_time, 100.0 as amount, "other" as category'
    result = ws.statement_execution.execute_statement(
        statement=insert_sql,
        warehouse_id=warehouse_id,
        wait_timeout='30s'
    )
    print(f'Insert result: {result.status.state}')
    print(f'Error: {result.status.error}')

if __name__ == '__main__':
    main()