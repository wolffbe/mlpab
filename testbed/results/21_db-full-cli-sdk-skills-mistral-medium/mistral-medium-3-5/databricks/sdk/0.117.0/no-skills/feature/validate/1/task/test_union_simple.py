#!/usr/bin/env python3

import csv
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
    
    # Read just 5 rows
    valid_rows = []
    with open('data/events.csv', 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 5:
                break
            valid_rows.append(row)
    
    print(f'Read {len(valid_rows)} rows')
    
    # Build UNION ALL SQL
    select_parts = []
    for row in valid_rows:
        row_id = row['row_id']
        account_id = row['account_id']
        event_time = row['event_time']
        amount = row['amount'].strip()
        category = row['category'].strip()
        
        select_parts.append(f'SELECT "{row_id}" as row_id, "{account_id}" as account_id, {event_time} as event_time, {amount} as amount, "{category}" as category')
    
    union_sql = ' UNION ALL '.join(select_parts)
    insert_sql = f'INSERT INTO workspace.mlpab0e0197.eventsa45e2a {union_sql}'
    
    print(f'SQL: {insert_sql}')
    
    result = ws.statement_execution.execute_statement(
        statement=insert_sql,
        warehouse_id=warehouse_id,
        wait_timeout='30s'
    )
    print(f'Batch insert result: {result.status.state}')
    print(f'Error: {result.status.error}')

if __name__ == '__main__':
    main()