#!/usr/bin/env python3

import csv
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
    
    # Read and filter the CSV data
    valid_categories = {'grocery', 'travel', 'salary', 'rent', 'other'}
    valid_rows = []
    rejected_rows = []
    
    print('Reading CSV...')
    with open('data/events.csv', 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            row_id = row['row_id']
            amount_str = row['amount'].strip()
            category = row['category'].strip()
            
            if not amount_str:
                rejected_rows.append(row_id)
                continue
            try:
                amount = float(amount_str)
                if amount < 0 or amount > 10000:
                    rejected_rows.append(row_id)
                    continue
            except ValueError:
                rejected_rows.append(row_id)
                continue
            
            if category not in valid_categories:
                rejected_rows.append(row_id)
                continue
            
            valid_rows.append(row)
            if i >= 10:  # Just read first 10 valid rows for testing
                break
    
    print(f'Valid rows: {len(valid_rows)}, Rejected: {len(rejected_rows)}')
    
    # Create table
    create_sql = 'CREATE TABLE IF NOT EXISTS workspace.mlpab0e0197.eventsa45e2a (row_id STRING, account_id STRING, event_time BIGINT, amount DOUBLE, category STRING) USING DELTA'
    result = ws.statement_execution.execute_statement(
        statement=create_sql,
        warehouse_id=warehouse_id,
        wait_timeout='30s'
    )
    print(f'Table created: {result.status.state}')
    
    # Try insert with first row
    if valid_rows:
        row = valid_rows[0]
        row_id = row['row_id']
        account_id = row['account_id']
        event_time = row['event_time']
        amount = row['amount'].strip()
        category = row['category'].strip()
        
        print(f'First row: {row_id}, {account_id}, {event_time}, {amount}, {category}')
        
        insert_sql = f'INSERT INTO workspace.mlpab0e0197.eventsa45e2a SELECT "{row_id}" as row_id, "{account_id}" as account_id, {event_time} as event_time, {amount} as amount, "{category}" as category'
        print(f'SQL: {insert_sql}')
        
        result = ws.statement_execution.execute_statement(
            statement=insert_sql,
            warehouse_id=warehouse_id,
            wait_timeout='30s'
        )
        print(f'Insert result: {result.status.state}')
        print(f'Error: {result.status.error}')

if __name__ == '__main__':
    main()