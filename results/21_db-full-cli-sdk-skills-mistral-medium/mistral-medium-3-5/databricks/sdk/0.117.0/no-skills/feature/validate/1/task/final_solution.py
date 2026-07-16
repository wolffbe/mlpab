#!/usr/bin/env python3

import csv
import json
import os
from databricks.sdk import WorkspaceClient

def main():
    warehouse_id = '4dfab06c923fe3cc'
    schema_name = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpab0e0197')
    
    ws = WorkspaceClient()
    
    # Drop table first to start fresh
    try:
        ws.tables.delete('workspace.mlpab0e0197.eventsa45e2a')
        print('Table dropped')
    except:
        pass
    
    # Read and filter the CSV data
    valid_categories = {'grocery', 'travel', 'salary', 'rent', 'other'}
    valid_rows = []
    rejected_rows = []
    
    with open('data/events.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = row['row_id']
            amount_str = row['amount'].strip()
            category = row['category'].strip()
            
            # Check rule 1: amount is present
            if not amount_str:
                rejected_rows.append(row_id)
                continue
            
            # Check rule 2: amount is within [0, 10000]
            try:
                amount = float(amount_str)
                if amount < 0 or amount > 10000:
                    rejected_rows.append(row_id)
                    continue
            except ValueError:
                rejected_rows.append(row_id)
                continue
            
            # Check rule 3: category is valid
            if category not in valid_categories:
                rejected_rows.append(row_id)
                continue
            
            valid_rows.append(row)
    
    # Write the answers.json file
    os.makedirs('submission', exist_ok=True)
    with open('submission/answers.json', 'w') as f:
        json.dump({"rejected": sorted(rejected_rows)}, f)
    
    print(f"Valid rows: {len(valid_rows)}, Rejected: {len(rejected_rows)}")
    
    # Create table
    create_sql = 'CREATE TABLE IF NOT EXISTS workspace.mlpab0e0197.eventsa45e2a (row_id STRING, account_id STRING, event_time BIGINT, amount DOUBLE, category STRING) USING DELTA'
    result = ws.statement_execution.execute_statement(
        statement=create_sql,
        warehouse_id=warehouse_id,
        wait_timeout='30s'
    )
    print(f'Table created: {result.status.state}')
    
    # Insert data using individual statements
    success_count = 0
    for i, row in enumerate(valid_rows):
        row_id = row['row_id']
        account_id = row['account_id']
        event_time = row['event_time']
        amount = row['amount'].strip()
        category = row['category'].strip()
        
        insert_sql = f'INSERT INTO workspace.mlpab0e0197.eventsa45e2a SELECT "{row_id}" as row_id, "{account_id}" as account_id, {event_time} as event_time, {amount} as amount, "{category}" as category'
        
        # Only print progress every 50 rows to reduce output
        if i % 50 == 0:
            print(f'Inserting row {i+1}/{len(valid_rows)}...')
        
        result = ws.statement_execution.execute_statement(
            statement=insert_sql,
            warehouse_id=warehouse_id,
            wait_timeout='30s'
        )
        
        if result.status.state == 'SUCCESS':
            success_count += 1
        else:
            print(f'Failed at row {row_id}: {result.status.error}')
            break
    
    print(f'Total inserted: {success_count}')
    
    # Verify final count
    count_result = ws.statement_execution.execute_statement(
        statement='SELECT COUNT(*) FROM workspace.mlpab0e0197.eventsa45e2a',
        warehouse_id=warehouse_id,
        wait_timeout='30s'
    )
    
    if count_result.result and count_result.result.data_array:
        final_count = int(count_result.result.data_array[0][0])
        print(f'Final table count: {final_count}')
    
    print("Table creation and data loading complete!")

if __name__ == '__main__':
    main()