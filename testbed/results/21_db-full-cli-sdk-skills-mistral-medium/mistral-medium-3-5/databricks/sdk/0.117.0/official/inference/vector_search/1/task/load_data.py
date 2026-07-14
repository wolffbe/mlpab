#!/usr/bin/env python3
"""
Load data into Databricks table using SDK.
This script reads CSV files locally and uses SDK to execute SQL on the platform.
"""
import os
import json
import csv
from databricks.sdk import WorkspaceClient

def main():
    wc = WorkspaceClient()
    schema = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpabde8d0a')
    table_name = f'{schema}.itemsffc8a7'
    
    warehouses = list(wc.warehouses.list())
    warehouse_id = warehouses[0].id
    
    # Read items
    items = []
    with open('data/items.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)
    
    print(f'Loaded {len(items)} items')
    
    # Insert items one by one
    for i, item in enumerate(items):
        item_id = item['item_id']
        embedding = json.loads(item['embedding'])
        label = item['label']
        # Format embedding as ARRAY[...] with proper float formatting
        embedding_str = ', '.join([f'{x:.10f}' for x in embedding])
        
        insert_sql = f"INSERT INTO {table_name} (item_id, embedding, label) VALUES ('{item_id}', ARRAY[{embedding_str}], '{label}')"
        
        if i % 10 == 0:
            print(f'Inserting row {i+1}/{len(items)}...')
        
        result = wc.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=insert_sql,
        )
        
        if result.status.state.name != 'SUCCEEDED':
            print(f'Error at row {i+1}: {result.status}')
            print(f'SQL: {insert_sql[:200]}...')
            break
    
    print('Data loaded')

if __name__ == '__main__':
    main()
