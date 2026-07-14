#!/usr/bin/env python3
import os
import json
import csv
from databricks.sdk import WorkspaceClient

wc = WorkspaceClient()
schema = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpabde8d0a')
table_name = schema + '.itemsffc8a7'

warehouses = list(wc.warehouses.list())
warehouse_id = warehouses[0].id

# Truncate table
print('Truncating table...')
truncate_sql = 'TRUNCATE TABLE ' + table_name
result = wc.statement_execution.execute_statement(
    warehouse_id=warehouse_id,
    statement=truncate_sql,
)
print('Truncate result:', result.status.state)

# Read items
items = []
with open('data/items.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        items.append(row)

print('Loaded', len(items), 'items')

# Insert in batches of 50
batch_size = 50
for i in range(0, len(items), batch_size):
    batch = items[i:i+batch_size]
    values = []
    for item in batch:
        item_id = item['item_id']
        embedding = json.loads(item['embedding'])
        label = item['label']
        
        # Format floats properly
        embedding_str = []
        for x in embedding:
            s = f'{x:.15f}'
            if '.' in s:
                s = s.rstrip('0').rstrip('.')
            embedding_str.append(s)
        
        embedding_str = ', '.join(embedding_str)
        values.append('(\'' + item_id + '\', array(' + embedding_str + '), \'' + label + '\')')
    
    insert_sql = 'INSERT INTO ' + table_name + ' VALUES ' + ', '.join(values)
    
    print('Inserting batch', i//batch_size + 1, 'of', (len(items)+batch_size-1)//batch_size)
    
    result = wc.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=insert_sql,
    )
    
    if result.status.state.name != 'SUCCEEDED':
        print('Error at batch', i//batch_size + 1)
        print(result.status)
        break

print('Done')
