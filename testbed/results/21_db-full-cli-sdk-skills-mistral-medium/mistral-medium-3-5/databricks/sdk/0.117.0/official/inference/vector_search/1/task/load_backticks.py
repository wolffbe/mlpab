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

items = []
with open('data/items.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        items.append(row)

print('Loaded', len(items), 'items')

for i, item in enumerate(items):
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
    
    # Use backticks for table name
    sql = 'INSERT INTO `' + table_name + '` VALUES (\'' + item_id + '\', ARRAY[' + embedding_str + '], \'' + label + '\')'
    
    if i % 10 == 0:
        print('Inserting row', i+1, 'of', len(items))
    
    result = wc.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=sql,
    )
    
    if result.status.state.name != 'SUCCEEDED':
        print('Error at row', i+1)
        print(result.status)
        print('SQL:', sql[:200])
        break

print('Done')
