#!/usr/bin/env python3
import os
from databricks.sdk import WorkspaceClient

wc = WorkspaceClient()
schema = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpabde8d0a')
table_name = schema + '.itemsffc8a7'

warehouses = list(wc.warehouses.list())
warehouse_id = warehouses[0].id

# Try different SQL statements
statements = [
    'INSERT INTO ' + table_name + ' VALUES (\'I0000\', ARRAY[1.0], \'c5\')',
    'INSERT INTO ' + table_name + ' VALUES (\'I0000\', ARRAY[1.0, 2.0], \'c5\')',
    'INSERT INTO ' + table_name + ' VALUES (\'I0000\', ARRAY[1.0, 2.0, 3.0], \'c5\')',
]

for i, sql in enumerate(statements):
    print('Trying statement', i+1, ':', sql[:80])
    result = wc.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=sql,
    )
    print('Result:', result.status.state)
    if result.status.state.name != 'SUCCEEDED':
        print('Error:', result.status.error)
    print()
