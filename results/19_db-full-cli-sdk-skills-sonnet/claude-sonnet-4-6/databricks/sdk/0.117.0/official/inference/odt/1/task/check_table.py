#!/usr/bin/env python3
"""Check the feature table status and properties."""
import os
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
schema_full = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog_name, schema_name = schema_full.split('.')
full_table_name = f'{catalog_name}.{schema_name}.scored50223c'

# Get table info via Unity Catalog
try:
    table_info = w.tables.get(full_name=full_table_name)
    print(f'Table: {table_info.full_name}')
    print(f'Type: {table_info.table_type}')
    print(f'Data source format: {table_info.data_source_format}')
    print(f'Storage location: {table_info.storage_location}')
    print(f'Comment: {table_info.comment}')
    print(f'Properties: {table_info.properties}')
    print(f'Columns:')
    for col in (table_info.columns or []):
        print(f'  - {col.name}: {col.type_text}, nullable={col.nullable}')
    print(f'Table constraints: {table_info.table_constraints}')
except Exception as e:
    print(f'Error getting table: {e}')

# Try SQL statement execution to verify data
try:
    warehouses = list(w.warehouses.list())
    print(f'\nWarehouses available: {len(warehouses)}')
    for wh in warehouses:
        print(f'  - {wh.name}: {wh.state}')

    if warehouses:
        warehouse_id = warehouses[0].id
        result = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=f"SELECT COUNT(*) as cnt FROM {full_table_name}",
            catalog=catalog_name,
            schema=schema_name
        )
        print(f'\nStatement result: {result.status}')
        if result.result:
            print(f'Row count: {result.result.data_array}')
except Exception as e:
    print(f'Error querying table: {e}')
