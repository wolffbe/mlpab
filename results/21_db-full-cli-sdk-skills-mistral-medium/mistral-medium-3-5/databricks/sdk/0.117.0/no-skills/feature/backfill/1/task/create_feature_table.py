#!/usr/bin/env python3
"""
Script to create feature table accounts9ad208 with latest revisions from batch files.
"""
import os
import databricks.sdk
from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode
from databricks.sdk.service.workspace import ImportFormat

def main():
    # Initialize workspace client
    ws = databricks.sdk.WorkspaceClient()
    
    # Environment variables
    schema_name = os.environ['MLPAB_DATABRICKS_SCHEMA']  # workspace.mlpab965cb7
    prefix = os.environ['MLPAB_DATABRICKS_PREFIX']  # mlpab965cb7
    
    # Table configuration
    table_name = 'accounts9ad208'
    full_table_name = f'{schema_name}.{table_name}'
    online_table_name = f'{prefix}_{table_name}'
    
    print(f'Schema: {schema_name}')
    print(f'Table name: {full_table_name}')
    print(f'Online table name: {online_table_name}')
    
    # Get warehouse ID
    warehouses = list(ws.warehouses.list())
    warehouse_id = warehouses[0].id if warehouses else None
    print(f'Using warehouse ID: {warehouse_id}')
    
    # Step 1: Upload CSV files to workspace
    print('\n=== Uploading CSV files to workspace ===')
    workspace_path = '/data/accounts_batches'
    
    # Create directory
    ws.workspace.mkdirs(workspace_path)
    
    # Upload batch files
    batch_files = ['data/batch_1.csv', 'data/batch_2.csv', 'data/batch_3.csv']
    for batch_file in batch_files:
        filename = os.path.basename(batch_file)
        workspace_file = f'{workspace_path}/{filename}'
        print(f'Uploading {batch_file} to {workspace_file}')
        with open(batch_file, 'rb') as f:
            content = f.read()
            ws.workspace.upload(workspace_file, content, format=ImportFormat.RAW, overwrite=True)
    
    # Step 2: Create Delta table with merged data (latest revision per row_id)
    print('\n=== Creating Delta table with latest revisions ===')
    
    # Use the CSV format to read from workspace files
    backtick = chr(96)  # Use chr(96) for backtick to avoid bash interpretation
    batch_paths = [
        f'/Workspace{workspace_path}/batch_1.csv',
        f'/Workspace{workspace_path}/batch_2.csv',
        f'/Workspace{workspace_path}/batch_3.csv'
    ]
    
    # Set the current catalog and schema
    ws.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement='USE CATALOG workspace'
    )
    ws.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=f'USE SCHEMA {schema_name}'
    )
    
    # Create the table
    create_table_sql = f"""
CREATE OR REPLACE TABLE {table_name} (
    row_id STRING,
    status STRING,
    balance DOUBLE,
    updated_at BIGINT
) USING DELTA
COMMENT 'Accounts feature table with latest revisions'
"""
    print(f'Creating Delta table')
    result = ws.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=create_table_sql
    )
    print(f'Create table status: {result.status}')
    
    # Insert data with deduplication (keep latest per row_id)
    # CSV files have header row, so skip it with WHERE row_id != 'row_id'
    insert_sql = f"""
INSERT INTO {table_name}
WITH all_batches AS (
  SELECT * FROM csv.{backtick}{batch_paths[0]}{backtick} WHERE row_id != 'row_id'
  UNION ALL
  SELECT * FROM csv.{backtick}{batch_paths[1]}{backtick} WHERE row_id != 'row_id'
  UNION ALL
  SELECT * FROM csv.{backtick}{batch_paths[2]}{backtick} WHERE row_id != 'row_id'
)
SELECT 
  row_id,
  status,
  CAST(balance AS DOUBLE) as balance,
  CAST(updated_at AS BIGINT) as updated_at
FROM all_batches
QUALIFY ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY CAST(updated_at AS BIGINT) DESC) = 1
"""
    print(f'Inserting latest revisions')
    result = ws.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=insert_sql
    )
    print(f'Insert status: {result.status}')
    
    # Verify the table
    count_sql = f'SELECT COUNT(*) as count FROM {full_table_name}'
    print(f'\n=== Verifying table ===')
    result = ws.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=count_sql
    )
    print(f'Table created with {result.result.row_count} rows')
    
    # Check some sample data
    sample_sql = f'SELECT * FROM {full_table_name} LIMIT 5'
    sample_result = ws.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=sample_sql
    )
    if sample_result.result:
        print(f'Sample data: {sample_result.result.data_array}')
    
    # Step 3: Publish as online table for low-latency access
    print('\n=== Publishing as online table ===')
    
    # First, check if online store exists
    online_store_name = f'{prefix}_store'
    try:
        stores = list(ws.feature_store.list_online_stores())
        store_names = [s.name for s in stores]
        print(f'Existing online stores: {store_names}')
        
        if online_store_name not in store_names:
            # Create online store
            print(f'Creating online store: {online_store_name}')
            from databricks.sdk.service.ml import OnlineStore
            # OnlineStore requires name and capacity
            online_store = OnlineStore(name=online_store_name, capacity='SMALL')
            ws.feature_store.create_online_store(online_store)
            print(f'Created online store: {online_store_name}')
        else:
            print(f'Using existing online store: {online_store_name}')
    except Exception as e:
        print(f'Error with online stores: {e}')
    
    # Publish the table
    publish_spec = PublishSpec(
        online_store=online_store_name,
        online_table_name=online_table_name,
        publish_mode=PublishSpecPublishMode.CONTINUOUS
    )
    
    print(f'Publishing table {full_table_name} as online table {online_table_name}')
    try:
        response = ws.feature_store.publish_table(
            source_table_name=full_table_name,
            publish_spec=publish_spec
        )
        print(f'Publish response: {response}')
    except Exception as e:
        print(f'Error publishing: {e}')
    
    print('\n=== Done ===')
    print(f'Feature table {full_table_name} created')

if __name__ == '__main__':
    main()
