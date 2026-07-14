#!/usr/bin/env python3

import csv
import json
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import Disposition, Format

def main():
    # Environment variables
    schema_name = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpab0e0197')
    catalog_name, schema_part = schema_name.split('.')
    warehouse_id = '4dfab06c923fe3cc'
    
    ws = WorkspaceClient()
    
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
    
    print(f"Total rows: {len(valid_rows) + len(rejected_rows)}")
    print(f"Valid rows: {len(valid_rows)}")
    print(f"Rejected rows: {len(rejected_rows)}")
    
    # Create the table
    table_name = 'eventsa45e2a'
    full_table_name = f"{schema_name}.{table_name}"
    
    create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {full_table_name} (
    row_id STRING,
    account_id STRING,
    event_time BIGINT,
    amount DOUBLE,
    category STRING
) 
USING DELTA
COMMENT 'Feature table for valid events data'
"""
    
    print("Creating table...")
    result = ws.statement_execution.execute_statement(
        statement=create_table_sql,
        warehouse_id=warehouse_id,
        catalog=catalog_name,
        schema=schema_part,
        wait_timeout="30s"
    )
    print(f"Table creation result: {result.status.state}")
    
    # Use smaller batches and individual inserts to avoid SQL length limits
    batch_size = 10  # Smaller batch size
    success_count = 0
    
    for i in range(0, len(valid_rows), batch_size):
        batch = valid_rows[i:i + batch_size]
        insert_values = []
        
        for row in batch:
            row_id = row['row_id']
            account_id = row['account_id']
            event_time = row['event_time']
            amount = row['amount'].strip()
            category = row['category'].strip()
            
            # Escape single quotes in strings
            row_id_escaped = row_id.replace("'", "''")
            account_id_escaped = account_id.replace("'", "''")
            category_escaped = category.replace("'", "''")
            
            insert_values.append(f"('{row_id_escaped}', '{account_id_escaped}', {event_time}, {amount}, '{category_escaped}')")
        
        insert_sql = f"""
INSERT INTO {full_table_name} (row_id, account_id, event_time, amount, category)
VALUES {', '.join(insert_values)}
"""
        
        print(f"Inserting batch {i//batch_size + 1} with {len(batch)} rows...")
        result = ws.statement_execution.execute_statement(
            statement=insert_sql,
            warehouse_id=warehouse_id,
            catalog=catalog_name,
            schema=schema_part,
            wait_timeout="30s"
        )
        
        if result.status.state == 'SUCCESS':
            success_count += len(batch)
            print(f"Batch {i//batch_size + 1} succeeded")
        else:
            print(f"Batch {i//batch_size + 1} failed: {result.status.error}")
            break
    
    print(f"Total rows inserted: {success_count}")
    
    # Verify the count
    count_result = ws.statement_execution.execute_statement(
        statement=f'SELECT COUNT(*) as count FROM {full_table_name}',
        warehouse_id=warehouse_id,
        catalog=catalog_name,
        schema=schema_part,
        wait_timeout="30s",
        format=Format.JSON_ARRAY,
        disposition=Disposition.INLINE
    )
    
    if count_result.status.state == 'SUCCESS' and count_result.result.data_array:
        final_count = int(count_result.result.data_array[0][0])
        print(f"Final row count in table: {final_count}")
    
    print("Done!")

if __name__ == "__main__":
    main()