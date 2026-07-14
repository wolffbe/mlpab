#!/usr/bin/env python3

import csv
import json
import os
from databricks.sdk import WorkspaceClient

def main():
    # Environment variables
    schema_name = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpab0e0197')
    catalog_name, schema_part = schema_name.split('.')
    warehouse_id = '4dfab06c923fe3cc'
    
    ws = WorkspaceClient()
    
    # Drop table first to start fresh
    try:
        ws.tables.delete('workspace.mlpab0e0197.eventsa45e2a')
        print('Table dropped')
    except Exception as e:
        print(f'Table drop error: {e}')
    
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
    
    # Create the table with simple SQL
    table_name = 'eventsa45e2a'
    full_table_name = schema_name + '.' + table_name
    
    create_table_sql = 'CREATE TABLE IF NOT EXISTS ' + full_table_name + ' (row_id STRING, account_id STRING, event_time BIGINT, amount DOUBLE, category STRING) USING DELTA'
    
    print("Creating table...")
    result = ws.statement_execution.execute_statement(
        statement=create_table_sql,
        warehouse_id=warehouse_id,
        catalog=catalog_name,
        schema=schema_part,
        wait_timeout="30s"
    )
    print(f"Table creation result: {result.status.state}")
    
    # Test a simple insert first
    test_sql = 'INSERT INTO ' + full_table_name + ' SELECT "TEST001" as row_id, "TESTACCT" as account_id, 1234567890000 as event_time, 100.0 as amount, "other" as category'
    print("Testing simple insert...")
    result = ws.statement_execution.execute_statement(
        statement=test_sql,
        warehouse_id=warehouse_id,
        catalog=catalog_name,
        schema=schema_part,
        wait_timeout="30s"
    )
    print(f"Test insert result: {result.status.state}")
    if result.status.error:
        print(f"Test insert error: {result.status.error}")
    
    # Use UNION ALL approach with string concatenation
    batch_size = 5  # Start with small batches
    success_count = 0
    
    for i in range(0, len(valid_rows), batch_size):
        batch = valid_rows[i:i + batch_size]
        
        # Build the UNION ALL query using string concatenation
        union_parts = []
        for row in batch:
            row_id = row['row_id']
            account_id = row['account_id']
            event_time = row['event_time']
            amount = row['amount'].strip()
            category = row['category'].strip()
            
            # Use string concatenation with double quotes for SQL strings
            part = "SELECT " + '"' + row_id + '"' + " as row_id, " + '"' + account_id + '"' + " as account_id, " + \
                   event_time + " as event_time, " + amount + " as amount, " + '"' + category + '"' + " as category"
            union_parts.append(part)
        
        # Join with UNION ALL
        union_sql = " UNION ALL ".join(union_parts)
        insert_sql = "INSERT INTO " + full_table_name + " " + union_sql
        
        print(f"Inserting batch {i//batch_size + 1} with {len(batch)} rows...")
        print(f"SQL: {insert_sql[:100]}...")  # Print first 100 chars of SQL
        
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
        statement="SELECT COUNT(*) as count FROM " + full_table_name,
        warehouse_id=warehouse_id,
        catalog=catalog_name,
        schema=schema_part,
        wait_timeout="30s"
    )
    
    print(f"Query result: {count_result.status.state}")
    
    print("Done!")

if __name__ == "__main__":
    main()