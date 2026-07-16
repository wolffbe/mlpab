#!/usr/bin/env python3

import csv
import os
from databricks.sdk import WorkspaceClient

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
            if len(valid_rows) >= 6:  # Just get first 6 for testing
                break
    
    print(f"Valid rows: {len(valid_rows)}")
    
    # Create the table
    table_name = 'eventsa45e2a'
    full_table_name = schema_name + '.' + table_name
    
    # Use UNION ALL approach with string concatenation
    batch_size = 3
    
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
        
        print(f"Generated SQL for batch {i//batch_size + 1}:")
        print(repr(insert_sql))
        print("Actual SQL:")
        print(insert_sql)
        print()
        
        result = ws.statement_execution.execute_statement(
            statement=insert_sql,
            warehouse_id=warehouse_id,
            catalog=catalog_name,
            schema=schema_part,
            wait_timeout="30s"
        )
        
        print(f"Result: {result.status.state}")
        if result.status.error:
            print(f"Error: {result.status.error}")
        print()

if __name__ == "__main__":
    main()